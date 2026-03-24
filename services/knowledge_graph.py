"""
knowledge_graph.py — KG Automatisé
═══════════════════════════════════

VISION : Le KG ne doit jamais être écrit à la main.
Il se construit tout seul depuis 3 sources :

  Source 1 — Front-matter des .md (KB de règles)
      sql_injection.md contient :
          kg_nodes: [SQL_Injection, Resource_Leak]
          kg_relations:
            - [SQL_Injection, FIXED_BY, PreparedStatement]
      → KG lit ces métadonnées → construit les nœuds et arêtes

  Source 2 — AST du code projet (via code_parser.py)
      Auth.java contient class TokenManager, method authenticate()
      → KG crée automatiquement :
          nœud "Auth.java::TokenManager"   (type: entity_class)
          nœud "Auth.java::authenticate"   (type: entity_method)
          arête TokenManager → HAS_METHOD → authenticate
          arête Auth.java    → CONTAINS   → TokenManager

  Source 3 — Liaison sémantique (heuristiques + LLM optionnel)
      authenticate(username, password) → HANDLES → Authentication
      Authentication → IS_CONCEPT_OF → Security
      → N-hop depuis Auth.java trouve les règles Security

TRAVERSAL N-HOP :
  Auth.java modifié
      → NetworkX : impacte UserController.java
      → KG entités : UserController::login → HANDLES → Authentication
      → KG concepts : Authentication → IS_CONCEPT_OF → Security
      → ChromaDB : chercher règles Security + Authentication
  Le LLM reçoit des règles qu'il n'aurait jamais trouvées par recherche textuelle.
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)
# KGNode
@dataclass
class KGNode:
    id:          str
    node_type:   str                        # vulnerability|fix|concept|entity_class|entity_method|file
    severity:    str | None  = None
    languages:   list[str]   = field(default_factory=list)
    kb_queries:  dict[str, str] = field(default_factory=dict)
    source_file: str | None  = None
    extra:       dict        = field(default_factory=dict)



# KGBuilder — Sources 1 & 2


class KGBuilder:

    def __init__(self, graph: nx.DiGraph):
        self._g = graph

    # ── Source 1 : depuis les .md ─────────────────────────────────────────────

    def build_from_kb(self, kb_dir: Path) -> int:
        """
        Lit les .md et extrait les sections KG du front-matter YAML.

        Format attendu dans chaque .md :

            kg_nodes:
              - name: SQL_Injection
                type: vulnerability
                severity: CRITICAL
                languages: [java, python]
                kb_queries:
                  java:   "SQL injection PreparedStatement JDBC java"
                  python: "SQL injection cursor execute parameterized python"

            kg_relations:
              - [SQL_Injection, FIXED_BY, PreparedStatement]
              - [SQL_Injection, OFTEN_WITH, Resource_Leak]

            pattern_map:
              java:
                "executeQuery": SQL_Injection
                "ResultSet":    Resource_Leak
              python:
                "cursor.execute": SQL_Injection

        Si ces sections sont absentes → .md ignoré pour le KG (rétro-compatible).
        """
        if not kb_dir.exists():
            logger.warning("KB dir introuvable : %s", kb_dir)
            return 0

        md_files = list(kb_dir.rglob("*.md"))
        total_nodes = 0
        total_edges = 0
        pattern_map: dict[str, dict[str, str]] = {}

        for md_path in md_files:
            try:
                content = md_path.read_text(encoding="utf-8")
                front   = self._parse_front_matter(content)

                for node_def in front.get("kg_nodes", []):
                    if isinstance(node_def, str):
                        node_def = {"name": node_def}
                    nid = node_def.get("name", "")
                    if not nid:
                        continue
                    self._add_node(KGNode(
                        id          = nid,
                        node_type   = node_def.get("type", "concept"),
                        severity    = node_def.get("severity"),
                        languages   = node_def.get("languages", []),
                        kb_queries  = node_def.get("kb_queries", {}),
                        source_file = md_path.name,
                    ))
                    total_nodes += 1

                for rel in front.get("kg_relations", []):
                    if isinstance(rel, list) and len(rel) == 3:
                        subj, relation, obj = rel
                        self._ensure_node(subj)
                        self._ensure_node(obj)
                        self._g.add_edge(subj, obj,
                                         relation=relation,
                                         source_file=md_path.name)
                        total_edges += 1

                for lang, patterns in front.get("pattern_map", {}).items():
                    if lang not in pattern_map:
                        pattern_map[lang] = {}
                    if isinstance(patterns, dict):
                        pattern_map[lang].update(patterns)

            except Exception as e:
                logger.debug("Erreur KG depuis %s : %s", md_path.name, e)

        self._g.graph["pattern_map"] = pattern_map
        logger.info("KG depuis KB : +%d nœuds, +%d arêtes (%d .md)",
                    total_nodes, total_edges, len(md_files))
        return total_nodes

    def build_from_kb_single_file(self, md_path: Path) -> int:
        """
        Recharge UN SEUL fichier .md dans le KG existant.
        Utilisé par FeedbackProcessor et LearningAgent après promotion d'un fix.

        Complète le pattern_map existant au lieu de le remplacer.
        Retourne le nombre de nœuds ajoutés.
        """
        if not md_path.exists():
            logger.warning("build_from_kb_single_file : fichier introuvable %s", md_path)
            return 0

        total_nodes = 0
        total_edges = 0
        pattern_map = self._g.graph.get("pattern_map", {})

        try:
            content = md_path.read_text(encoding="utf-8")
            front   = self._parse_front_matter(content)

            for node_def in front.get("kg_nodes", []):
                if isinstance(node_def, str):
                    node_def = {"name": node_def}
                nid = node_def.get("name", "")
                if not nid:
                    continue
                self._add_node(KGNode(
                    id          = nid,
                    node_type   = node_def.get("type", "concept"),
                    severity    = node_def.get("severity"),
                    languages   = node_def.get("languages", []),
                    kb_queries  = node_def.get("kb_queries", {}),
                    source_file = md_path.name,
                ))
                total_nodes += 1

            for rel in front.get("kg_relations", []):
                if isinstance(rel, list) and len(rel) == 3:
                    subj, relation, obj = rel
                    self._ensure_node(subj)
                    self._ensure_node(obj)
                    self._g.add_edge(subj, obj,
                                     relation=relation,
                                     source_file=md_path.name)
                    total_edges += 1

            # Fusionner le pattern_map sans écraser les patterns existants
            for lang, patterns in front.get("pattern_map", {}).items():
                if lang not in pattern_map:
                    pattern_map[lang] = {}
                if isinstance(patterns, dict):
                    pattern_map[lang].update(patterns)

            self._g.graph["pattern_map"] = pattern_map

            logger.debug(
                "KG single_file reload : %s → +%d nœuds +%d arêtes",
                md_path.name, total_nodes, total_edges,
            )

        except Exception as e:
            logger.debug("build_from_kb_single_file erreur %s : %s", md_path.name, e)

        return total_nodes



    def build_from_project_indexer(self, project_indexer) -> int:
        """
        Construit les nœuds AST depuis project_indexer.context.files.

        POURQUOI c'est mieux que build_from_ast() :
          1. Zéro re-parsing — les entités sont déjà calculées
          2. Criticité incluse — project_indexer a déjà calculé
             combien de fichiers dépendent de chaque fichier
          3. Imports déjà résolus — dependency_graph a résolu
             les imports ambigus que code_parser ne peut pas résoudre
          4. Cohérence garantie — KG et projet partagent exactement
             les mêmes données (pas deux parsings divergents)

        Structure de project_indexer.context.files :
          {
            "/path/UserService.java": {
              "entities":    [{"name": "login", "type": "method",
                               "parameters": ["username","password"], ...}],
              "imports":     ["UserRepository", "EmailService"],
              "language":    "java",
              "criticality": 3,   ← nombre de fichiers qui en dépendent
            }
          }
        """
        if not project_indexer or not project_indexer.context:
            logger.warning("project_indexer absent — fallback sur build_from_ast")
            return 0

        context_files = project_indexer.context.files
        total = 0

        for file_path_str, file_info in context_files.items():
            try:
                file_path   = Path(file_path_str)
                file_id     = file_path.name
                language    = file_info.get("language", "unknown")
                entities    = file_info.get("entities", [])
                imports     = file_info.get("imports", [])
                criticality = file_info.get("criticality", 0)

                # Nœud fichier — avec criticité depuis project_indexer
                self._add_node(KGNode(
                    id          = file_id,
                    node_type   = "file",
                    languages   = [language],
                    source_file = file_path_str,
                    extra       = {
                        "full_path":   file_path_str,
                        "criticality": criticality,   # ← BONUS vs build_from_ast
                    },
                ))

                current_class = None
                for entity in entities:
                    # Les entités de project_indexer sont des dicts
                    name     = entity.get("name", "")
                    etype    = entity.get("type", "")
                    params   = entity.get("parameters", [])

                    if not name:
                        continue

                    node_type = ("entity_class"
                                 if etype in ("class", "interface", "enum")
                                 else "entity_method")
                    node_id   = f"{file_id}::{name}"

                    self._add_node(KGNode(
                        id          = node_id,
                        node_type   = node_type,
                        languages   = [language],
                        source_file = file_path_str,
                        extra       = {
                            "entity_type": etype,
                            "parameters":  params,
                            "criticality": criticality,  # ← hérité du fichier
                        },
                    ))

                    # Arêtes structurelles
                    self._g.add_edge(file_id, node_id,
                                     relation="CONTAINS",
                                     source_file=file_path_str)
                    self._g.add_edge(node_id, file_id,
                                     relation="DEFINED_IN",
                                     source_file=file_path_str)

                    if etype in ("class", "interface"):
                        current_class = node_id
                    elif etype in ("method", "function", "constructor") and current_class:
                        self._g.add_edge(current_class, node_id,
                                         relation="HAS_METHOD",
                                         source_file=file_path_str)
                    total += 1

                # Arêtes d'imports — depuis project_indexer (déjà résolus)
                for module in imports:
                    module_base = module.split(".")[-1] if "." in module else module
                    for node in list(self._g.nodes()):
                        if (self._g.nodes[node].get("node_type") == "file"
                                and module_base.lower() in node.lower()):
                            self._g.add_edge(file_id, node,
                                             relation="IMPORTS",
                                             source_file=file_path_str)
                            break

            except Exception as e:
                logger.debug("Erreur project_indexer → KG %s : %s",
                             Path(file_path_str).name, e)

        logger.info("KG depuis project_indexer : +%d entités depuis %d fichiers",
                    total, len(context_files))
        return total

    # ── Source 2B : depuis dependency_graph NetworkX ─────────────────────────

    def build_from_dependency_graph(self, nx_graph) -> int:
        """
        Importe les arêtes IMPORTS directement depuis le graphe NetworkX.

        POURQUOI c'est mieux que recréer depuis les imports bruts :
          1. NetworkX a déjà résolu les imports ambigus
             (ex: "from utils import execute_query" → résolu vers utils.py)
          2. Les arêtes sont vérifiées (fichier source ET destination existent)
          3. Évite les faux positifs (imports de librairies externes ignorés)

        Complète build_from_project_indexer() qui crée les nœuds mais
        avec des arêtes IMPORTS moins fiables (juste les noms de modules).
        """
        if nx_graph is None:
            return 0

        edges_added = 0
        for src, dst, data in nx_graph.edges(data=True):
            # Les nœuds NetworkX ont le format "file:/chemin/absolu/Fichier.java"
            src_name = Path(str(src).replace("file:", "")).name
            dst_name = Path(str(dst).replace("file:", "")).name

            if not src_name or not dst_name:
                continue

            # Créer les nœuds s'ils n'existent pas encore
            self._ensure_node(src_name)
            self._ensure_node(dst_name)

            # Ajouter l'arête IMPORTS depuis NetworkX
            # (remplace/complète les arêtes créées par build_from_project_indexer)
            if not self._g.has_edge(src_name, dst_name):
                self._g.add_edge(src_name, dst_name,
                                 relation="IMPORTS",
                                 source_file=str(src))
                edges_added += 1

        logger.info("KG depuis dependency_graph : +%d arêtes IMPORTS", edges_added)
        return edges_added

    # ── Source 2C : fallback code_parser (si project_indexer absent) ─────────

    def build_from_ast(self, project_path: Path) -> int:
        """
        Fallback — utilisé UNIQUEMENT si project_indexer n'est pas disponible.
        Préférer build_from_project_indexer() + build_from_dependency_graph().

        Inconvénients vs build_from_project_indexer :
          - Double parsing (lent)
          - Pas de criticité
          - Imports non résolus
          - Peut diverger des données de project_indexer
        """
        try:
            from services.code_parser import parser as code_parser
        except ImportError:
            logger.warning("code_parser non disponible")
            return 0

        extensions = {".py", ".java", ".js", ".ts", ".jsx", ".tsx"}
        excluded   = {"venv", "__pycache__", "node_modules", ".git",
                      "dist", "build", "target", ".pytest_cache"}

        files = [
            f for f in project_path.rglob("*")
            if f.suffix in extensions
            and not any(ex in f.parts for ex in excluded)
        ]

        total = 0
        for file_path in files:
            try:
                parsed = code_parser.parse_file(file_path)
                if "error" in parsed:
                    continue

                file_id  = file_path.name
                language = parsed.get("language", "unknown")
                entities = parsed.get("entities", [])
                imports  = parsed.get("imports",  [])

                self._add_node(KGNode(
                    id=file_id, node_type="file",
                    languages=[language], source_file=str(file_path),
                    extra={"full_path": str(file_path)},
                ))

                current_class = None
                for entity in entities:
                    name   = entity.name if hasattr(entity, "name") else entity.get("name", "")
                    etype  = entity.type if hasattr(entity, "type") else entity.get("type", "")
                    params = (entity.parameters if hasattr(entity, "parameters")
                              else entity.get("parameters", []))
                    if not name:
                        continue

                    node_type = ("entity_class"
                                 if etype in ("class", "interface", "enum")
                                 else "entity_method")
                    node_id = f"{file_id}::{name}"
                    self._add_node(KGNode(
                        id=node_id, node_type=node_type,
                        languages=[language], source_file=str(file_path),
                        extra={"entity_type": etype, "parameters": params},
                    ))
                    self._g.add_edge(file_id, node_id, relation="CONTAINS",
                                     source_file=str(file_path))
                    self._g.add_edge(node_id, file_id, relation="DEFINED_IN",
                                     source_file=str(file_path))
                    if etype in ("class", "interface"):
                        current_class = node_id
                    elif etype in ("method", "function", "constructor") and current_class:
                        self._g.add_edge(current_class, node_id,
                                         relation="HAS_METHOD",
                                         source_file=str(file_path))
                    total += 1

                for imp in imports:
                    module      = (imp.module if hasattr(imp, "module")
                                   else imp.get("module", ""))
                    module_base = module.split(".")[-1] if "." in module else module
                    for node in list(self._g.nodes()):
                        if (self._g.nodes[node].get("node_type") == "file"
                                and module_base.lower() in node.lower()):
                            self._g.add_edge(file_id, node, relation="IMPORTS",
                                             source_file=str(file_path))
                            break
                    total += 0  # don't double count

            except Exception as e:
                logger.debug("Erreur AST fallback %s : %s", file_path.name, e)

        logger.info("KG fallback AST : +%d entités depuis %d fichiers",
                    total, len(files))
        return total

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_node(self, node: KGNode) -> None:
        attrs = {
            "node_type":   node.node_type,
            "severity":    node.severity,
            "languages":   node.languages,
            "kb_queries":  node.kb_queries,
            "source_file": node.source_file,
            **node.extra,
        }
        if self._g.has_node(node.id):
            self._g.nodes[node.id].update(attrs)
        else:
            self._g.add_node(node.id, **attrs)

    def _ensure_node(self, node_id: str) -> None:
        if not self._g.has_node(node_id):
            self._g.add_node(node_id, node_type="concept",
                             severity=None, languages=[],
                             kb_queries={}, source_file=None)

    @staticmethod
    def _parse_front_matter(content: str) -> dict:
        if not content.startswith("---"):
            return {}
        end = content.find("---", 3)
        if end == -1:
            return {}
        yaml_str = content[3:end].strip()
        try:
            import yaml
            return yaml.safe_load(yaml_str) or {}
        except Exception:
            result = {}
            for line in yaml_str.splitlines():
                if ":" in line and not line.strip().startswith("-"):
                    key, _, val = line.partition(":")
                    result[key.strip()] = val.strip()
            return result


# ─────────────────────────────────────────────────────────────────────────────
# SemanticLinker — Source 3 : liaison sémantique
# ─────────────────────────────────────────────────────────────────────────────

class SemanticLinker:
    """
    Identifie l'intention sémantique de chaque méthode et crée des liens
    dans le KG vers des concepts universels (Authentication, Pagination...).

    authenticate(username, password)
        → HANDLES → Authentication
        → Authentication IS_CONCEPT_OF Security
        → N-hop trouve les règles Security

    Utilise les heuristiques (rapide) + LLM optionnel si heuristiques vides.
    Appelé UNE SEULE FOIS à l'indexation — résultats persistés dans le JSON.
    """

    UNIVERSAL_CONCEPTS = {
        "Authentication", "Authorization", "Serialization", "Deserialization",
        "Encryption", "Hashing", "TokenManagement", "SessionManagement",
        "DatabaseAccess", "FileIO", "NetworkIO", "Caching", "Logging",
        "Validation", "ErrorHandling", "Pagination", "RateLimiting",
        "UserManagement", "EmailSending",
    }

    CONCEPT_DOMAINS = {
        "Authentication": "Security",   "Authorization":  "Security",
        "TokenManagement":"Security",   "Hashing":        "Security",
        "Encryption":     "Security",   "DatabaseAccess": "DataManagement",
        "Pagination":     "Performance","Caching":        "Performance",
        "Serialization":  "DataManagement", "FileIO":     "IO",
        "NetworkIO":      "IO",         "Logging":        "Observability",
        "Validation":     "Quality",    "ErrorHandling":  "Quality",
        "EmailSending":   "Communication",
    }

    # ── Queries ChromaDB par concept ──────────────────────────────────────────
    # Ces queries sont injectées dans les nœuds concept/domain du KG.
    # Quand n_hop_retrieval traverse entity → HANDLES → concept → IS_CONCEPT_OF → domain,
    # il trouve ces queries et les soumet à ChromaDB.
    # AVANT : kb_queries={} → 0 n-hop résultats
    # APRÈS : kb_queries peuplés → n-hop retourne des règles pertinentes
    CONCEPT_KB_QUERIES: dict[str, dict[str, str]] = {
        "Authentication": {
            "java":       "authentication java password hashing BCrypt login security",
            "python":     "authentication python password hashing bcrypt login security",
            "javascript": "authentication javascript JWT token login security",
            "typescript": "authentication typescript JWT token login security",
        },
        "Authorization": {
            "java":       "authorization java roles permissions access control security",
            "python":     "authorization python roles permissions access control",
        },
        "Hashing": {
            "java":       "password hashing java BCrypt plain text insecure MD5 SHA",
            "python":     "password hashing python bcrypt argon2 plain text insecure",
        },
        "DatabaseAccess": {
            "java":       "SQL injection PreparedStatement JDBC java resource leak",
            "python":     "SQL injection cursor execute parameterized python database",
        },
        "Serialization": {
            "java":       "deserialization security java unsafe object serialization",
            "python":     "pickle deserialization security python unsafe",
        },
        "FileIO": {
            "java":       "file resource leak java FileWriter close try-with-resources",
            "python":     "file resource leak python context manager with open",
        },
        "ErrorHandling": {
            "java":       "exception swallowing java empty catch block poor error handling",
            "python":     "exception swallowing python bare except pass error handling",
        },
        "Pagination": {
            "java":       "pagination java no limit SELECT all users memory performance",
            "python":     "pagination python no limit query all rows memory performance",
        },
        "Caching": {
            "java":       "caching java performance Redis TTL invalidation",
            "python":     "caching python performance Redis TTL invalidation",
        },
        "Validation": {
            "java":       "input validation java sanitize user input injection",
            "python":     "input validation python sanitize user input injection",
        },
        "Logging": {
            "java":       "logging java sensitive data credentials password log",
            "python":     "logging python sensitive data credentials password log",
        },
        "Encryption": {
            "java":       "encryption java weak algorithm AES RSA key management",
            "python":     "encryption python weak algorithm AES RSA key management",
        },
        "TokenManagement": {
            "java":       "JWT token java expiration signature verification security",
            "python":     "JWT token python expiration signature verification security",
        },
        "NetworkIO": {
            "java":       "network java SSL TLS insecure connection timeout",
            "python":     "network python SSL TLS insecure connection timeout",
        },
    }

    DOMAIN_KB_QUERIES: dict[str, dict[str, str]] = {
        "Security": {
            "java":       "security vulnerability java OWASP injection authentication SQL",
            "python":     "security vulnerability python OWASP injection authentication SQL",
        },
        "Performance": {
            "java":       "performance java N+1 query pagination memory leak database",
            "python":     "performance python N+1 query pagination memory database",
        },
        "Quality": {
            "java":       "code quality java SRP clean code exception error handling",
            "python":     "code quality python SRP clean code exception error handling",
        },
        "DataManagement": {
            "java":       "data management java SQL resource leak connection pool",
            "python":     "data management python SQL resource leak connection",
        },
        "IO": {
            "java":       "IO resource leak java stream close try-with-resources",
            "python":     "IO resource leak python context manager close file",
        },
        "Observability": {
            "java":       "logging java audit trail sensitive data masking",
            "python":     "logging python audit trail sensitive data masking",
        },
    }

    HEURISTIC_RULES = [
        (["login","authenticate","auth","signin","verify_token",
          "check_password","validate_user"],                "Authentication"),
        (["authorize","permission","role","access_control"], "Authorization"),
        (["token","jwt","session","refresh_token"],          "TokenManagement"),
        (["password","passwd","hash_password","bcrypt"],     "Hashing"),
        (["encrypt","decrypt","cipher","aes","rsa"],         "Encryption"),
        (["query","execute","findby","select","insert",
          "update","delete","save","fetch","cursor"],        "DatabaseAccess"),
        (["paginate","page","limit","offset","getall"],      "Pagination"),
        (["cache","redis","memcache","ttl"],                 "Caching"),
        (["serialize","deserialize","marshal","to_json",
          "from_json","pickle"],                            "Serialization"),
        (["file","read_file","write_file","open",
          "filewriter","filereader"],                       "FileIO"),
        (["log","logger","audit","trace"],                  "Logging"),
        (["validate","validation","check","verify"],        "Validation"),
        (["error","exception","catch","handle_error"],      "ErrorHandling"),
        (["email","mail","smtp","send_message"],            "EmailSending"),
    ]

    def __init__(self, graph: nx.DiGraph, llm=None):
        self._g    = graph
        self._llm  = llm
        self._cache: dict[str, list[str]] = {}

    def link_entities(self, entities: list, file_path: str, language: str) -> int:
        """Crée les liens sémantiques pour toutes les entités d'un fichier."""
        links = 0
        file_id = Path(file_path).name

        for entity in entities:
            name  = entity.name if hasattr(entity, "name") else entity.get("name", "")
            etype = entity.type if hasattr(entity, "type") else entity.get("type", "")
            doc   = (entity.docstring if hasattr(entity, "docstring")
                     else entity.get("docstring", "")) or ""
            params = (entity.parameters if hasattr(entity, "parameters")
                      else entity.get("parameters", [])) or []

            if etype not in ("class", "method", "function", "interface"):
                continue

            node_id = f"{file_id}::{name}"
            if not self._g.has_node(node_id):
                continue

            # Heuristiques enrichies avec paramètres + return_type + décorateurs
            ret_type   = (entity.return_type if hasattr(entity, "return_type")
                          else entity.get("return_type") or "")
            decorators = (entity.decorators if hasattr(entity, "decorators")
                          else entity.get("decorators") or [])
            concepts = self._detect_heuristic(
                name, params, doc,
                return_type = ret_type,
                decorators  = decorators,
            )

            # LLM si heuristiques vides
            if not concepts and self._llm:
                cache_key = hashlib.md5(f"{file_id}{name}".encode()).hexdigest()[:8]
                if cache_key not in self._cache:
                    self._cache[cache_key] = self._detect_llm(name, params, doc, language)
                concepts = self._cache.get(cache_key, [])

            for concept in concepts:
                # Récupérer les kb_queries pour ce concept (peuple le nœud
                # pour que n_hop_retrieval trouve des queries ChromaDB)
                concept_queries = self.CONCEPT_KB_QUERIES.get(concept, {})

                if not self._g.has_node(concept):
                    self._g.add_node(concept, node_type="concept",
                                     severity=None, languages=list(concept_queries.keys()),
                                     kb_queries=concept_queries, source_file=None)
                elif concept_queries:
                    # Enrichir un nœud existant si ses kb_queries sont vides
                    existing = self._g.nodes[concept]
                    if not existing.get("kb_queries"):
                        existing["kb_queries"] = concept_queries
                        existing["languages"]  = list(concept_queries.keys())

                self._g.add_edge(node_id, concept,
                                 relation="HANDLES", source_file=file_path)

                parent = self.CONCEPT_DOMAINS.get(concept)
                if parent:
                    domain_queries = self.DOMAIN_KB_QUERIES.get(parent, {})
                    if not self._g.has_node(parent):
                        self._g.add_node(parent, node_type="domain",
                                         severity=None,
                                         languages=list(domain_queries.keys()),
                                         kb_queries=domain_queries,
                                         source_file=None)
                    elif domain_queries:
                        existing = self._g.nodes[parent]
                        if not existing.get("kb_queries"):
                            existing["kb_queries"] = domain_queries
                            existing["languages"]  = list(domain_queries.keys())

                    if not self._g.has_edge(concept, parent):
                        self._g.add_edge(concept, parent,
                                         relation="IS_CONCEPT_OF",
                                         source_file=None)
                links += 1

        return links

    def _detect_heuristic(
        self,
        name: str,
        params: list,
        doc: str,
        return_type: str = "",
        decorators: list | None = None,
    ) -> list[str]:
        """
        Fix Problème 4 — Heuristiques enrichies avec plus de contexte.

        Avant : seulement le nom de la méthode
            authenticate → Authentication  ✅
            checkCredentials → rien         ❌ (raté)
            verifyIdentity → rien           ❌ (raté)

        Après : nom + paramètres + return type + docstring + décorateurs
            checkCredentials(username, password) → "password" dans params → Hashing
            findUsers(page, size) → "page" dans params → Pagination
            getConnection() → return_type "Connection" → DatabaseAccess
            @login_required → decorator → Authentication
        """
        decorators = decorators or []
        dec_str = " ".join(str(d) for d in decorators).lower()

        # Combiner toutes les sources d'information
        combined = " ".join([
            name.lower(),
            " ".join(str(p) for p in params).lower(),
            (doc or "").lower(),
            (return_type or "").lower(),
            dec_str,
        ])

        detected = []
        for kws, concept in self.HEURISTIC_RULES:
            if any(kw in combined for kw in kws):
                detected.append(concept)

        # Règles supplémentaires sur les paramètres spécifiques
        params_lower = [str(p).lower() for p in params]

        # Si les params contiennent "password" ou "secret" → Hashing
        if any(p in ("password", "passwd", "secret", "credential")
               for p in params_lower):
            if "Hashing" not in detected:
                detected.append("Hashing")

        # Si les params contiennent "page" + "size" ou "limit" + "offset" → Pagination
        if (("page" in params_lower or "offset" in params_lower)
                and ("size" in params_lower or "limit" in params_lower)):
            if "Pagination" not in detected:
                detected.append("Pagination")

        # Si return_type est Connection/ResultSet/Cursor → DatabaseAccess
        if return_type and any(t in (return_type or "")
                               for t in ["Connection", "ResultSet", "Cursor",
                                         "Session", "Transaction"]):
            if "DatabaseAccess" not in detected:
                detected.append("DatabaseAccess")

        # Décorateurs spéciaux
        if any(d in dec_str for d in ["login_required", "authenticated",
                                       "jwt_required", "auth_required"]):
            if "Authentication" not in detected:
                detected.append("Authentication")

        return detected

    def _detect_llm(self, name: str, params: list, doc: str, language: str) -> list[str]:
        if not self._llm:
            return []
        prompt = (
            f"Given this {language} method: {name}({', '.join(params[:5])})\n"
            f"Docstring: {(doc or '')[:200]}\n"
            f"Which concepts apply? (comma-separated, max 3)\n"
            f"Options: {', '.join(sorted(self.UNIVERSAL_CONCEPTS))}\n"
            f"If none, respond NONE.\nResponse:"
        )
        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            if "NONE" in text.upper():
                return []
            return [c.strip() for c in text.split(",")
                    if c.strip() in self.UNIVERSAL_CONCEPTS][:3]
        except Exception as e:
            logger.debug("SemanticLinker LLM error : %s", e)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeGraph — classe principale
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    KG automatisé — construit depuis les .md, l'AST et les liens sémantiques.
    Aucun triplet écrit à la main.
    """

    def __init__(self, persist_path: Path | None = None):
        from config import config
        self._path    = persist_path or (config.DATA_DIR / "knowledge_graph.json")
        self._graph   = nx.DiGraph()
        self._built   = False
        self._builder = KGBuilder(self._graph)

    # ── Construction ──────────────────────────────────────────────────────────

    def build(
        self,
        kb_dir:            Path | None = None,
        project_path:      Path | None = None,   # fallback si project_indexer absent
        project_indexer                = None,   # prioritaire sur project_path
        dependency_graph               = None,   # graphe NetworkX existant
        llm                            = None,
        force:             bool        = False,
    ) -> "KnowledgeGraph":
        """
        Construit le KG automatiquement depuis les sources disponibles.

        Ordre de priorité des sources AST :
          1. project_indexer.context.files  (prioritaire — déjà calculé, avec criticité)
          2. dependency_graph NetworkX      (arêtes IMPORTS fiables et résolues)
          3. project_path + code_parser     (fallback — si project_indexer absent)

        Cela évite le double parsing et garantit la cohérence entre
        le KG et les données déjà calculées par le projet.
        """
        from config import config
        kb_dir = kb_dir or config.KNOWLEDGE_BASE_DIR

        if not force and self._path.exists():
            self._load()
            # Vérifier si le cache est obsolète (concept nodes sans kb_queries)
            # Cela arrive après la mise à jour de SemanticLinker avec CONCEPT_KB_QUERIES
            concept_nodes_empty = sum(
                1 for _, d in self._graph.nodes(data=True)
                if d.get("node_type") in ("concept", "domain")
                and not d.get("kb_queries")
            )
            total_concept = sum(
                1 for _, d in self._graph.nodes(data=True)
                if d.get("node_type") in ("concept", "domain")
            )
            if total_concept > 0 and concept_nodes_empty == total_concept:
                logger.info(
                    "KG cache obsolète : %d nœuds concept sans kb_queries → rebuild",
                    concept_nodes_empty,
                )
                force = True
            else:
                logger.info("KG chargé depuis cache (%d nœuds, %d arêtes)",
                            self._graph.number_of_nodes(),
                            self._graph.number_of_edges())
                self._built = True
                return self

        logger.info("Construction automatique du KG...")

        # ── Source 1 : .md de la KB ───────────────────────────────────────────
        n_kb = self._builder.build_from_kb(kb_dir)

        # ── Source 2A : project_indexer (prioritaire) ─────────────────────────
        n_entities = 0
        if project_indexer is not None:
            n_entities = self._builder.build_from_project_indexer(project_indexer)
        elif project_path and project_path.exists():
            # Fallback : code_parser si project_indexer absent
            logger.info("project_indexer absent — fallback sur code_parser")
            n_entities = self._builder.build_from_ast(project_path)

        # ── Source 2B : dependency_graph (arêtes IMPORTS fiables) ────────────
        n_imports = 0
        if dependency_graph is not None:
            n_imports = self._builder.build_from_dependency_graph(dependency_graph)

        # ── Source 3 : liens sémantiques ──────────────────────────────────────
        n_sem = 0
        if project_indexer is not None:
            # Utilise les entités déjà dans project_indexer — pas de re-parsing
            n_sem = self._run_semantic_linking_from_indexer(project_indexer, llm)
        elif project_path and project_path.exists():
            n_sem = self._run_semantic_linking(project_path, llm)

        self._built = True
        self._save()

        logger.info(
            "KG prêt : %d nœuds, %d arêtes "
            "(KB:%d entités:%d imports:%d sem:%d)",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            n_kb, n_entities, n_imports, n_sem,
        )
        return self

    def update_file(
        self,
        file_path:       Path,
        project_indexer  = None,
        llm              = None,
    ) -> None:
        """
        Mise à jour incrémentale après une modification de fichier.

        Utilise project_indexer si disponible (prioritaire) — pas de re-parsing.
        Sinon fallback sur code_parser.

        Appelé depuis IncrementalAnalyzer._analyze_file() à l'ÉTAPE 4.5.
        """
        if not self._built:
            return

        file_id = file_path.name

        # Supprimer les anciens nœuds de ce fichier
        nodes_to_remove = [
            n for n, d in self._graph.nodes(data=True)
            if (d.get("node_type") in ("entity_class", "entity_method")
                and file_id in (d.get("source_file") or ""))
            or n.startswith(f"{file_id}::")
        ]
        self._graph.remove_nodes_from(nodes_to_remove)
        logger.debug("KG : %d anciens nœuds supprimés pour %s",
                     len(nodes_to_remove), file_id)

        try:
            if project_indexer and project_indexer.context:
                # ── Priorité 1 : project_indexer (zéro re-parsing) ───────────
                file_info = project_indexer.context.files.get(str(file_path), {})
                if file_info:
                    entities    = file_info.get("entities", [])
                    language    = file_info.get("language", "unknown")
                    criticality = file_info.get("criticality", 0)

                    # Recréer le nœud fichier
                    self._builder._add_node(KGNode(
                        id          = file_id,
                        node_type   = "file",
                        languages   = [language],
                        source_file = str(file_path),
                        extra       = {"full_path": str(file_path),
                                       "criticality": criticality},
                    ))

                    # Recréer les entités
                    current_class = None
                    for entity in entities:
                        name  = entity.get("name", "")
                        etype = entity.get("type", "")
                        if not name:
                            continue
                        node_type = ("entity_class"
                                     if etype in ("class", "interface", "enum")
                                     else "entity_method")
                        node_id = f"{file_id}::{name}"
                        self._builder._add_node(KGNode(
                            id=node_id, node_type=node_type,
                            languages=[language],
                            source_file=str(file_path),
                            extra={"entity_type": etype,
                                   "parameters": entity.get("parameters", []),
                                   "criticality": criticality},
                        ))
                        self._graph.add_edge(file_id, node_id,
                                             relation="CONTAINS",
                                             source_file=str(file_path))
                        self._graph.add_edge(node_id, file_id,
                                             relation="DEFINED_IN",
                                             source_file=str(file_path))
                        if etype in ("class", "interface"):
                            current_class = node_id
                        elif etype in ("method", "function", "constructor") and current_class:
                            self._graph.add_edge(current_class, node_id,
                                                 relation="HAS_METHOD",
                                                 source_file=str(file_path))

                    # Liens sémantiques
                    linker = SemanticLinker(self._graph, llm)
                    linker.link_entities(entities, str(file_path), language)

            else:
                # ── Fallback : code_parser ────────────────────────────────────
                from services.code_parser import parser as code_parser
                parsed = code_parser.parse_file(file_path)
                if "error" not in parsed:
                    self._builder.build_from_ast(file_path.parent)
                    linker = SemanticLinker(self._graph, llm)
                    linker.link_entities(
                        parsed.get("entities", []),
                        str(file_path),
                        parsed.get("language", "unknown"),
                    )

        except Exception as e:
            logger.debug("KG update_file erreur pour %s : %s", file_id, e)

        # Invalider le cache N-hop pour ce fichier
        self.invalidate_cache(file_id)
        self._save()
        logger.debug("KG mis à jour pour %s", file_id)

    def _run_semantic_linking_from_indexer(self, project_indexer, llm=None) -> int:
        """
        Crée les liens sémantiques depuis project_indexer.context.files.
        Zéro re-parsing — utilise les entités déjà calculées.
        """
        if not project_indexer or not project_indexer.context:
            return 0

        linker = SemanticLinker(self._graph, llm)
        total  = 0

        for file_path_str, file_info in project_indexer.context.files.items():
            try:
                entities = file_info.get("entities", [])
                language = file_info.get("language", "unknown")
                if entities:
                    total += linker.link_entities(entities, file_path_str, language)
            except Exception as e:
                logger.debug("SemanticLinker erreur %s : %s",
                             Path(file_path_str).name, e)

        return total

    def _run_semantic_linking(self, project_path: Path, llm=None) -> int:
        """Fallback — utilisé si project_indexer absent."""
        try:
            from services.code_parser import parser as code_parser
        except ImportError:
            return 0

        linker     = SemanticLinker(self._graph, llm)
        total      = 0
        extensions = {".py", ".java", ".js", ".ts", ".jsx", ".tsx"}
        excluded   = {"venv", "__pycache__", "node_modules", ".git",
                      "dist", "build", "target"}

        for fp in project_path.rglob("*"):
            if (fp.suffix not in extensions
                    or any(ex in fp.parts for ex in excluded)):
                continue
            try:
                parsed = code_parser.parse_file(fp)
                if "error" not in parsed:
                    total += linker.link_entities(
                        parsed.get("entities", []),
                        str(fp),
                        parsed.get("language", "unknown"),
                    )
            except Exception:
                pass

        return total

    # ── Self-Improving RAG — rechargement incrémental ────────────────────────

    def reload_kb_file(self, md_path: Path) -> bool:
        """
        Recharge UN SEUL fichier .md dans le KG sans reconstruire tout.
        Appelé par FeedbackProcessor et LearningAgent après promotion d'un fix.

        Retourne True si au moins un nœud a été ajouté.
        """
        if not self._built:
            logger.debug("reload_kb_file appelé avant build() — ignoré")
            return False

        n_before = self._graph.number_of_nodes()
        e_before = self._graph.number_of_edges()

        try:
            added = self._builder.build_from_kb_single_file(md_path)
            n_after = self._graph.number_of_nodes()
            e_after = self._graph.number_of_edges()
            dn = n_after - n_before
            de = e_after - e_before
            if dn > 0 or de > 0:
                self._save()
                logger.info(
                    "KG reload_kb_file : +%d nœuds +%d arêtes depuis %s",
                    dn, de, md_path.name,
                )
            return added > 0
        except Exception as e:
            logger.error("reload_kb_file erreur : %s", e)
            return False

    # ── Détection ─────────────────────────────────────────────────────────────

    def detect_patterns(
        self,
        code: str,
        language: str,
        parsed_entities: list | None = None,
    ) -> list[tuple[str, float]]:
        """
        Détecte les nœuds KG pertinents avec un SCORE DE CONFIANCE.

        Retourne une liste de (kg_node, confidence) triée par confiance décroissante.

        Fix Problème 1 — 3 niveaux de détection par ordre de fiabilité :
          Niveau 1 (confidence=1.0) : via entités AST parsées
            → utilise les CodeEntity de code_parser
            → évite les faux positifs dans les commentaires/strings
            → Ex : entity.name="execute" + entity.type="method" → SQL_Injection

          Niveau 2 (confidence=0.8) : via pattern_map des .md
            → correspond exactement aux patterns définis dans les règles
            → encore du texte mais les patterns .md sont précis

          Niveau 3 (confidence=0.5) : fallback heuristique
            → actif si les .md n'ont pas de pattern_map
            → cherche dans le texte brut (moins fiable)

        Fix Problème 5 — score de confiance utilisé pour :
          → prioriser les queries ChromaDB (haute confiance en premier)
          → filtrer les résultats peu fiables (seuil configurable)

        Args:
            code             : code source à analyser
            language         : langage de programmation
            parsed_entities  : entités AST depuis code_parser (optionnel mais recommandé)

        Returns:
            [(kg_node, confidence), ...] trié par confiance décroissante
        """
        if not self._built:
            self.build()

        # Résultats : kg_node → meilleur score de confiance
        scores: dict[str, float] = {}

        # ── Niveau 1 : AST entities (confidence=1.0) ─────────────────────────
        if parsed_entities:
            ast_detected = self._detect_from_ast(parsed_entities, language)
            for kg_node in ast_detected:
                scores[kg_node] = max(scores.get(kg_node, 0.0), 1.0)

        # ── Niveau 2 : pattern_map des .md (confidence=0.8) ──────────────────
        pattern_map   = self._graph.graph.get("pattern_map", {})
        lang_patterns = pattern_map.get(language.lower(), {})
        for pattern, kg_node in lang_patterns.items():
            if pattern in code:
                if self._graph.has_node(kg_node):
                    scores[kg_node] = max(scores.get(kg_node, 0.0), 0.8)

        # ── Niveau 3 : fallback heuristique (confidence=0.5) ─────────────────
        if not lang_patterns and not parsed_entities:
            fallback = self._fallback_detect(code, language, set(scores.keys()))
            for kg_node in fallback:
                scores[kg_node] = max(scores.get(kg_node, 0.0), 0.5)

        # Trier par confiance décroissante
        result = sorted(scores.items(), key=lambda x: -x[1])

        if result:
            logger.debug("KG patterns (%s) : %s",
                         language, [(n, round(c, 1)) for n, c in result])
        return result

    def detect_pattern_nodes(
        self,
        code: str,
        language: str,
        parsed_entities: list | None = None,
        min_confidence: float = 0.5,
    ) -> list[str]:
        """
        Version simplifiée — retourne juste les nœuds (sans scores).
        Filtre par min_confidence.
        Utilisé par expand_queries() et n_hop_retrieval().
        """
        return [
            node for node, conf
            in self.detect_patterns(code, language, parsed_entities)
            if conf >= min_confidence
        ]

    def has_pattern(self, code: str, language: str) -> bool:
        return bool(self.detect_patterns(code, language))

    def _detect_from_ast(self, entities: list, language: str) -> list[str]:
        """
        Fix Problème 1 — Détection via entités AST.

        Plus fiable que la recherche textuelle :
          → ne détecte pas dans les commentaires
          → ne détecte pas dans les strings
          → prend en compte le type de l'entité (méthode vs classe)
          → utilise les paramètres pour affiner

        Mapping AST → KG :
          method name "execute" + params contain "query" → SQL_Injection
          method name "authenticate"/"login"             → PlainText_Password check
          class name contains "Statement"/"ResultSet"    → Resource_Leak
          return type "Connection"                       → Resource_Leak
        """
        AST_RULES: dict[str, list[dict]] = {
            "java": [
                # SQL Injection — méthodes d'exécution SQL
                {"entity_types": ["method"],
                 "name_patterns": ["execute", "executeQuery", "executeUpdate",
                                   "prepareStatement", "createStatement"],
                 "kg_node": "SQL_Injection"},
                # Resource Leak — types de ressources DB
                {"entity_types": ["class", "method"],
                 "name_patterns": ["ResultSet", "Statement", "Connection",
                                   "FileWriter", "InputStream", "OutputStream"],
                 "kg_node": "Resource_Leak"},
                # Resource Leak — retour de type ressource
                {"entity_types": ["method"],
                 "return_type_patterns": ["Connection", "ResultSet", "Statement"],
                 "kg_node": "Resource_Leak"},
                # Auth — méthodes d'authentification
                {"entity_types": ["method"],
                 "name_patterns": ["authenticate", "login", "checkPassword",
                                   "verifyToken", "validateCredentials"],
                 "kg_node": "PlainText_Password"},
                # Thread safety — champs statiques
                {"entity_types": ["class"],
                 "decorator_patterns": ["static"],
                 "kg_node": "Static_Mutable_State"},
            ],
            "python": [
                # SQL Injection — appels cursor
                {"entity_types": ["method", "function"],
                 "name_patterns": ["execute", "executemany", "executescript"],
                 "kg_node": "SQL_Injection"},
                # Resource Leak — fonctions qui ouvrent des ressources
                {"entity_types": ["method", "function"],
                 "name_patterns": ["open", "connect", "cursor", "acquire"],
                 "kg_node": "Resource_Leak"},
                # Code injection
                {"entity_types": ["method", "function"],
                 "name_patterns": ["eval", "exec", "compile", "loads"],
                 "kg_node": "Code_Injection"},
                # Auth
                {"entity_types": ["method", "function"],
                 "name_patterns": ["authenticate", "login", "verify_password",
                                   "check_password", "validate_token"],
                 "kg_node": "PlainText_Password"},
            ],
            "typescript": [
                {"entity_types": ["method", "function"],
                 "name_patterns": ["innerHTML", "insertAdjacentHTML", "write"],
                 "kg_node": "XSS_Injection"},
                {"entity_types": ["method", "function"],
                 "name_patterns": ["eval", "Function"],
                 "kg_node": "Code_Injection"},
                {"entity_types": ["method", "function"],
                 "name_patterns": ["setItem", "getItem", "localStorage"],
                 "kg_node": "Sensitive_Storage"},
            ],
        }

        detected = []
        rules = AST_RULES.get(language.lower(), [])

        for entity in entities:
            name      = (entity.name if hasattr(entity, "name")
                         else entity.get("name", "")).lower()
            etype     = (entity.type if hasattr(entity, "type")
                         else entity.get("type", ""))
            ret_type  = (entity.return_type if hasattr(entity, "return_type")
                         else entity.get("return_type") or "")
            decorators = (entity.decorators if hasattr(entity, "decorators")
                          else entity.get("decorators") or [])
            params     = (entity.parameters if hasattr(entity, "parameters")
                          else entity.get("parameters") or [])

            for rule in rules:
                # Vérifier le type d'entité
                if etype not in rule.get("entity_types", []):
                    continue

                kg_node = rule["kg_node"]

                # Règle sur le nom
                if "name_patterns" in rule:
                    if any(pat.lower() in name for pat in rule["name_patterns"]):
                        if kg_node not in detected:
                            detected.append(kg_node)
                        continue

                # Règle sur le type de retour
                if "return_type_patterns" in rule and ret_type:
                    if any(pat.lower() in ret_type.lower()
                           for pat in rule["return_type_patterns"]):
                        if kg_node not in detected:
                            detected.append(kg_node)
                        continue

                # Règle sur les décorateurs
                if "decorator_patterns" in rule and decorators:
                    dec_str = " ".join(decorators).lower()
                    if any(pat.lower() in dec_str
                           for pat in rule["decorator_patterns"]):
                        if kg_node not in detected:
                            detected.append(kg_node)

        return detected

    def _fallback_detect(self, code: str, language: str, seen: set) -> list[str]:
        """Détection heuristique intégrée — active si les .md sans pattern_map."""
        FALLBACK: dict[str, dict[str, str]] = {
            "java": {
                "Statement": "SQL_Injection", "executeQuery": "SQL_Injection",
                "executeUpdate": "SQL_Injection", "ResultSet": "Resource_Leak",
                "DriverManager": "Resource_Leak", "Connection": "Resource_Leak",
                "FileWriter": "Resource_Leak", "password": "PlainText_Password",
                "Password": "PlainText_Password", "static ": "Static_Mutable_State",
            },
            "python": {
                "cursor.execute": "SQL_Injection", "cursor.executemany": "SQL_Injection",
                "execute(": "SQL_Injection", "cursor": "Resource_Leak",
                "open(": "Resource_Leak", "password": "PlainText_Password",
                "eval(": "Code_Injection", "exec(": "Code_Injection",
                "pickle.loads": "Code_Injection", "subprocess": "Code_Injection",
            },
            "typescript": {
                "innerHTML": "XSS_Injection", "dangerouslySetInnerHTML": "XSS_Injection",
                "eval(": "Code_Injection", "localStorage": "Sensitive_Storage",
                "password": "PlainText_Password", "jwt": "Token_Exposure",
            },
            "javascript": {
                "innerHTML": "XSS_Injection", "eval(": "Code_Injection",
                "password": "PlainText_Password", "localStorage": "Sensitive_Storage",
            },
        }
        detected = []
        for pattern, kg_node in FALLBACK.get(language.lower(), {}).items():
            if pattern in code and kg_node not in seen:
                if not self._graph.has_node(kg_node):
                    self._graph.add_node(kg_node, node_type="vulnerability",
                                         severity="HIGH", languages=[language],
                                         kb_queries={}, source_file=None)
                detected.append(kg_node)
                seen.add(kg_node)
        return detected

    # ── Query expansion ───────────────────────────────────────────────────────

    # Poids des relations pour le traversal — Fix Problème 2
    # Plus le poids est élevé, plus la relation est prioritaire
    RELATION_WEIGHTS: dict[str, float] = {
        "FIXED_BY":       1.0,   # correction directe — priorité max
        "OFTEN_WITH":     0.9,   # co-occurrence fréquente — très utile
        "REQUIRES":       0.8,   # prérequis — utile
        "CAUSED_BY":      0.7,   # cause — utile pour comprendre
        "IS_A":           0.5,   # généralisation — moins urgent
        "HANDLES":        0.6,   # lien sémantique — modéré
        "IS_CONCEPT_OF":  0.4,   # domaine parent — faible
        "CONTAINS":       0.2,   # structure fichier — très faible
        "DEFINED_IN":     0.1,   # structure fichier — très faible
        "HAS_METHOD":     0.2,   # structure fichier — très faible
        "IMPORTS":        0.3,   # dépendance — faible
    }

    def expand_queries(
        self,
        detected_nodes: list[str] | list[tuple[str, float]],
        language: str,
        depth: int = 2,
        min_weight: float = 0.4,
    ) -> list[str]:
        """
        BFS pondéré depuis les nœuds détectés.

        Fix Problème 2 — les arêtes ne sont plus toutes égales :
          FIXED_BY (1.0) > OFTEN_WITH (0.9) > IS_A (0.5) > IS_CONCEPT_OF (0.4)

        Le score de chaque nœud visité = confiance_nœud_départ × poids_relation.
        Seuls les nœuds avec score ≥ min_weight sont traversés.

        Résultat : les queries sont triées par pertinence réelle,
        pas par ordre de découverte BFS.

        Args:
            detected_nodes : nœuds KG détectés (liste de str ou de (str, float))
            language       : pour choisir la kb_query par langue
            depth          : profondeur max de traversal
            min_weight     : seuil minimal pour traverser une arête (défaut 0.4)

        Returns:
            Liste de queries ChromaDB triées par pertinence décroissante
        """
        if not self._built:
            self.build()

        # Normaliser l'input — accepte str ou (str, float)
        if detected_nodes and isinstance(detected_nodes[0], tuple):
            start_nodes = [(n, c) for n, c in detected_nodes]
        else:
            start_nodes = [(n, 1.0) for n in detected_nodes]

        # query → meilleur score cumulé
        query_scores: dict[str, float] = {}
        visited: set[str] = set()
        lang = language.lower()

        # Priority queue simulée : (score_négatif, nœud, profondeur)
        # On utilise une liste triée car heapq n'est pas nécessaire pour des graphes petits
        queue = [(-conf, node, 0) for node, conf in start_nodes]
        queue.sort()  # tri par score décroissant

        while queue:
            neg_score, current, d = queue.pop(0)
            current_score = -neg_score

            if current in visited:
                continue
            visited.add(current)

            # Collecter la kb_query de ce nœud
            node_data  = self._graph.nodes.get(current, {})
            kb_queries = node_data.get("kb_queries", {})
            query = (kb_queries.get(lang)
                     or kb_queries.get("java")
                     or kb_queries.get("python"))

            if query:
                # Garder le meilleur score pour cette query
                existing = query_scores.get(query, 0.0)
                query_scores[query] = max(existing, current_score)

            # Traversal des voisins si profondeur non atteinte
            if d < depth:
                for neighbor in self._graph.successors(current):
                    if neighbor in visited:
                        continue

                    edge_data    = self._graph.edges.get((current, neighbor), {})
                    relation     = edge_data.get("relation", "")
                    rel_weight   = self.RELATION_WEIGHTS.get(relation, 0.3)
                    child_score  = current_score * rel_weight

                    # Ignorer les arêtes trop faibles
                    if child_score < min_weight:
                        continue

                    queue.append((-child_score, neighbor, d + 1))
                    queue.sort()  # maintenir l'ordre de priorité

        # Trier les queries par score décroissant
        sorted_queries = [
            q for q, _ in sorted(query_scores.items(), key=lambda x: -x[1])
        ]

        logger.debug("KG expand pondéré : %d nœuds → %d queries (depth=%d)",
                     len(visited), len(sorted_queries), depth)
        return sorted_queries

    # ── N-Hop Retrieval ───────────────────────────────────────────────────────

    # Cache LRU pour les résultats n_hop — Fix Problème 6
    # Clé : (modified_file, language, depth) → queries résultantes
    _nhop_cache: dict[tuple, list[str]] = {}
    _nhop_cache_max: int = 50   # max 50 entrées en cache

    # Relations utiles pour le N-hop sémantique — Fix Problème 3
    # On ne traverse que les relations à valeur sémantique élevée
    NHOP_USEFUL_RELATIONS: frozenset = frozenset({
        "HANDLES", "IS_CONCEPT_OF", "FIXED_BY", "OFTEN_WITH", "REQUIRES",
    })

    def n_hop_retrieval(
        self,
        modified_file:  str,
        networkx_graph: nx.DiGraph | None = None,
        language:       str = "java",
        depth:          int = 3,
        min_score:      float = 0.3,
        use_cache:      bool = True,
    ) -> list[str]:
        """
        Traversal combiné graphe de fichiers (NetworkX) + KG de concepts.

        Fix Problème 3 — BFS avec scoring décroissant :
          Distance 1 → score 1.0 (très pertinent)
          Distance 2 → score 0.6 (pertinent)
          Distance 3 → score 0.3 (limite de pertinence)
          → Queries triées par score décroissant

        Fix Problème 3 — Filtrage par type de relation :
          Seules HANDLES, IS_CONCEPT_OF, FIXED_BY, OFTEN_WITH, REQUIRES
          sont traversées dans la partie KG sémantique.
          CONTAINS, DEFINED_IN, HAS_METHOD sont ignorées
          (structure de fichier, pas sémantique).

        Fix Problème 6 — Cache LRU :
          Même fichier + même language → résultat mis en cache
          Invalidé quand update_file() est appelé pour ce fichier.

        Algorithme complet :
          Étape 1 : NetworkX → fichiers impactés (prédécesseurs + successeurs)
          Étape 2 : KG → entités de chaque fichier impacté
          Étape 3 : KG → traversal pondéré depuis chaque entité
            - Relations sémantiques seulement (HANDLES, IS_CONCEPT_OF...)
            - Score décroissant avec la distance
          Étape 4 : Collecter et trier les kb_queries par score
        """
        if not self._built:
            self.build()

        #  Cache LRU
        cache_key = (modified_file, language, depth)
        if use_cache and cache_key in self._nhop_cache:
            logger.debug("N-hop cache hit pour %s", modified_file)
            return self._nhop_cache[cache_key]

        query_scores:  dict[str, float] = {}
        visited_files: set[str]         = set()
        visited_nodes: set[str]         = set()

        # ── Étape 1 : Fichiers impactés via NetworkX ──────────────────────────
        # Score de fichier : 1.0 pour le fichier modifié, 0.7 pour ses voisins
        impacted: dict[str, float] = {modified_file: 1.0}

        if networkx_graph is not None:
            for node_id in list(networkx_graph.nodes()):
                if modified_file not in str(node_id):
                    continue
                # Prédécesseurs (fichiers qui dépendent du fichier modifié)
                for pred in networkx_graph.predecessors(node_id):
                    fname = Path(str(pred).replace("file:", "")).name
                    impacted[fname] = max(impacted.get(fname, 0.0), 0.7)
                # Successeurs (fichiers dont dépend le fichier modifié)
                for succ in networkx_graph.successors(node_id):
                    fname = Path(str(succ).replace("file:", "")).name
                    impacted[fname] = max(impacted.get(fname, 0.0), 0.6)
                break

        # ── Étapes 2-4 : Traversal KG pour chaque fichier impacté ─────────────
        for fname, file_score in sorted(impacted.items(), key=lambda x: -x[1]):
            if fname in visited_files:
                continue
            visited_files.add(fname)

            # Entités KG de ce fichier
            file_entities = [
                (n, file_score)
                for n, d in self._graph.nodes(data=True)
                if (
                    (d.get("node_type") in ("entity_class", "entity_method")
                     and fname in (d.get("source_file") or ""))
                    or n.startswith(f"{fname}::")
                    or n == fname
                )
            ]

            for entity_node, entity_score in file_entities:
                if entity_node in visited_nodes:
                    continue

                # BFS pondéré depuis l'entité
                # On ne traverse QUE les relations sémantiques utiles
                bfs_queue = [(-entity_score, entity_node, 0)]

                while bfs_queue:
                    bfs_queue.sort()
                    neg_score, current, d = bfs_queue.pop(0)
                    current_score = -neg_score

                    if current in visited_nodes or d > depth:
                        continue
                    if current_score < min_score:
                        continue
                    visited_nodes.add(current)

                    # Collecter kb_query
                    node_data  = self._graph.nodes.get(current, {})
                    kb_queries = node_data.get("kb_queries", {})
                    query = (kb_queries.get(language.lower())
                             or kb_queries.get("java")
                             or kb_queries.get("python"))
                    if query:
                        existing = query_scores.get(query, 0.0)
                        query_scores[query] = max(existing, current_score)

                    # Traversal : seulement les relations sémantiques
                    if d < depth:
                        for neighbor in self._graph.successors(current):
                            if neighbor in visited_nodes:
                                continue
                            edge_data = self._graph.edges.get((current, neighbor), {})
                            relation  = edge_data.get("relation", "")

                            # Ignorer les relations structurelles
                            if relation not in self.NHOP_USEFUL_RELATIONS:
                                continue

                            rel_weight  = self.RELATION_WEIGHTS.get(relation, 0.3)
                            child_score = current_score * rel_weight
                            if child_score >= min_score:
                                bfs_queue.append((-child_score, neighbor, d + 1))

        # Trier les queries par score décroissant
        result = [q for q, _ in sorted(query_scores.items(), key=lambda x: -x[1])]

        # ── Mise en cache ──────────────────────────────────────────────────────
        if use_cache:
            if len(self._nhop_cache) >= self._nhop_cache_max:
                # Éviction simple : supprimer la première entrée
                oldest_key = next(iter(self._nhop_cache))
                del self._nhop_cache[oldest_key]
            self._nhop_cache[cache_key] = result

        logger.debug("N-hop '%s' : %d fichiers, %d nœuds, %d queries scorées",
                     modified_file, len(visited_files),
                     len(visited_nodes), len(result))
        return result

    def invalidate_cache(self, file_name: str | None = None) -> None:
        """
        Invalide le cache N-hop.
        Appelé par update_file() après modification d'un fichier.

        Args:
            file_name : invalide seulement les entrées de ce fichier.
                        None = invalide tout le cache.
        """
        if file_name is None:
            self._nhop_cache.clear()
        else:
            keys_to_remove = [k for k in self._nhop_cache if k[0] == file_name]
            for k in keys_to_remove:
                del self._nhop_cache[k]

    # ── Persistance ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            data = nx.node_link_data(self._graph)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("KG sauvegardé → %s (%d nœuds)",
                         self._path.name, self._graph.number_of_nodes())
        except Exception as e:
            logger.warning("Sauvegarde KG impossible : %s", e)

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._graph   = nx.node_link_graph(data)
            self._builder = KGBuilder(self._graph)
        except Exception as e:
            logger.warning("Chargement KG impossible, reconstruction : %s", e)
            self.build(force=True)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        if not self._built:
            self.build()
        g = self._graph
        print("\n" + "═" * 60)
        print("  Knowledge Graph — Statistiques")
        print("═" * 60)
        print(f"  Nœuds : {g.number_of_nodes()}   Arêtes : {g.number_of_edges()}")

        type_counts: dict[str, int] = {}
        for _, d in g.nodes(data=True):
            t = d.get("node_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print("\n  Types de nœuds :")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t:<25} {c}")

        rel_counts: dict[str, int] = {}
        for _, _, d in g.edges(data=True):
            r = d.get("relation", "?")
            rel_counts[r] = rel_counts.get(r, 0) + 1
        print("\n  Types de relations :")
        for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
            print(f"    {r:<25} {c}")

        pm = g.graph.get("pattern_map", {})
        if pm:
            total_p = sum(len(v) for v in pm.values())
            print(f"\n  Pattern map : {total_p} patterns — langages : {list(pm.keys())}")
        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Instance globale
# ─────────────────────────────────────────────────────────────────────────────

knowledge_graph = KnowledgeGraph()


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    kg = KnowledgeGraph()
    kg.build(force=True)
    kg.print_stats()

    code = """
def process_data(connection, user_input):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM data WHERE id = " + user_input)
    return cursor.fetchall()
"""
    print("--- detect_patterns ---")
    patterns = kg.detect_patterns(code, "python")
    print(f"  Patterns : {patterns}")

    print("\n--- expand_queries (depth=2) ---")
    for i, q in enumerate(kg.expand_queries(patterns, "python", depth=2), 1):
        print(f"  Q{i} : {q}")