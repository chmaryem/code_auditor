"""
IncrementalAnalyzer — System-Aware Edition
─────────────────────────────────────────────────────────────────────────────
Chaque analyse est maintenant "consciente du système" :
  1. GraphNeighborhoodExtractor  — interroge NetworkX pour trouver les voisins
     directs (predecessors = qui m'appelle, successors = ce que j'appelle).
  2. SystemAwareRAG              — fait 3 recherches ChromaDB au lieu d'1 :
       • le code du fichier actuel
       • les signatures des fichiers qui l'appellent (predecessors)
       • les signatures des fichiers qu'il appelle (successors)
     puis fusionne et déduplique les résultats.
  3. _build_system_impact_section() — construit la section [IMPACT SUR LE
     SYSTÈME] injectée dans le prompt pour que le LLM sache exactement quels
     fichiers et quelles méthodes sont à risque.
  4. Filtrage intelligent renforcé : changements mineurs → "RAS" immédiat.
"""

from pathlib import Path
from typing  import Dict, List, Any, Optional, Tuple
from queue   import Queue
import threading
import time
import difflib
import re
import hashlib

from services.code_parser import parser
from services.llm_service import assistant_agent
from services.graph_service import dependency_builder
from services.cache_service import CacheService as CacheManager
from services.project_indexer import get_project_index
from services.knowledge_loader import ProjectCodeIndexer
from services.knowledge_graph import knowledge_graph       # Phase 1 — KG RAG



# ─────────────────────────────────────────────────────────────────────────────
# ANSI / affichage
# ─────────────────────────────────────────────────────────────────────────────

def _enable_windows_ansi():
    try:
        import ctypes, sys
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
    except Exception:
        pass

_enable_windows_ansi()

_R  = "\033[0m"
_BD = "\033[1m"
_DM = "\033[2m"
_RD = "\033[91m"
_GR = "\033[92m"
_YL = "\033[93m"
_CY = "\033[96m"
_GY = "\033[90m"
_OR = "\033[38;5;208m"

_W    = 72
_SEP  = "\u2500" * _W
_SEP2 = "\u2550" * _W

_SEV = {
    "CRITICAL": (_RD, "\U0001f534", "CRITIQUE"),
    "HIGH":     (_OR, "\U0001f7e0", "HAUTE"),
    "MEDIUM":   (_YL, "\U0001f7e1", "MOYENNE"),
    "LOW":      ("\033[94m", "\U0001f535", "FAIBLE"),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS D'AFFICHAGE (inchangés)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fix_blocks(text: str) -> list:
    blocks = []
    parts  = re.split(r'-{3,}\s*FIX START\s*-{3,}', text, flags=re.IGNORECASE)
    for raw in parts[1:]:
        end = re.search(r'-{3,}\s*FIX END\s*-{3,}', raw, re.IGNORECASE)
        if end:
            raw = raw[:end.start()]

        def _f(name):
            m = re.search(
                r'\*\*' + re.escape(name) + r'\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|\Z)',
                raw, re.DOTALL | re.IGNORECASE
            )
            return m.group(1).strip() if m else ''

        def _code(section):
            m = re.search(
                r'\*\*' + re.escape(section) + r'\*\*.*?```\w*\n(.*?)```',
                raw, re.DOTALL | re.IGNORECASE
            )
            return m.group(1).rstrip() if m else ''

        sev_raw  = _f("SEVERITY").upper().split()[0] if _f("SEVERITY") else "MEDIUM"
        severity = sev_raw if sev_raw in _SEV else "MEDIUM"
        location = _f("LOCATION")
        line_m   = re.search(r'[:\s](\d{1,5})\b', location)
        problem  = _f("PROBLEM")
        if not problem:
            continue
        blocks.append({
            "problem":      problem,
            "severity":     severity,
            "location":     location,
            "line_number":  int(line_m.group(1)) if line_m else None,
            "current_code": _code("CURRENT CODE"),
            "fixed_code":   _code("FIXED CODE"),
            "why":          _f("WHY"),
        })
    return blocks


def _make_diff(current: str, fixed: str) -> str:
    if not current and not fixed:
        return ""
    cur_set = {l.strip() for l in current.splitlines() if l.strip()}
    fix_set = {l.strip() for l in fixed.splitlines()   if l.strip()}
    out = []
    for l in current.splitlines():
        if l.strip() in cur_set - fix_set:
            out.append(f"  {_RD}- {l}{_R}")
    for l in fixed.splitlines():
        if l.strip() in fix_set - cur_set:
            out.append(f"  {_GR}+ {l}{_R}")
    return "\n".join(out[:12])


def _print_block(block: dict, file_name: str) -> None:
    color, icon, label = _SEV.get(block["severity"], (_YL, "🟡", "MOYENNE"))
    loc      = block.get("location", "")
    line_num = block.get("line_number")

    print(f"\n{icon} [{_BD}{color}{label}{_R}] {_BD}{block['problem']}{_R}")
    if line_num:
        print(f"   \U0001f4cd {_CY}{file_name}:{line_num}{_R}  {_DM}({loc}){_R}")
    elif loc:
        print(f"   \U0001f4cd {_CY}{file_name}{_R}  {_DM}\u2192 {loc}{_R}")

    diff = _make_diff(block.get("current_code", ""), block.get("fixed_code", ""))
    if diff:
        print()
        print(diff)

    if block.get("why"):
        why = block["why"].replace("\n", " ").strip()
        if len(why) > 140:
            why = why[:137] + "\u2026"
        print(f"\n   \U0001f4a1 {why}")


def _print_results(text: str, file_name: str, context: dict,
                   elapsed: float, analyzed_count: int, score: int,
                   impacted: list) -> None:
    blocks = _parse_fix_blocks(text)
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for b in blocks:
        counts[b["severity"]] = counts.get(b["severity"], 0) + 1

    from datetime import datetime
    now = datetime.now().strftime("%H:%M:%S")
    hc  = _RD if counts["CRITICAL"] else _OR if counts["HIGH"] else _YL if counts["MEDIUM"] else _GR

    parts = [
        f"{_DM}[{now}]{_R}",
        f"{_BD}{file_name}{_R}",
        f"Score: {_BD}{hc}{score}/100{_R}",
    ]
    if counts["CRITICAL"]: parts.append(f"\U0001f534 {_RD}{_BD}{counts['CRITICAL']} Critique(s){_R}")
    if counts["HIGH"]:     parts.append(f"\U0001f7e0 {_OR}{counts['HIGH']} Haute(s){_R}")
    if counts["MEDIUM"]:   parts.append(f"\U0001f7e1 {_YL}{counts['MEDIUM']} Moyenne(s){_R}")
    if not blocks:         parts.append(f"\U0001f7e2 {_GR}OK{_R}")
    parts.append(f"{_DM}{elapsed:.1f}s{_R}")

    print(f"\n{_DM}{_SEP}{_R}")
    print("  " + f"  {_DM}\u2502{_R}  ".join(parts))
    print(f"{_DM}{_SEP}{_R}")

    if not blocks:
        clean = text.strip()
        if any(k in clean for k in ("\u2705", "no major issues", "code quality is good", "RAS")):
            print(f"\n  {_GR}\u2705  Aucun problème majeur détecté.{_R}\n")
        else:
            print(f"\n{clean}\n")
        print(f"{_DM}{_SEP2}{_R}")
        print(f"  {_DM}{elapsed:.1f}s  \u2502  Analysés : {analyzed_count}{_R}\n")
        return

    for block in blocks:
        print(_SEP)
        _print_block(block, file_name)

    print(f"\n{_DM}{_SEP}{_R}")
    if impacted:
        names = ", ".join(Path(p).name for p in impacted[:4])
        extra = f" +{len(impacted)-4}" if len(impacted) > 4 else ""
        print(f"\u26a0\ufe0f  {_YL}Impact sur {len(impacted)} dépendant(s) : {_BD}{names}{extra}{_R}")
    print(f"{_DM}{_SEP2}{_R}")
    print(f"  {_DM}{elapsed:.1f}s  \u2502  Analysés : {_BD}{analyzed_count}{_R}\n")


# ─────────────────────────────────────────────────────────────────────────────
# NOUVEAU — GraphNeighborhoodExtractor
# ─────────────────────────────────────────────────────────────────────────────

class GraphNeighborhoodExtractor:
    """
    Interroge le graphe NetworkX pour extraire le voisinage d'un fichier.

    Vocabulaire du graphe (dependency_graph.py) :
      node_id  = "file:/chemin/absolu/Fichier.java"
      arête A → B  signifie  "A importe B"  (A dépend de B)

    Donc :
      successors(node)   = fichiers que A UTILISE      (dépendances)
      predecessors(node) = fichiers qui UTILISENT A    (dépendants = impactés)

    Changement 2 — deux sources de voisinage combinées :
      • NetworkX    : imports résolus (connexions certaines)
      • ProjectIndexer.get_related_files() : convention de nommage
        (UserService → UserController même si l'import n'est pas résolu)

    Changement 3 — entités depuis project_indexer.context.files :
      Remplace dependency_builder.file_entities qui ne donnait que
      (name, type). context.files donne aussi (parameters, criticality)
      → signatures complètes dans le prompt [IMPACT SUR LE SYSTÈME].
    """

    def __init__(self, graph, project_indexer):
        self.graph          = graph           # nx.DiGraph
        self.project_indexer = project_indexer  # ProjectIndexer — source riche

    def get_neighborhood(self, file_path: Path) -> Dict[str, Any]:
        """
        Retourne le voisinage complet du fichier depuis deux sources :
          1. NetworkX predecessors/successors  (imports résolus)
          2. ProjectIndexer.get_related_files() (convention de nommage)

        Retourne :
          predecessors        : fichiers qui utilisent ce fichier (impactés)
          successors          : fichiers que ce fichier utilise
          indirect_impacted   : predecessors des predecessors (profondeur 2)
          predecessor_entities: {fichier: [{name, params, criticality}]}
          successor_entities  : idem
          criticality         : nb de predecessors directs
        """
        node_id = f"file:{file_path}"

        # ── Source 1 : NetworkX (imports résolus) ────────────────────────────
        if self.graph.has_node(node_id):
            pred_paths_nx = [
                n.replace("file:", "")
                for n in self.graph.predecessors(node_id)
                if n.startswith("file:")
            ]
            succ_paths_nx = [
                n.replace("file:", "")
                for n in self.graph.successors(node_id)
                if n.startswith("file:")
            ]
        else:
            pred_paths_nx = []
            succ_paths_nx = []

        # ── Source 2 : ProjectIndexer — convention de nommage ────────────────
        # Filet de sécurité : trouve UserController.java depuis UserService.java
        # même si l'import Java n'a pas été résolu par NetworkX
        related_by_name = []
        if self.project_indexer and self.project_indexer.context:
            try:
                related_by_name = self.project_indexer.get_related_files(file_path)
            except Exception:
                pass

        # ── Union des deux sources sans doublons (ordre stable) ───────────────
        # dict.fromkeys préserve l'ordre et déduplique
        all_preds = list(dict.fromkeys(pred_paths_nx + related_by_name))
        all_succs = list(dict.fromkeys(succ_paths_nx))

        # ── Entités enrichies depuis project_indexer.context.files ────────────
        # Changement 3 : on utilise context.files au lieu de file_entities
        # → on a les paramètres et la criticité de chaque méthode
        pred_entities = self._collect_entities_rich(all_preds)
        succ_entities = self._collect_entities_rich(all_succs)

        # ── Profondeur 2 — impact indirect (predecessors des predecessors) ─────
        indirect_impacted = set()
        for fp in all_preds:
            pred_node = f"file:{fp}"
            if self.graph.has_node(pred_node):
                for grand_pred in self.graph.predecessors(pred_node):
                    if grand_pred.startswith("file:") and grand_pred != node_id:
                        indirect_impacted.add(grand_pred.replace("file:", ""))

        return {
            "predecessors":          all_preds,
            "successors":            all_succs,
            "indirect_impacted":     list(indirect_impacted),
            "predecessor_entities":  pred_entities,
            "successor_entities":    succ_entities,
            "criticality":           len(all_preds),
        }

    def _collect_entities_rich(self, file_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Changement 3 — utilise project_indexer.context.files au lieu de
        dependency_builder.file_entities.

        Retourne :
          {
            "UserController.java": [
              {"name": "login",      "params": "username, password", "criticality": 2},
              {"name": "register",   "params": "username, email",    "criticality": 2},
              {"name": "getAllUsers","params": "",                    "criticality": 2},
            ]
          }

        Avant (file_entities) : juste les noms → "login, register, getAllUsers"
        Après (context.files) : noms + signatures → "login(username, password)"
        Le LLM voit les signatures exactes et peut détecter les cassures.
        """
        result = {}

        if not self.project_indexer or not self.project_indexer.context:
            return result

        context_files = self.project_indexer.context.files

        for fp in file_paths:
            file_name = Path(fp).name
            file_info = context_files.get(fp, {})
            entities  = file_info.get("entities", [])
            file_crit = file_info.get("criticality", 0)

            rich_entities = []
            for e in entities:
                if e.get("type") not in ("method", "function", "class"):
                    continue

                # Formater les paramètres : ["username", "password"] → "username, password"
                raw_params = e.get("parameters", [])
                params_str = ", ".join(raw_params[:6])   # max 6 params affichés
                if len(raw_params) > 6:
                    params_str += ", ..."

                rich_entities.append({
                    "name":        e.get("name", ""),
                    "params":      params_str,
                    "criticality": file_crit,
                })

            if rich_entities:
                result[file_name] = rich_entities

        return result

    @staticmethod
    def _empty_neighborhood() -> Dict[str, Any]:
        return {
            "predecessors":          [],
            "successors":            [],
            "indirect_impacted":     [],
            "predecessor_entities":  {},
            "successor_entities":    {},
            "criticality":           0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NOUVEAU — SystemAwareRAG
# ─────────────────────────────────────────────────────────────────────────────

class SystemAwareRAG:
    """
    RAG conscient du système — queries ChromaDB enrichies par le Knowledge Graph.

    Pipeline complet :
      1. detect_patterns()   → nœuds KG détectés dans le code
      2. expand_queries()    → traverse le graphe → requêtes enrichies
      3. _build_queries()    → queries structurelles (voisinage)
      4. Fusion              → toutes les queries → ChromaDB → top-K chunks

    Avant (3 queries fixes) :
      Query 1 : code brut
      Query 2 : signatures appelants
      Query 3 : signatures dépendances

    Après (3 fixes + N queries KG) :
      Query 1 : code brut
      Query 2 : signatures appelants
      Query 3 : signatures dépendances
      Query 4 : "resource leak cursor close python"   ← KG : SQL_Injection OFTEN_WITH Resource_Leak
      Query 5 : "try-with-resources context manager"  ← KG : Resource_Leak FIXED_BY TryWithResources
      ...
    """

    # Score max retenu — ChromaDB avec Jina retourne des distances L2 (pas cosinus)
    # Les valeurs typiques sont entre 0.3 et 1.5 selon la pertinence.
    # 0.75 était trop strict (filtrait tout) → 1.2 capture les chunks pertinents
    THRESHOLD = 1.2
    TOP_K     = 8

    def __init__(self, vector_store, language: str, project_code_indexer=None):
        self.vs                   = vector_store
        self.language             = language.lower()
        self.project_code_indexer = project_code_indexer

        # Phase 1 — KG initialisé une seule fois (singleton global)
        self._kg = knowledge_graph
        if not self._kg._built:
            self._kg.build()

    def retrieve(
        self,
        current_code: str,
        neighborhood: Dict[str, Any],
        current_file_name: str = "",
        networkx_graph=None,
    ) -> Tuple[list, list]:
        """
        Retourne (docs, scores) fusionnés depuis toutes les sources :
          1-3. Queries structurelles (code + voisinage)      → KB règles
          4-N. KG expand_queries (depth=2)                   → KB règles enrichies
          N+1. KG n_hop_retrieval (depth=3, NetworkX+KG)     → KB règles N-hop
          Last. Project code index                           → code projet similaire
        """
        # ── Étape A : Queries structurelles ──────────────────────────────────
        queries = self._build_queries(current_code, neighborhood)

        # ── Étape B : KG expand_queries (depth=2) ────────────────────────────
        # Passer les entités AST pour une détection plus précise (Fix Problème 1)
        # detect_patterns retourne maintenant [(node, confidence), ...]
        parsed_entities = neighborhood.get("_parsed_entities", [])
        detected_with_scores = self._kg.detect_patterns(
            current_code, self.language,
            parsed_entities = parsed_entities,
        )
        detected_patterns = [n for n, _ in detected_with_scores]

        kg_queries = self._kg.expand_queries(
            detected_nodes = detected_with_scores,  # passe les scores pour pondération
            language       = self.language,
            depth          = 2,
        )

        # ── Étape C : N-Hop Retrieval (depth=3, combine NetworkX + KG) ───────
        # La vraie innovation Graph RAG :
        # fichier modifié → fichiers impactés (NetworkX) → concepts (KG) → règles
        nhop_queries: list[str] = []
        if current_file_name:
            nhop_queries = self._kg.n_hop_retrieval(
                modified_file  = current_file_name,
                networkx_graph = networkx_graph,
                language       = self.language,
                depth          = 3,
            )

        # Affichage
        if detected_patterns:
            print(f"   • KG patterns : {detected_patterns}")
        total_kg = len(kg_queries) + len(nhop_queries)
        if total_kg:
            print(f"   • KG queries  : {len(kg_queries)} expand + {len(nhop_queries)} n-hop")

        # Fusionner toutes les queries sans doublons
        all_queries = queries.copy()
        for q in kg_queries + nhop_queries:
            if q not in all_queries:
                all_queries.append(q)

        seen: Dict[str, Tuple[Any, float]] = {}

        # ── Recherches dans la KB de règles génériques ────────────────────────
        for query in all_queries:
            if not query.strip():
                continue
            results = self.vs.similarity_search_with_score(query, k=self.TOP_K)
            # Affiche les scores pour calibration (première query seulement)
            if results and query == all_queries[0]:
                scores_str = ", ".join(f"{s:.3f}" for _, s in results[:4])
                print(f"   • Scores RAG : [{scores_str}] (seuil={self.THRESHOLD})")
            for doc, score in results:
                if score > self.THRESHOLD:
                    continue
                meta = doc.metadata
                key  = (
                    "kb:"
                    + meta.get("source_file", "")
                    + str(meta.get("chunk_index", hash(doc.page_content)))
                )
                if key not in seen or score < seen[key][1]:
                    seen[key] = (doc, score)

        # ── Recherche dans project_code_index ─────────────────────────────────
        if self.project_code_indexer:
            project_results = self.project_code_indexer.search(
                query        = current_code[:600],
                k            = 4,
                exclude_file = current_file_name,
                threshold    = self.THRESHOLD,
            )
            for doc, score in project_results:
                meta = doc.metadata
                key  = (
                    "proj:"
                    + meta.get("source_file", "")
                    + meta.get("entity_name", "")
                    + str(meta.get("start_line", ""))
                )
                if key not in seen or score < seen[key][1]:
                    seen[key] = (doc, score)

        if not seen:
            return [], []

        # Tri final : langue correspondante > KB rules > project code > score
        ranked = sorted(
            seen.values(),
            key=lambda pair: (
                0 if pair[0].metadata.get("language", "") == self.language else 1,
                0 if pair[0].metadata.get("collection", "") != "project_code" else 1,
                pair[1],
            ),
        )
        ranked = ranked[:self.TOP_K]

        docs   = [d for d, _ in ranked]
        scores = [s for _, s in ranked]
        return docs, scores

    def _build_queries(
        self,
        current_code: str,
        neighborhood: Dict[str, Any],
    ) -> List[str]:
        """
        Construit les 3 requêtes de recherche.
        Changement 3 — utilise les signatures complètes (name + params)
        pour des requêtes RAG plus précises.
        """
        queries = []

        # Query 1 — code du fichier actuel (800 chars suffisent pour l'embedding)
        queries.append(current_code[:800])

        # Query 2 — signatures des predecessors (fichiers qui appellent ce fichier)
        # Avant : "caller methods java: login, register"
        # Après  : "caller methods java: login(username, password), register(username, email)"
        pred_entities = neighborhood.get("predecessor_entities", {})
        if pred_entities:
            sigs = []
            for file_name, entities in pred_entities.items():
                for e in entities[:5]:
                    sig = f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                    sigs.append(sig)
            if sigs:
                queries.append(
                    f"caller methods {self.language}: "
                    + ", ".join(sigs[:15])
                )

        # Query 3 — signatures des successors (fichiers que ce fichier utilise)
        succ_entities = neighborhood.get("successor_entities", {})
        if succ_entities:
            sigs = []
            for file_name, entities in succ_entities.items():
                for e in entities[:5]:
                    sig = f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                    sigs.append(sig)
            if sigs:
                queries.append(
                    f"dependency interface {self.language}: "
                    + ", ".join(sigs[:15])
                )

        return queries


# ─────────────────────────────────────────────────────────────────────────────
# NOUVEAU — _build_system_impact_section
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_impact_section(
    file_name: str,
    neighborhood: Dict[str, Any],
) -> str:
    """
    Génère la section [IMPACT SUR LE SYSTÈME] injectée dans le prompt LLM.

    Format :
      [IMPACT SUR LE SYSTÈME]
      Tu analyses : UserService.java
      Fichiers qui l'appellent (RISQUE DE CASSE) : UserController.java, AuthFilter.java
        • UserController.java expose : login(), register(), getAllUsers()
        • AuthFilter.java expose     : doFilter()
      Fichiers que ce fichier utilise : UserRepository.java, DatabaseHelper.java
        • UserRepository.java fournit : findByUsername(), save()

      RÈGLE : Ne rename PAS les méthodes publiques sans vérifier chaque appelant.
              Tout changement de signature DOIT rester compatible avec B et C.
    """
    preds = neighborhood.get("predecessors", [])
    succs = neighborhood.get("successors",   [])
    indirect = neighborhood.get("indirect_impacted", [])
    pred_ent = neighborhood.get("predecessor_entities", {})
    succ_ent = neighborhood.get("successor_entities",   {})

    if not preds and not succs:
        return ""   # fichier isolé → pas de section système

    lines = [
        "",
        "═" * 68,
        "  [IMPACT SUR LE SYSTÈME]",
        "═" * 68,
        f"  Tu analyses            : {file_name}",
    ]

    # ── Predecessors (fichiers qui UTILISENT ce fichier) ─────────────────────
    if preds:
        pred_names = [Path(p).name for p in preds]
        lines.append(
            f"  ⚠️  Appelé par (RISQUE)  : {', '.join(pred_names)}"
        )
        for fp in preds:
            fname    = Path(fp).name
            # Changement 3 — rich_entities est maintenant une liste de dicts
            # {name, params, criticality} au lieu d'une liste de strings
            entities = pred_ent.get(fname, [])
            if entities:
                # Formater avec signatures : "login(username, password)"
                sigs = [
                    f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                    for e in entities[:8]
                ]
                crit = entities[0].get("criticality", 0)
                crit_tag = f"  [criticité {crit}]" if crit > 0 else ""
                lines.append(
                    f"      • {fname}{crit_tag} appelle : "
                    + ", ".join(sigs)
                )

    # ── Successors (ce que ce fichier UTILISE) ────────────────────────────────
    if succs:
        succ_names = [Path(p).name for p in succs]
        lines.append(
            f"  🔗 Utilise              : {', '.join(succ_names)}"
        )
        for fp in succs:
            fname    = Path(fp).name
            entities = succ_ent.get(fname, [])
            if entities:
                sigs = [
                    f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                    for e in entities[:8]
                ]
                lines.append(
                    f"      • {fname} fournit : "
                    + ", ".join(sigs)
                )

    # ── Impact indirect ───────────────────────────────────────────────────────
    if indirect:
        indirect_names = [Path(p).name for p in indirect[:5]]
        extra = f" +{len(indirect)-5}" if len(indirect) > 5 else ""
        lines.append(
            f"  📡 Impact indirect      : {', '.join(indirect_names)}{extra}"
        )

    # ── Règles d'architecture ─────────────────────────────────────────────────
    lines += [
        "",
        "  RÈGLES ARCHITECTURALES :",
        "  • Ne JAMAIS renommer une méthode publique sans vérifier chaque appelant.",
        "  • Tout changement de signature doit rester rétro-compatible.",
        "  • Si tu proposes un fix, vérifie qu'il ne casse pas les appelants listés.",
        "  • Signale explicitement si un fix impacte un fichier dépendant.",
        "═" * 68,
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ChangeAnalyzer (inchangé)
# ─────────────────────────────────────────────────────────────────────────────

class ChangeAnalyzer:
    """Analyse l'importance d'un changement pour décider s'il faut analyser."""

    # Types de changements considérés comme "mineurs" → réponse RAS immédiate
    MINOR_TYPES = frozenset({
        "whitespace_only", "comment_only", "import_only", "docstring_only", "no_change"
    })

    @staticmethod
    def analyze_change(old_content: str, new_content: str) -> Dict[str, Any]:
        if not old_content and not new_content:
            return {
                "significant": False, "score": 0, "lines_changed": 0,
                "change_type": "no_change", "reason": "Aucun contenu"
            }

        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines()
        diff      = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
        added     = [l[1:].strip() for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed   = [l[1:].strip() for l in diff if l.startswith("-") and not l.startswith("---")]

        lines_changed = len(added) + len(removed)
        change_type   = ChangeAnalyzer._classify_change(added, removed)
        score         = ChangeAnalyzer._calculate_score(change_type, lines_changed, added, removed)
        significant   = score >= 20
        reason        = ChangeAnalyzer._get_reason(change_type, lines_changed, significant)

        return {
            "significant":   significant,
            "score":         score,
            "lines_changed": lines_changed,
            "change_type":   change_type,
            "reason":        reason,
        }

    @staticmethod
    def _classify_change(added, removed):
        all_lines = added + removed
        if not all_lines:
            return "no_change"
        non_empty = [l for l in all_lines if l]
        if not non_empty:
            return "whitespace_only"
        if all(l.startswith(("import ", "from ")) for l in non_empty):
            return "import_only"
        if all(l.startswith(("#", "//", "/*", "*", "*/")) for l in non_empty):
            return "comment_only"
        if all('"""' in l or "'''" in l for l in non_empty):
            return "docstring_only"
        if any("def " in l or "class " in l or "function " in l for l in added):
            if not any("def " in l or "class " in l or "function " in l for l in removed):
                return "new_function"
        if (any("def " in l or "class " in l for l in added) and
                any("def " in l or "class " in l for l in removed)):
            return "function_signature"
        return "logic_change"

    @staticmethod
    def _calculate_score(change_type, lines_changed, added, removed):
        base_score  = lines_changed * 10
        type_scores = {
            "import_only":        5,
            "comment_only":       0,
            "whitespace_only":    0,
            "docstring_only":     10,
            "new_function":       max(base_score, 50),
            "function_signature": max(base_score, 70),
            "logic_change":       max(base_score, 30),
        }
        return min(type_scores.get(change_type, base_score), 100)

    @staticmethod
    def _get_reason(change_type, lines_changed, significant):
        if not significant:
            reasons = {
                "import_only":     f"Import seulement ({lines_changed} ligne(s))",
                "comment_only":    "Commentaires seulement",
                "whitespace_only": "Formatage seulement",
                "docstring_only":  "Documentation seulement",
                "no_change":       "Aucun changement",
            }
            return reasons.get(change_type, f"Changement mineur ({lines_changed} ligne(s))")
        reasons = {
            "logic_change":       f"Logique modifiée ({lines_changed} ligne(s))",
            "new_function":       f"Nouvelle fonction ({lines_changed} ligne(s))",
            "function_signature": f"Signature modifiée ({lines_changed} ligne(s)) — Impact possible",
        }
        return reasons.get(change_type, f"Changement important ({lines_changed} ligne(s))")


# ─────────────────────────────────────────────────────────────────────────────
# IncrementalAnalyzer — System-Aware Edition
# ─────────────────────────────────────────────────────────────────────────────

class IncrementalAnalyzer:
    """
    Analyseur incrémental SYSTEM-AWARE.

    Nouveautés vs version précédente :
      • GraphNeighborhoodExtractor : interroge NetworkX avant chaque analyse
        pour identifier predecessors (appelants) et successors (dépendances).
      • SystemAwareRAG : 3 recherches ChromaDB (code + appelants + dépendances)
        fusionnées et dédupliquées → meilleurs chunks RAG.
      • Section [IMPACT SUR LE SYSTÈME] dans le prompt : le LLM sait exactement
        quels fichiers et méthodes sont à risque.
      • Filtrage intelligent : changements mineurs → affiche "RAS" immédiatement
        sans appeler le LLM ni consommer de tokens.
    """

    def __init__(self, project_path: Path):
        self.project_path     = project_path
        self.cache            = CacheManager()
        self.dependency_graph = None
        self.analysis_queue   = Queue()
        self.worker_thread    = None
        self.is_running       = False
        self.project_indexer  = None
        self.file_contents:   Dict[str, str] = {}
        self._print_lock      = threading.Lock()
        self._last_hash:      Dict[str, str] = {}

        # Composants System-Aware (initialisés dans initialize())
        self._neighborhood_extractor: Optional[GraphNeighborhoodExtractor] = None
        self._project_code_indexer:   Optional[ProjectCodeIndexer]         = None
       

        self.stats = {
            "analyzed":      0,
            "skipped_hash":  0,
            "skipped_minor": 0,
            "time_total":    0.0,
            "by_type":       {},
        }

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self):
        """Initialise le graphe, l'indexeur projet, le neighborhood extractor et le KG."""
        print(" Initialisation System-Aware RAG...")
        print(" Indexation du projet...")

        self.project_indexer  = get_project_index(self.project_path)
        self.dependency_graph = dependency_builder.build_from_project(self.project_path)

        # Neighbourhood extractor — utilise project_indexer (riche) + graphe NetworkX
        self._neighborhood_extractor = GraphNeighborhoodExtractor(
            graph            = self.dependency_graph,
            project_indexer  = self.project_indexer,
        )

        # ProjectCodeIndexer — réutilise les embeddings déjà chargés
        self._project_code_indexer = ProjectCodeIndexer(
            embeddings = assistant_agent.embeddings,
        )
        n_chunks = self._project_code_indexer.index_project(self.project_path)
        print(f" ProjectCodeIndexer : {n_chunks} chunks de code projet dans ChromaDB")

        # ── Phase 1 — Knowledge Graph ─────────────────────────────────────────
        # Construit depuis les sources DÉJÀ calculées — zéro double parsing :
        #   Source 1 : .md de la KB (front-matter YAML)
        #   Source 2A: project_indexer.context.files (entités + criticité)
        #   Source 2B: dependency_graph NetworkX (arêtes IMPORTS résolues)
        #   Source 3 : liens sémantiques depuis project_indexer (heuristiques)
        print(" Knowledge Graph RAG...")
        knowledge_graph.build(
            project_indexer  = self.project_indexer,    # prioritaire
            dependency_graph = self.dependency_graph,   # arêtes fiables
            llm              = assistant_agent.llm,
        )
        kg_nodes = knowledge_graph._graph.number_of_nodes()
        kg_edges = knowledge_graph._graph.number_of_edges()
        print(f" KG : {kg_nodes} concepts, {kg_edges} relations")

        nodes = self.dependency_graph.number_of_nodes()
        edges = self.dependency_graph.number_of_edges()
        print(f" Graphe : {nodes} nœuds, {edges} arêtes")

      
        # Initialiser le processeur de feedback pour enrichir la KB
        # après validation des fixes par l'utilisateur.
        try:
            from services.knowledge_loader import KnowledgeBaseLoader
            self._kb_loader = KnowledgeBaseLoader()
            
        except Exception as e:
            print(f"⚠️  Warning : KnowledgeBaseLoader non initialisé — feedback non traité : {e}")
            self._kb_loader = None
            

        self._start_worker()

    # ── Worker ────────────────────────────────────────────────────────────────

    def _start_worker(self):
        self.is_running    = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        while self.is_running:
            try:
                task = self.analysis_queue.get(timeout=1)
                if task.get("deleted"):
                    self._handle_deletion(task["file_path"])
                else:
                    self._analyze_file(task["file_path"])
                self.analysis_queue.task_done()
            except Exception:
                pass

    def queue_analysis(self, file_path: Path, deleted: bool = False):
        if deleted:
            self.analysis_queue.put({"file_path": file_path, "deleted": True})
            return
        if not self.cache.has_file_changed(file_path):
            return
        self.analysis_queue.put({"file_path": file_path, "deleted": False})

    # ── Suppression ───────────────────────────────────────────────────────────

    def _handle_deletion(self, file_path: Path):
        print(f"\n {file_path.name} supprimé")
        self.cache.remove_file_from_cache(file_path)
        node_id = f"file:{file_path}"
        if self.dependency_graph.has_node(node_id):
            self.dependency_graph.remove_node(node_id)
        self.cache.save()
        print()

    # ── Analyse principale ────────────────────────────────────────────────────

    def _analyze_file(self, file_path: Path):
        """
        Pipeline System-Aware complet :
          1. Hash check
          2. Lecture du fichier
          3. ChangeAnalyzer — filtre les changements mineurs (→ RAS)
          4. Parsing AST
          5. Mise à jour du graphe NetworkX
          6. GraphNeighborhoodExtractor — voisinage (predecessors + successors)
          7. SystemAwareRAG — 3 recherches ChromaDB fusionnées
          8. Construction du contexte + section [IMPACT SUR LE SYSTÈME]
          9. Appel LLM avec le prompt enrichi
         10. Affichage + cache
        """
        start = time.time()

        print(f"\n{'─'*70}")
        print(f" {file_path.name}")

        # ── ÉTAPE 1 : Hash check ──────────────────────────────────────────────
        if not self.cache.has_file_changed(file_path):
            print("  Ignoré : Hash identique\n")
            self.stats["skipped_hash"] += 1
            return

        # ── ÉTAPE 2 : Lecture ─────────────────────────────────────────────────
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                new_content = f.read()
        except Exception as e:
            print(f" Erreur lecture : {e}\n")
            return

        # ── ÉTAPE 3 : Filtre intelligent — changements mineurs ────────────────
        old_content = self.file_contents.get(str(file_path), "")
        change_info = ChangeAnalyzer.analyze_change(old_content, new_content)

        print(f" {change_info['reason']} (score: {change_info['score']}/100)")

        if not change_info["significant"]:
            # Changement mineur : RAS immédiat, aucun token consommé
            print(f"  {_GR}✅ RAS — changement mineur, analyse non nécessaire{_R}\n")
            self.stats["skipped_minor"] += 1
            self.stats["by_type"][change_info["change_type"]] = (
                self.stats["by_type"].get(change_info["change_type"], 0) + 1
            )
            self.cache.update_file_cache(
                file_path,
                {"analysis": "RAS — changement mineur", "relevant_knowledge": []},
                [], [],
            )
            self.cache.save()
            self.file_contents[str(file_path)] = new_content
            return

        print(" Analyse System-Aware lancée...")

        # ── ÉTAPE 4 : Parsing AST ─────────────────────────────────────────────
        parsed = parser.parse_file(file_path)
        if "error" in parsed:
            print(f" Erreur parsing : {parsed['error']}\n")
            return

        entities = len(parsed.get("entities", []))
        imports  = len(parsed.get("imports",  []))
        print(f"   • {entities} entité(s), {imports} import(s)")

        # ── ÉTAPE 4.5 : Mise à jour ProjectCodeIndexer + KG ─────────────────
        # NON-BLOQUANT : timeout réel via shutdown(wait=False).
        if self._project_code_indexer:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            ex = ThreadPoolExecutor(max_workers=1)
            try:
                future = ex.submit(
                    self._project_code_indexer.index_file,
                    file_path,
                    new_content,
                    parsed.get("entities", []),
                )
                future.result(timeout=4)
            except FuturesTimeout:
                logger.debug("ProjectCodeIndexer timeout — pipeline non bloqué")
            except Exception as e:
                logger.debug("ProjectCodeIndexer ignoré pour %s : %s", file_path.name, e)
            finally:
                ex.shutdown(wait=False)

        # Mise à jour incrémentale du KG — utilise project_indexer (zéro re-parsing)
        try:
            knowledge_graph.update_file(
                file_path       = file_path,
                project_indexer = self.project_indexer,   # source prioritaire
                llm             = None,  # LLM désactivé pour les updates incrémentaux
            )
        except Exception as e:
            logger.debug("KG update_file ignoré pour %s : %s", file_path.name, e)

        # ── ÉTAPE 5 : Mise à jour graphe NetworkX ────────────────────────────
        self._update_graph(file_path, parsed)

        # ── ÉTAPE 6 : Voisinage NetworkX (NOUVEAU) ────────────────────────────
        neighborhood = self._neighborhood_extractor.get_neighborhood(file_path)

        preds    = neighborhood["predecessors"]
        succs    = neighborhood["successors"]
        indirect = neighborhood["indirect_impacted"]
        crit     = neighborhood["criticality"]

        # Affichage du voisinage
        if preds:
            pred_names = [Path(p).name for p in preds[:3]]
            extra = f" +{len(preds)-3}" if len(preds) > 3 else ""
            print(f" ⚠️  Appelé par : {', '.join(pred_names)}{extra}  ({crit} dépendant(s))")
        if succs:
            succ_names = [Path(p).name for p in succs[:3]]
            print(f" 🔗 Utilise     : {', '.join(succ_names)}")
        if indirect:
            print(f" 📡 Impact indirect : {len(indirect)} fichier(s)")

        crit_emoji = "🔴" if crit > 5 else "🟡" if crit > 0 else "🟢"
        print(f"{crit_emoji} Criticité : {crit}")

        # ── ÉTAPE 7 : SystemAwareRAG (NOUVEAU) ───────────────────────────────
        language = self._detect_language(file_path)
        print(f" RAG System-Aware ({language})...", flush=True)

        # Injecter les entités dans le neighborhood pour detect_patterns
        # Priorité : project_indexer (cohérent) > parsed (code_parser)
        file_info = (self.project_indexer.context.files.get(str(file_path), {})
                     if self.project_indexer and self.project_indexer.context
                     else {})
        indexer_entities = file_info.get("entities", [])
        neighborhood["_parsed_entities"] = (
            indexer_entities if indexer_entities
            else parsed.get("entities", [])
        )

        system_rag = SystemAwareRAG(
            vector_store         = assistant_agent.vector_store,
            language             = language,
            project_code_indexer = self._project_code_indexer,
        )
        relevant_docs, rag_scores = system_rag.retrieve(
            current_code      = new_content,
            neighborhood      = neighborhood,
            current_file_name = file_path.name,
            networkx_graph    = self.dependency_graph,   # N-hop RAG
        )
        print(f"   • {len(relevant_docs)} chunk(s) RAG (KB rules + project code)")

        # ── ÉTAPE 8 : Construction du contexte enrichi ────────────────────────
        context = self._build_context(file_path, neighborhood)

        # Section [IMPACT SUR LE SYSTÈME] — injectée dans le prompt
        system_impact_section = _build_system_impact_section(
            file_name    = file_path.name,
            neighborhood = neighborhood,
        )

        context["project_context"]      = self.project_indexer.format_for_llm(file_path)
        context["change_type"]          = change_info["change_type"]
        context["lines_changed"]        = change_info["lines_changed"]
        context["system_impact_section"] = system_impact_section   # ← NOUVEAU
        context["neighborhood"]          = neighborhood              # ← NOUVEAU

        # ── ÉTAPE 9 : LLM ────────────────────────────────────────────────────
        print(" Analyse LLM (System-Aware)...", flush=True)

        analysis = assistant_agent.analyze_code_with_rag(
            code    = new_content,
            context = context,
            # Passer les docs RAG déjà calculés — évite une 2e recherche dans assistant_agent
            precomputed_docs   = relevant_docs,
            precomputed_scores = rag_scores,
        )

        # ── ÉTAPE 10 : Cache ──────────────────────────────────────────────────
        self.cache.update_file_cache(
            file_path, analysis,
            context["dependencies"],
            context["dependents"],
        )
        self.cache.save()
        self.file_contents[str(file_path)] = new_content

        # ── Affichage ─────────────────────────────────────────────────────────
        elapsed     = time.time() - start
        result_text = analysis["analysis"]
        result_hash = hashlib.md5(result_text.encode("utf-8", errors="replace")).hexdigest()
        file_key    = str(file_path)

        self.stats["analyzed"]  += 1
        self.stats["time_total"] += elapsed
        self.stats["by_type"][change_info["change_type"]] = (
            self.stats["by_type"].get(change_info["change_type"], 0) + 1
        )

        if self._last_hash.get(file_key) == result_hash:
            return   # même résultat déjà affiché (watchdog double-fire)
        self._last_hash[file_key] = result_hash

        with self._print_lock:
            _print_results(
                text           = result_text,
                file_name      = file_path.name,
                context        = context,
                elapsed        = elapsed,
                analyzed_count = self.stats["analyzed"],
                score          = change_info["score"],
                impacted       = preds,
            )

        # ── ÉTAPE 11 : Self-Improving RAG — Feedback Loop ─────────────────────
        # Proposer à l'utilisateur de valider un fix pour enrichir la KB.
        # Non-bloquant — timeout de 10s, l'utilisateur peut ignorer.
        # Si un fix est validé :
        #   1. LLM extrait une règle générale depuis ce fix spécifique
        #   2. ChromaDB vérifie si une règle similaire existe déjà
        #   3. Génère un .md structuré avec kg_nodes + kg_relations
        #   4. reload_file() + KG rebuild → disponible immédiatement
        if self._feedback_processor:
            # Extraire les blocs de fix parsés depuis le texte
            fix_blocks = _parse_fix_blocks(result_text)
            if fix_blocks:
                try:
                    self._feedback_processor.collect_feedback(
                        blocks           = fix_blocks,
                        code_before      = new_content,
                        language         = language,
                        file_name        = file_path.name,
                        project_indexer  = self.project_indexer,
                        dependency_graph = self.dependency_graph,
                    )
                except Exception as e:
                    logger.debug("Feedback Loop erreur : %s", e)

    # ── Helpers graphe ────────────────────────────────────────────────────────

    def _update_graph(self, file_path: Path, parsed: dict):
        """Met à jour le graphe NetworkX après chaque sauvegarde."""
        node_id = f"file:{file_path}"
        if self.dependency_graph.has_node(node_id):
            # Supprimer les anciennes arêtes sortantes (imports)
            self.dependency_graph.remove_edges_from(
                list(self.dependency_graph.out_edges(node_id))
            )
        else:
            self.dependency_graph.add_node(node_id)

        for imp in parsed.get("imports", []):
            target = self._resolve_import(imp, file_path.parent)
            if target:
                t_node = f"file:{target}"
                if not self.dependency_graph.has_node(t_node):
                    self.dependency_graph.add_node(t_node)
                self.dependency_graph.add_edge(node_id, t_node, relation="imports")

    def _resolve_import(self, imp, current_dir: Path) -> Optional[str]:
        """Résout un import relatif Python vers un chemin absolu."""
        if not imp.module or not imp.module.startswith("."):
            return None
        parts  = imp.module.split(".")
        path   = current_dir
        for p in parts:
            path = path.parent if p == "" else path / p
        py_file = path.with_suffix(".py")
        return str(py_file) if py_file.exists() else None

    def _build_context(self, file_path: Path, neighborhood: Dict[str, Any]) -> dict:
        """Construit le contexte pour l'agent LLM depuis le voisinage NetworkX."""
        return {
            "file_path":        str(file_path),
            "language":         self._detect_language(file_path),
            "dependencies":     neighborhood["successors"],
            "dependents":       neighborhood["predecessors"],
            "criticality_score": neighborhood["criticality"],
            "is_entry_point":   neighborhood["criticality"] == 0,
        }

    def _find_impacted(self, file_path: Path) -> List[str]:
        """Raccourci legacy — utilise le neighborhood extractor."""
        if self._neighborhood_extractor:
            nb = self._neighborhood_extractor.get_neighborhood(file_path)
            return nb["predecessors"]
        node_id = f"file:{file_path}"
        if not self.dependency_graph.has_node(node_id):
            return []
        return [
            n.replace("file:", "")
            for n in self.dependency_graph.predecessors(node_id)
            if n.startswith("file:")
        ]

    @staticmethod
    def _detect_language(file_path: Path) -> str:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self):
        print("\n Arrêt...")
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.cache.save()
        print(f"\n Statistiques:")
        print(f"   Analysés      : {self.stats['analyzed']}")
        print(f"   Ignorés hash  : {self.stats['skipped_hash']}")
        print(f"   Ignorés mineur: {self.stats['skipped_minor']}")
        if self.stats["analyzed"] > 0:
            avg = self.stats["time_total"] / self.stats["analyzed"]
            print(f"   Temps moyen   : {avg:.1f}s")
        if self.stats["by_type"]:
            print(f"   Par type      : {self.stats['by_type']}")