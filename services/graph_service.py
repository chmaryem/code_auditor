from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

import networkx as nx

from services.code_parser import parser, CodeEntity, ImportStatement

logger = logging.getLogger(__name__)




@dataclass
class DependencyNode:
    """Nœud dans le graphe de dépendances."""
    identifier: str
    type: str        # file | class | function | method | module | external
    file_path: str
    references:    Set[str] = field(default_factory=set)
    referenced_by: Set[str] = field(default_factory=set)


# ─────────────────────────────────────────────────────────────────────────────
# RÉSOLVEUR D'IMPORTS MULTI-STRATÉGIE
# ─────────────────────────────────────────────────────────────────────────────

class MultiStrategyImportResolver:
    

    # Extensions candidates dans l'ordre de priorité (JS/TS)
    JS_EXTENSIONS = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

        # Cache : (module_raw, current_dir_str) → resolved_path | None
        self._cache: Dict[Tuple[str, str], Optional[str]] = {}

        # Index : nom_court → Set[chemin_absolu]
        # Exemple : "config" → {"…/config.py"}
        self._name_index: Dict[str, Set[str]] = {}

        # Tous les fichiers du projet connus
        self._project_files: Set[str] = set()

        self._stats = {
            "resolved_python_relative": 0,
            "resolved_python_absolute": 0,
            "resolved_js_relative":     0,
            "resolved_js_absolute":     0,
            "resolved_java":            0,
            "resolved_index":           0,
            "unresolved":               0,
            "cache_hits":               0,
        }

    # ── alimentation de l'index ───────────────────────────────────────────────

    def register_file(self, file_path: Path):
        """Enregistre un fichier du projet dans l'index."""
        abs_path = str(file_path.resolve())
        self._project_files.add(abs_path)

        stem = file_path.stem.lower()
        if stem not in self._name_index:
            self._name_index[stem] = set()
        self._name_index[stem].add(abs_path)

        # Pour Python : "config" ET "mypackage.config"
        try:
            rel = file_path.resolve().relative_to(self.project_root)
            # Module path sans extension : src/myapp/config.py → myapp.config
            parts = list(rel.with_suffix("").parts)
            # Retirer les préfixes communs (src, main, app…)
            skip = {"src", "main", "app", "lib", "source"}
            while parts and parts[0].lower() in skip:
                parts = parts[1:]
            if parts:
                module_path = ".".join(parts)
                if module_path not in self._name_index:
                    self._name_index[module_path] = set()
                self._name_index[module_path].add(abs_path)
        except ValueError:
            pass

    def build_index(self, all_files: List[Path]):
        """Construit l'index à partir de tous les fichiers du projet."""
        for f in all_files:
            self.register_file(f)
        logger.debug(f"Import resolver index : {len(self._name_index)} entrées, "
                     f"{len(self._project_files)} fichiers")

    # ── résolution principale ─────────────────────────────────────────────────

    def resolve(self, import_stmt: ImportStatement, current_file: Path) -> Optional[str]:
        """
        Résout un import vers un chemin absolu de fichier.
        Retourne None si l'import est externe ou non résolvable.
        """
        current_dir = current_file.parent.resolve()
        cache_key = (import_stmt.module, str(current_dir), import_stmt.import_type)

        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        result = self._resolve_uncached(import_stmt, current_dir, current_file)

        # Ne stocker dans le cache que les chemins du projet
        if result and result in self._project_files:
            self._cache[cache_key] = result
        else:
            self._cache[cache_key] = None
            result = None

        return result

    def _resolve_uncached(self, imp: ImportStatement,
                           current_dir: Path, current_file: Path) -> Optional[str]:
        """Logique de résolution sans cache."""
        import_type = imp.import_type
        module      = imp.module

        # ── Python ───────────────────────────────────────────────────────────
        if import_type == "python_import":
            if imp.is_relative or module.startswith("."):
                return self._resolve_python_relative(module, current_dir, current_file)
            else:
                return self._resolve_python_absolute(module, current_dir)

        # ── JavaScript / TypeScript ───────────────────────────────────────────
        elif import_type in ("es_import", "commonjs_require"):
            if imp.is_relative or module.startswith(("./", "../")):
                return self._resolve_js_relative(module, current_dir)
            else:
                return self._resolve_js_absolute(module)

        # ── Java ──────────────────────────────────────────────────────────────
        elif import_type == "java_import":
            return self._resolve_java(module)

        # ── Inconnu : essai générique ─────────────────────────────────────────
        else:
            return self._resolve_by_index(module)

    # ── Python relatif ────────────────────────────────────────────────────────

    def _resolve_python_relative(self, module: str, current_dir: Path,
                                   current_file: Path) -> Optional[str]:
        """
        Résout les imports relatifs Python correctement.

        from .config import X       → current_dir/config.py
        from ..utils import Y       → current_dir/../utils.py
        from ...core.base import Z  → current_dir/../../core/base.py
        """
        # Compter les points de tête pour déterminer le niveau
        level = 0
        while level < len(module) and module[level] == ".":
            level += 1
        relative_module = module[level:]   # partie après les points

        # Remonter d'autant de niveaux que de points
        base_dir = current_dir
        for _ in range(level - 1):        # -1 car level=1 = dossier courant
            base_dir = base_dir.parent

        # Résoudre le nom de module en chemin
        if relative_module:
            parts = relative_module.split(".")
            candidate = base_dir.joinpath(*parts)
        else:
            candidate = base_dir

        # Essayer .py puis __init__.py
        for suffix in (".py", "/__init__.py", ".py/__init__.py"):
            path = Path(str(candidate) + suffix.replace("/", str(Path("/"))))
            if not suffix.startswith("/"):
                path = candidate.with_suffix(".py") if suffix == ".py" else candidate / "__init__.py"
            if path.exists():
                resolved = str(path.resolve())
                if resolved in self._project_files:
                    self._stats["resolved_python_relative"] += 1
                    return resolved

        # Dernier essai : juste base_dir/__init__.py
        init = candidate / "__init__.py"
        if init.exists():
            resolved = str(init.resolve())
            if resolved in self._project_files:
                self._stats["resolved_python_relative"] += 1
                return resolved

        self._stats["unresolved"] += 1
        return None

    # ── Python absolu ─────────────────────────────────────────────────────────

    def _resolve_python_absolute(self, module: str, current_dir: Path) -> Optional[str]:
        """
        Résout les imports absolus Python.

        from config import Config        → project_root/config.py
        from myapp.utils import helper   → project_root/myapp/utils.py
        import os.path                   → (standard library, ignoré)
        """
        # Chercher d'abord dans l'index (cas le plus fréquent)
        found = self._resolve_by_index(module)
        if found:
            self._stats["resolved_python_absolute"] += 1
            return found

        # Reconstruire le chemin à partir de la racine du projet
        parts = module.split(".")
        for search_root in self._python_search_roots(current_dir):
            candidate = search_root.joinpath(*parts)

            # Essai 1 : module.py
            py_file = candidate.with_suffix(".py")
            if py_file.exists():
                resolved = str(py_file.resolve())
                if resolved in self._project_files:
                    self._stats["resolved_python_absolute"] += 1
                    return resolved

            # Essai 2 : module/__init__.py (package)
            init_file = candidate / "__init__.py"
            if init_file.exists():
                resolved = str(init_file.resolve())
                if resolved in self._project_files:
                    self._stats["resolved_python_absolute"] += 1
                    return resolved

            # Essai 3 : premier composant seulement (import myapp.models → myapp/models.py)
            if len(parts) > 1:
                top_candidate = search_root.joinpath(*parts[:-1]) / (parts[-1] + ".py")
                if top_candidate.exists():
                    resolved = str(top_candidate.resolve())
                    if resolved in self._project_files:
                        self._stats["resolved_python_absolute"] += 1
                        return resolved

        self._stats["unresolved"] += 1
        return None

    def _python_search_roots(self, current_dir: Path) -> List[Path]:
        """
        Retourne les dossiers dans lesquels chercher les modules Python.
        Ordre : dossier courant → racine projet → src/ → app/.
        """
        roots = [current_dir, self.project_root]
        for candidate in ("src", "app", "lib", "source"):
            extra = self.project_root / candidate
            if extra.is_dir():
                roots.append(extra)
        return roots

    # ── JavaScript / TypeScript relatif ──────────────────────────────────────

    def _resolve_js_relative(self, module: str, current_dir: Path) -> Optional[str]:
        """
        Résout les imports JS/TS relatifs selon l'algorithme Node.js.

        ./utils        → utils.js | utils.jsx | utils.ts | utils/index.js …
        ../components  → ../components.tsx | ../components/index.tsx …
        """
        raw_path = current_dir / module

        # Essai 1 : chemin exact (import './app.js')
        if raw_path.exists() and raw_path.is_file():
            resolved = str(raw_path.resolve())
            if resolved in self._project_files:
                self._stats["resolved_js_relative"] += 1
                return resolved

        # Essai 2 : ajouter des extensions
        for ext in self.JS_EXTENSIONS:
            candidate = raw_path.with_suffix(ext) if raw_path.suffix == "" else Path(str(raw_path) + ext)
            if not candidate.exists():
                candidate = Path(str(raw_path) + ext)
            if candidate.exists():
                resolved = str(candidate.resolve())
                if resolved in self._project_files:
                    self._stats["resolved_js_relative"] += 1
                    return resolved

        # Essai 3 : dossier/index.EXT
        if raw_path.is_dir():
            for ext in self.JS_EXTENSIONS:
                index_file = raw_path / f"index{ext}"
                if index_file.exists():
                    resolved = str(index_file.resolve())
                    if resolved in self._project_files:
                        self._stats["resolved_js_relative"] += 1
                        return resolved

        # Essai 4 : package.json#main dans le dossier
        pkg_json = raw_path / "package.json" if raw_path.is_dir() else None
        if pkg_json and pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                main = pkg.get("main") or pkg.get("module")
                if main:
                    main_file = raw_path / main
                    if main_file.exists():
                        resolved = str(main_file.resolve())
                        if resolved in self._project_files:
                            self._stats["resolved_js_relative"] += 1
                            return resolved
            except Exception:
                pass

        self._stats["unresolved"] += 1
        return None

    # ── JavaScript absolu (alias / monorepo) ─────────────────────────────────

    def _resolve_js_absolute(self, module: str) -> Optional[str]:
        """
        Résout les imports JS absolus (aliases Webpack/Vite, monorepo).

        @/utils         → src/utils.ts  (alias @ = src/)
        @app/button     → packages/button/index.ts
        utils           → (node_modules — ignoré)
        """
        # Alias @ commun dans Vue/React : @ → src/
        if module.startswith("@/"):
            relative = module[2:]
            for src_dir in ("src", "app", "lib"):
                base = self.project_root / src_dir
                if base.is_dir():
                    result = self._resolve_js_relative(relative, base)
                    if result:
                        self._stats["resolved_js_absolute"] += 1
                        return result

        # Essai dans l'index par nom court
        last_part = module.split("/")[-1].split(".")[0]
        found = self._resolve_by_index(last_part)
        if found:
            self._stats["resolved_js_absolute"] += 1
            return found

        self._stats["unresolved"] += 1
        return None

    

    def _resolve_java(self, module: str) -> Optional[str]:
        """
        Résout un import Java vers un fichier source.

        import com.example.service.UserService;
        → src/main/java/com/example/service/UserService.java
        → src/com/example/service/UserService.java
        → (partout dans le projet)

        Les imports java.*, javax.*, org.* des bibliothèques standard sont ignorés.
        """
        # Filtrer les bibliothèques standard et courantes
        std_prefixes = (
            "java.", "javax.", "jakarta.", "sun.", "com.sun.",
            "android.", "kotlin.", "scala.",
        )
        if any(module.startswith(p) for p in std_prefixes):
            self._stats["unresolved"] += 1
            return None

        # Ignorer les imports wildcard (import com.example.*)
        if module.endswith(".*"):
            self._stats["unresolved"] += 1
            return None

        
        parts = module.split(".")
        rel_path = Path(*parts).with_suffix(".java")

     
        java_roots = [
            self.project_root / "src" / "main" / "java",
            self.project_root / "src" / "test" / "java",
            self.project_root / "src",
            self.project_root,
        ]

        for root in java_roots:
            candidate = root / rel_path
            if candidate.exists():
                resolved = str(candidate.resolve())
                if resolved in self._project_files:
                    self._stats["resolved_java"] += 1
                    return resolved

        
        class_name = parts[-1]
        found = self._resolve_by_index(class_name.lower())
        if found and found.endswith(".java"):
            self._stats["resolved_java"] += 1
            return found

        self._stats["unresolved"] += 1
        return None

  

    def _resolve_by_index(self, module: str) -> Optional[str]:
        """
        Cherche dans l'index nom_court → fichier.
        Utile pour les imports courts ('config', 'utils', 'helper').
        """
        key = module.lower().replace("/", ".").replace("\\", ".")

        # Recherche directe
        if key in self._name_index:
            candidates = self._name_index[key]
            if len(candidates) == 1:
                self._stats["resolved_index"] += 1
                return next(iter(candidates))
            # Ambiguïté : retourner le plus probable (le plus court chemin)
            best = min(candidates, key=lambda p: len(p))
            self._stats["resolved_index"] += 1
            return best

        # Recherche sur le dernier composant seulement
        last = key.split(".")[-1]
        if last != key and last in self._name_index:
            candidates = self._name_index[last]
            if candidates:
                self._stats["resolved_index"] += 1
                return min(candidates, key=lambda p: len(p))

        return None

    # ── statistiques ─────────────────────────────────────────────────────────

    def print_stats(self):
        total_resolved = sum(v for k, v in self._stats.items()
                             if k.startswith("resolved_") and not k.endswith("index"))
        total_resolved += self._stats["resolved_index"]
        total = total_resolved + self._stats["unresolved"]

        print(f"\n{'─'*60}")
        print(" Import resolution stats :")
        for key, val in self._stats.items():
            if val:
                label = key.replace("_", " ").capitalize()
                print(f"   • {label:<36} {val:>5}")
        if total:
            pct = total_resolved / total * 100
            print(f"   {'Résolution totale':<36} {total_resolved:>5}/{total} ({pct:.0f}%)")
        print(f"{'─'*60}\n")

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class DependencyGraphBuilder:
    """
    Construit et analyse le graphe de dépendances d'un projet.

    Interface publique identique à l'original :
      build_from_project(project_path) → nx.DiGraph
      analyze_flows()                  → Dict
      file_entities                    → Dict[str, List[CodeEntity]]
      file_imports                     → Dict[str, List[ImportStatement]]

    Amélioration interne : résolution d'imports multi-stratégie.
    """

    def __init__(self):
        self.graph           = nx.DiGraph()
        self.nodes:          Dict[str, DependencyNode]    = {}
        self.file_entities:  Dict[str, List[CodeEntity]]  = {}
        self.file_imports:   Dict[str, List[ImportStatement]] = {}

        self._resolver: Optional[MultiStrategyImportResolver] = None
        self._project_root: Optional[Path] = None

    # ── API publique (inchangée) ──────────────────────────────────────────────

    def build_from_project(self, project_path: Path) -> nx.DiGraph:
        """
        Construit le graphe à partir d'un projet.
        Retourne le même nx.DiGraph qu'avant (rétro-compatible).
        """
        self._project_root = project_path.resolve()

        # Réinitialiser pour permettre les builds successifs
        self.reset()

        # ── Étape 1 : scanner les fichiers ───────────────────────────────────
        all_files = self._scan_project(project_path)
        logger.info(f"Fichiers trouvés : {len(all_files)}")

        # ── Étape 2 : construire l'index de résolution ───────────────────────
        self._resolver = MultiStrategyImportResolver(project_path)
        self._resolver.build_index(all_files)

        # ── Étape 3 : parser chaque fichier ──────────────────────────────────
        errors = 0
        for file_path in all_files:
            if not self._parse_file(file_path):
                errors += 1
        if errors:
            logger.warning(f"{errors} fichier(s) n'ont pas pu être parsés")

        # ── Étape 4 : construire les nœuds ───────────────────────────────────
        self._build_nodes()

        # ── Étape 5 : construire les arêtes ──────────────────────────────────
        self._build_edges()

        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        logger.info(f"Graphe construit : {n_nodes} nœuds, {n_edges} arêtes")

        return self.graph

    def reset(self):
        """Réinitialise le graphe pour un nouveau build."""
        self.graph          = nx.DiGraph()
        self.nodes          = {}
        self.file_entities  = {}
        self.file_imports   = {}

    # ── scan ─────────────────────────────────────────────────────────────────

    def _scan_project(self, project_path: Path) -> List[Path]:
        """Scan récursif des fichiers du projet (inchangé)."""
        files = []
        extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java'}
        exclude_dirs = {
            'node_modules', '__pycache__', 'venv', 'env', '.venv',
            '.git', 'dist', 'build', '.pytest_cache', '.mypy_cache',
            '.tox', 'coverage', '.next', '.nuxt', 'target', 'out',
        }

        for file_path in project_path.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix not in extensions:
                continue
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            files.append(file_path)

        return files

    # ── parse ─────────────────────────────────────────────────────────────────

    def _parse_file(self, file_path: Path) -> bool:
        """Parse un fichier et stocke les résultats. Retourne True si succès."""
        try:
            result = parser.parse_file(file_path)
            if "error" in result:
                logger.debug(f"Parse error {file_path.name}: {result['error']}")
                return False
            self.file_entities[str(file_path)] = result.get("entities", [])
            self.file_imports[str(file_path)]  = result.get("imports",  [])
            return True
        except Exception as e:
            logger.warning(f"Exception lors du parsing de {file_path}: {e}")
            return False

    # ── nœuds (identique à l'original) ──────────────────────────────────────

    def _build_nodes(self):
        """Construit les nœuds du graphe."""
        # Nœuds fichiers
        for file_path in self.file_entities:
            node_id = f"file:{file_path}"
            node = DependencyNode(identifier=node_id, type="file", file_path=file_path)
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.__dict__)

        # Nœuds entités + arête entité→fichier
        for file_path, entities in self.file_entities.items():
            file_node_id = f"file:{file_path}"
            for entity in entities:
                node_id = f"{entity.type}:{file_path}:{entity.name}"
                node = DependencyNode(
                    identifier=node_id, type=entity.type, file_path=file_path)
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **node.__dict__)
                self.graph.add_edge(node_id, file_node_id, relation="defined_in")

    # ── arêtes (AMÉLIORÉ : résolution multi-stratégie) ───────────────────────

    def _build_edges(self):
        """
        Construit les arêtes d'import.
        Utilise MultiStrategyImportResolver pour résoudre chaque import.
        """
        external_count  = 0
        resolved_count  = 0
        unresolved_count = 0

        for file_path_str, imports in self.file_imports.items():
            file_path   = Path(file_path_str)
            file_node   = f"file:{file_path_str}"

            for import_stmt in imports:
                target_path = self._resolver.resolve(import_stmt, file_path)

                if target_path:
                    target_node = f"file:{target_path}"

                    # Ajouter le nœud cible s'il n'existe pas encore
                    # (peut arriver si le fichier n'a pas été parsé)
                    if target_node not in self.nodes:
                        node = DependencyNode(
                            identifier=target_node,
                            type="file",
                            file_path=target_path,
                        )
                        self.nodes[target_node] = node
                        self.graph.add_node(target_node, **node.__dict__)

                    self.graph.add_edge(file_node, target_node,
                                        relation="imports",
                                        module=import_stmt.module)
                    resolved_count += 1
                else:
                    # Import externe : créer un nœud "external" pour les stats
                    module_key = import_stmt.module.split(".")[0].split("/")[0]
                    if module_key:
                        ext_node = f"external:{module_key}"
                        if ext_node not in self.graph:
                            self.graph.add_node(
                                ext_node,
                                identifier=ext_node,
                                type="external",
                                file_path="",
                            )
                        self.graph.add_edge(file_node, ext_node,
                                            relation="external_dep",
                                            module=import_stmt.module)
                        external_count += 1
                    else:
                        unresolved_count += 1

        logger.info(
            f"Arêtes : {resolved_count} internes résolues, "
            f"{external_count} externes, {unresolved_count} non résolues"
        )

    # ── analyse (inchangée pour rétro-compatibilité totale) ──────────────────

    def analyze_flows(self) -> Dict[str, Any]:
        """Analyse les flux dans le graphe (inchangé)."""
        return {
            "entry_points":           self._find_entry_points(),
            "critical_paths":         self._find_critical_paths(),
            "circular_dependencies":  self._find_circular_dependencies(),
            "orphaned_modules":       self._find_orphaned_modules(),
            "coupling_metrics":       self._calculate_coupling(),
        }

    def _find_entry_points(self) -> List[str]:
        """Trouve les points d'entrée (nœuds sans dépendances entrantes)."""
        return [n for n in self.graph.nodes()
                if self.graph.in_degree(n) == 0
                # Exclure les nœuds externes — on veut uniquement les fichiers projet
                and not str(n).startswith("external:")]

    def _find_critical_paths(self) -> List[List[str]]:
        """Trouve les chemins critiques (plus longues chaînes de dépendances)."""
        try:
            dag = self.graph.copy()
            # Retirer les nœuds externes pour simplifier
            external = [n for n in dag.nodes() if str(n).startswith("external:")]
            dag.remove_nodes_from(external)
            # Supprimer les cycles
            for cycle in list(nx.simple_cycles(dag)):
                if len(cycle) >= 2:
                    dag.remove_edge(cycle[-1], cycle[0])
            longest = []
            for source in self._find_entry_points():
                for target in dag.nodes():
                    if source != target:
                        try:
                            path = nx.shortest_path(dag, source, target)
                            if len(path) > 5:
                                longest.append(path)
                        except nx.NetworkXNoPath:
                            pass
            return sorted(longest, key=len, reverse=True)[:10]
        except Exception as e:
            logger.debug(f"_find_critical_paths: {e}")
            return []

    def _find_circular_dependencies(self) -> List[List[str]]:
        """Détecte les dépendances circulaires (fichiers projet uniquement)."""
        try:
            # Sous-graphe fichiers uniquement
            file_nodes = [n for n in self.graph.nodes()
                          if str(n).startswith("file:")]
            sub = self.graph.subgraph(file_nodes)
            return list(nx.simple_cycles(sub))
        except Exception:
            return []

    def _find_orphaned_modules(self) -> List[str]:
        """Trouve les modules isolés (fichiers sans aucune connexion)."""
        return [n for n in self.graph.nodes()
                if self.graph.degree(n) == 0
                and str(n).startswith("file:")]

    def _calculate_coupling(self) -> Dict[str, Dict[str, float]]:
        """
        Calcule les métriques de couplage.
        Les nœuds externes sont ignorés dans le calcul d'instabilité.
        """
        metrics: Dict[str, Dict[str, float]] = {}

        for node in self.graph.nodes():
            in_deg  = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)

            # Pour les fichiers : ne compter que les arêtes vers d'autres fichiers
            if str(node).startswith("file:"):
                internal_out = sum(
                    1 for _, tgt, data
                    in self.graph.out_edges(node, data=True)
                    if data.get("relation") == "imports"
                    and str(tgt).startswith("file:")
                )
                internal_in = sum(
                    1 for src, _, data
                    in self.graph.in_edges(node, data=True)
                    if data.get("relation") == "imports"
                    and str(src).startswith("file:")
                )
                total = internal_in + internal_out
                metrics[node] = {
                    "efferent":    internal_out,
                    "afferent":    internal_in,
                    "instability": internal_out / total if total > 0 else 0,
                }
            else:
                metrics[node] = {
                    "efferent":    out_deg,
                    "afferent":    in_deg,
                    "instability": out_deg / (in_deg + out_deg)
                                   if (in_deg + out_deg) > 0 else 0,
                }

        return metrics

    # ── export ───────────────────────────────────────────────────────────────

    def export_graph(self, output_path: Path, format: str = "dot"):
        """Exporte le graphe (inchangé)."""
        if format == "dot":
            nx.drawing.nx_pydot.write_dot(self.graph, str(output_path))
        elif format == "gexf":
            nx.write_gexf(self.graph, str(output_path))

    # ── stats ─────────────────────────────────────────────────────────────────

    def print_stats(self):
        """Affiche les statistiques du graphe et de la résolution d'imports."""
        file_nodes    = sum(1 for n in self.graph.nodes() if str(n).startswith("file:"))
        external_nodes = sum(1 for n in self.graph.nodes() if str(n).startswith("external:"))
        import_edges  = sum(1 for _, _, d in self.graph.edges(data=True)
                            if d.get("relation") == "imports")
        ext_edges     = sum(1 for _, _, d in self.graph.edges(data=True)
                            if d.get("relation") == "external_dep")

        print(f"\n{'═'*60}")
        print(f" Graphe de dépendances")
        print(f"{'═'*60}")
        print(f"  Fichiers projet  : {file_nodes}")
        print(f"  Modules externes : {external_nodes}")
        print(f"  Imports internes : {import_edges}")
        print(f"  Imports externes : {ext_edges}")
        print(f"  Nœuds total      : {self.graph.number_of_nodes()}")
        print(f"  Arêtes total     : {self.graph.number_of_edges()}")
        print(f"{'═'*60}")

        if self._resolver:
            self._resolver.print_stats()

        parser.print_stats()


# ── instance globale (rétro-compatible) ──────────────────────────────────────
dependency_builder = DependencyGraphBuilder()