"""
ProjectIndexer — Indexe tout le projet pour donner du contexte au LLM.

Corrections vs original :
  - from dependency_graph → from services.graph_service
  - Cache Redis MCP (migration depuis SQLite)
  - Toute la logique métier est identique à l'original
"""
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from services.graph_service import dependency_builder
from services.mcp_redis_service import get_mcp_redis, key_hash, KEY_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Contexte complet du projet avec toutes les métadonnées."""
    total_files:       int
    total_entities:    int
    languages:         Dict[str, int]
    packages:          List[str]
    files:             Dict[str, Dict]
    architecture_info: Dict[str, Any]


class ProjectIndexer:
    """
    Indexeur optimisé qui réutilise dependency_graph.

    Workflow :
      1. build_from_project() → données déjà parsées dans dependency_builder
      2. Enrichit avec packages et criticité depuis le graphe
      3. Sauvegarde dans SQLite (thread-safe) au lieu de JSON sans verrou
      4. Recharge en < 0.5s les fois suivantes
    """

    LANGUAGE_SUFFIXES = {
        "java": [
            "service", "controller", "repository", "dto", "entity", "model",
            "mapper", "dao", "impl", "config", "exception", "request",
            "response", "validator", "helper", "util",
        ],
        "python": [
            "service", "controller", "repository", "model", "dao", "helper",
            "utils", "config", "views", "serializer", "schema", "handler",
            "manager", "middleware", "forms", "admin", "tests", "factory",
        ],
        "javascript": [
            "service", "controller", "component", "module", "repository",
            "model", "helper", "provider", "guard", "interceptor", "pipe",
            "middleware", "store", "action", "reducer", "saga", "context",
            "hook", "utils", "config", "constants",
        ],
        "typescript": [
            "service", "controller", "component", "module", "repository",
            "model", "entity", "dto", "interface", "type", "guard",
            "interceptor", "pipe", "middleware", "resolver", "decorator",
        ],
    }

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.context: Optional[ProjectContext] = None
        self._lock = threading.Lock()
        self._redis = None  # Lazy init

    @property
    def redis(self):
        """Lazy init du client MCP Redis."""
        if self._redis is None:
            self._redis = get_mcp_redis()
        return self._redis

    def _ps_key(self) -> str:
        """Clé Redis pour le project snapshot."""
        return f"{KEY_PREFIX}ps:{key_hash(str(self.project_path))}"

    # ── Build principal ───────────────────────────────────────────────────────

    def build_index(self, dependency_graph=None, force_rebuild: bool = False) -> ProjectContext:
        if not force_rebuild and self._load_from_cache():
            return self.context

        print(" Indexation du projet...")

        if dependency_graph is None:
            print("   • Construction du graphe de dépendances...")
            dependency_graph = dependency_builder.build_from_project(self.project_path)

        file_entities     = dependency_builder.file_entities
        file_imports      = dependency_builder.file_imports
        architecture_info = dependency_builder.analyze_flows()
        coupling_metrics  = architecture_info.get("coupling_metrics", {})
        packages          = self._extract_packages(file_entities.keys())

        files_index    = {}
        languages      = {}
        total_entities = 0

        for file_path, entities in file_entities.items():
            language = self._detect_language(Path(file_path))
            languages[language] = languages.get(language, 0) + 1

            entities_list = [
                {
                    "name":       e.name,
                    "type":       e.type,
                    "start_line": e.start_line,
                    "end_line":   e.end_line,
                    "parameters": getattr(e, "parameters", []),
                }
                for e in entities
            ]
            total_entities += len(entities_list)

            imports     = [i.module for i in file_imports.get(file_path, [])]
            node_id     = f"file:{file_path}"
            criticality = coupling_metrics.get(node_id, {}).get("afferent", 0)

            files_index[file_path] = {
                "entities":     entities_list,
                "imports":      imports,
                "language":     language,
                "criticality":  criticality,
                "entity_count": len(entities_list),
            }

        self.context = ProjectContext(
            total_files       = len(file_entities),
            total_entities    = total_entities,
            languages         = languages,
            packages          = sorted(packages),
            files             = files_index,
            architecture_info = {
                "entry_points_count":  len(architecture_info.get("entry_points", [])),
                "circular_deps_count": len(architecture_info.get("circular_dependencies", [])),
                "orphaned_count":      len(architecture_info.get("orphaned_modules", [])),
            },
        )

        self._save_to_cache()
        print(f" Indexation terminée : {self.context.total_files} fichiers\n")
        return self.context

    # ── Cache Redis MCP ────────────────────────────────────────────────────────

    def _save_to_cache(self):
        now = datetime.now().isoformat()
        ps_key = self._ps_key()
        snapshot = {
            "saved_at":       now,
            "total_files":    self.context.total_files,
            "total_entities": self.context.total_entities,
            "languages":      self.context.languages,
            "packages":       self.context.packages,
            "files":          self.context.files,
            "architecture":   self.context.architecture_info,
        }
        with self._lock:
            self.redis.set(ps_key, json.dumps(snapshot, ensure_ascii=False))
        print(f" Index sauvegardé dans Redis : {ps_key}")

    def _load_from_cache(self) -> bool:
        ps_key = self._ps_key()
        try:
            raw = self.redis.get(ps_key)
        except Exception:
            return False
        if not raw:
            return False
        try:
            data = json.loads(raw)
            files_data = data.get("files", {})
            # Defensive: ensure files is a dict, not a string (cache corruption)
            if isinstance(files_data, str):
                logger.warning("Cache corruption: files_data is a string, resetting")
                files_data = {}
            self.context = ProjectContext(
                total_files       = data.get("total_files", 0),
                total_entities    = data.get("total_entities", 0),
                languages         = data.get("languages", {}),
                packages          = data.get("packages", []),
                files             = files_data,
                architecture_info = data.get("architecture", {}),
            )
            print(f" Index chargé depuis Redis : {self.context.total_files} fichiers\n")
            return True
        except Exception as e:
            logger.warning("Erreur chargement index cache Redis : %s", e)
            return False

    # ── Fichiers liés (logique originale conservée) ───────────────────────────

    def get_related_files(self, file_path: Path) -> List[str]:
        if self.context is None:
            return []
        file_lang = self._detect_language(file_path)
        base_name = self._extract_base_name(file_path.stem, file_lang)
        related   = []
        for indexed_path in self.context.files:
            if indexed_path == str(file_path):
                continue
            indexed_stem = Path(indexed_path).stem
            indexed_lang = self.context.files[indexed_path]["language"]
            if self._is_related(base_name, indexed_stem, file_lang, indexed_lang):
                related.append(indexed_path)
        return related[:5]

    def _extract_base_name(self, file_stem: str, language: str) -> str:
        stem_lower = file_stem.lower()
        if "." in stem_lower:
            parts = stem_lower.split(".")
            if len(parts) >= 2 and parts[1] in self.LANGUAGE_SUFFIXES.get(language, []):
                return parts[0]
        for suffix in self.LANGUAGE_SUFFIXES.get(language, []):
            if language == "python" and stem_lower.endswith(f"_{suffix}"):
                return stem_lower[: -len(suffix) - 1]
            if language in ("javascript", "typescript") and stem_lower.endswith(f"-{suffix}"):
                return stem_lower[: -len(suffix) - 1]
            if stem_lower.endswith(suffix):
                return stem_lower[: -len(suffix)]
        return stem_lower

    def _is_related(self, base: str, candidate: str, bl: str, cl: str) -> bool:
        bn = base.lower().replace("_", "").replace("-", "").replace(".", "")
        cn = candidate.lower().replace("_", "").replace("-", "").replace(".", "")
        if cn.startswith(bn) and len(cn) > len(bn):
            return True
        b = base.lower()
        for pat in (f"{b}_", f"{b}-", f"{b}.", b):
            if candidate.lower().startswith(pat):
                return True
        for sep in ("_", "-", "."):
            if sep in candidate.lower() and b in candidate.lower().split(sep):
                return True
        return False

    # ── format_for_llm (logique originale conservée) ─────────────────────────

    def format_for_llm(self, target_file: Path = None) -> str:
        if self.context is None:
            return ""
        lines = [
            "=" * 70, "PROJECT CONTEXT", "=" * 70, "",
            "PROJECT SUMMARY:",
            f"  • {self.context.total_files} files indexed",
            f"  • {self.context.total_entities} total entities",
            f"  • Languages: {', '.join(f'{k} ({v})' for k, v in self.context.languages.items())}",
        ]
        if self.context.packages:
            pkgs = ", ".join(self.context.packages[:30])
            if len(self.context.packages) > 30:
                pkgs += f" ... and {len(self.context.packages) - 30} more"
            lines.append(f"  • Existing Internal Packages/Dirs: {pkgs}")
        arch = self.context.architecture_info
        lines += [
            f"  • Entry points: {arch['entry_points_count']}",
            f"  • Circular dependencies: {arch['circular_deps_count']}",
            f"  • Orphaned modules: {arch['orphaned_count']}", "",
        ]
        if target_file:
            related = self.get_related_files(target_file)
            if related:
                lines += ["=" * 70, f"RELATED FILES FOR: {target_file.name}", "=" * 70, ""]
                for rp in related:
                    info     = self.context.files[rp]
                    entities = info["entities"]
                    lines.append(f"FILE: {Path(rp).name}")
                    lines.append(f"  Language: {info['language']}")
                    lines.append(f"  Criticality: {info['criticality']}")
                    if entities:
                        lines.append(f"  Entities ({len(entities)}):")
                        for e in entities[:10]:
                            params = e.get("parameters", [])
                            p_str  = ", ".join(params[:3]) + ("..." if len(params) > 3 else "")
                            lines.append(
                                f"    • {e['type']}: {e['name']}"
                                + (f"({p_str})" if params else "")
                            )
                        if len(entities) > 10:
                            lines.append(f"    ... and {len(entities) - 10} more")
                    lines.append("")
        lines += [
            "  IMPORTANT:",
            "• These files/packages ALREADY EXIST in the project",
            "• Do NOT suggest creating new files/classes that exist",
            "• Suggest using EXISTING entities shown above",
            "• Follow the existing project structure and packages",
        ]
        return "\n".join(lines)

    def get_file_criticality(self, file_path: Path) -> int:
        if self.context is None:
            return 0
        return self.context.files.get(str(file_path), {}).get("criticality", 0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_packages(self, file_paths) -> Set[str]:
        packages = set()
        for fp_str in file_paths:
            try:
                fp      = Path(fp_str)
                rel_dir = fp.relative_to(self.project_path).parent
                d       = str(rel_dir).replace("\\", "/")
                if d != ".":
                    packages.add(d)
                    if "src/main/java/" in d:
                        packages.add(d.split("src/main/java/")[-1].replace("/", "."))
                    if "/" in d:
                        packages.add(d.replace("/", "."))
            except ValueError:
                pass
        return packages

    def _detect_language(self, file_path: Path) -> str:
        return {
            ".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript", ".java": "java",
        }.get(file_path.suffix, "unknown")


# ── Singleton global ──────────────────────────────────────────────────────────

_project_indexer: Optional[ProjectIndexer] = None


def get_project_index(
    project_path:    Path,
    dependency_graph = None,
    force_rebuild:   bool = False,
) -> ProjectIndexer:
    global _project_indexer
    if _project_indexer is None:
        _project_indexer = ProjectIndexer(project_path)
    _project_indexer.build_index(dependency_graph, force_rebuild)
    return _project_indexer