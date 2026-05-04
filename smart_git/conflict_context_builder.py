"""
conflict_context_builder.py — Injection de contexte projet pour la résolution de conflits.

╔══════════════════════════════════════════════════════════════════════╗
║  Fix 1 — Le LLM ne résout plus "à l'aveugle"                        ║
║                                                                      ║
║  Ce module collecte le contexte du projet cible et le formate        ║
║  pour injection dans le prompt de résolution de conflit.             ║
║                                                                      ║
║  3 Niveaux de priorité (budget ~1500 tokens / ~6000 chars) :         ║
║                                                                      ║
║    Niveau 1 — INDISPENSABLE (toujours injecté) :                     ║
║      • Champs/méthodes de l'entité en jeu (depuis project_indexer)   ║
║      • Dépendances du projet (pom.xml / build.gradle / requirements) ║
║      • Classes du même package                                       ║
║                                                                      ║
║    Niveau 2 — TRÈS UTILE (si disponible) :                           ║
║      • Analyse Watch depuis le cache SQLite                          ║
║      • Criticité du fichier (nombre de dépendants)                   ║
║                                                                      ║
║    Niveau 3 — OPTIONNEL (si budget tokens restant) :                 ║
║      • Historique episode_memory (patterns récurrents)               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import re
from services.mcp_redis_service import get_mcp_redis, key_hash, KEY_PREFIX
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Budget max en caractères (~1500 tokens ≈ 6000 chars)
MAX_CONTEXT_CHARS = 6000


class ConflictContextBuilder:
    """
    Construit le contexte projet à injecter dans le prompt de résolution.

    Usage :
        builder = ConflictContextBuilder(project_path)
        context = builder.build_context("src/main/java/.../UserService.java")
        # → str prêt à injecter dans le prompt
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._deps_cache: Optional[Set[str]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ─────────────────────────────────────────────────────────────────────────

    def build_context(self, conflict_file: str) -> str:
        """
        Construit le contexte complet pour un fichier en conflit.

        Retourne une string formatée à injecter dans le prompt LLM.
        """
        sections = []
        chars_used = 0

        # ── Niveau 1 — INDISPENSABLE ──────────────────────────────────────
        # 1a. Dépendances du projet
        deps = self._get_project_dependencies()
        if deps:
            dep_section = "AVAILABLE PROJECT DEPENDENCIES:\n" + "\n".join(
                f"  • {d}" for d in sorted(deps)[:30]
            )
            sections.append(dep_section)
            chars_used += len(dep_section)

        # 1b. Entités du même package / fichiers liés
        related = self._get_related_entities(conflict_file)
        if related and chars_used < MAX_CONTEXT_CHARS - 1000:
            sections.append(related)
            chars_used += len(related)

        # 1c. Structure du fichier (champs, méthodes)
        file_info = self._get_file_entities(conflict_file)
        if file_info and chars_used < MAX_CONTEXT_CHARS - 500:
            sections.append(file_info)
            chars_used += len(file_info)

        # ── Niveau 2 — TRÈS UTILE ────────────────────────────────────────
        # 2a. Analyse Watch précédente
        watch_analysis = self._get_watch_analysis(conflict_file)
        if watch_analysis and chars_used < MAX_CONTEXT_CHARS - 800:
            section = (
                f"PREVIOUS ANALYSIS (from Watch mode):\n"
                f"  Known issues in this file:\n{watch_analysis}"
            )
            sections.append(section)
            chars_used += len(section)

        # 2b. Criticité
        criticality = self._get_file_criticality(conflict_file)
        if criticality and chars_used < MAX_CONTEXT_CHARS - 200:
            sections.append(criticality)
            chars_used += len(criticality)

        # ── Niveau 3 — OPTIONNEL ─────────────────────────────────────────
        # 3a. Episode memory (patterns récurrents)
        episodes = self._get_episode_memory(conflict_file)
        if episodes and chars_used < MAX_CONTEXT_CHARS - 300:
            sections.append(episodes)

        if not sections:
            return ""

        header = "=" * 50 + "\nPROJECT CONTEXT (use this to resolve correctly)\n" + "=" * 50
        return header + "\n\n" + "\n\n".join(sections)

    # ─────────────────────────────────────────────────────────────────────────
    # Niveau 1a — Dépendances du projet
    # ─────────────────────────────────────────────────────────────────────────

    def _get_project_dependencies(self) -> Set[str]:
        """
        Lit les dépendances depuis pom.xml, build.gradle, ou requirements.txt.

        Retourne un set de noms de lib (ex: {"spring-boot", "jbcrypt", "slf4j"}).
        """
        if self._deps_cache is not None:
            return self._deps_cache

        deps = set()

        # Maven (pom.xml)
        pom = self.project_path / "pom.xml"
        if pom.exists():
            try:
                content = pom.read_text(encoding="utf-8", errors="replace")
                # Extraire les artifactId des dépendances
                for m in re.finditer(
                    r"<dependency>.*?<artifactId>(.*?)</artifactId>.*?</dependency>",
                    content, re.DOTALL
                ):
                    deps.add(m.group(1).strip())
                # Extraire aussi groupId pour les noms complets
                for m in re.finditer(
                    r"<dependency>.*?<groupId>(.*?)</groupId>.*?</dependency>",
                    content, re.DOTALL
                ):
                    deps.add(m.group(1).strip())
            except Exception as e:
                logger.debug("Erreur lecture pom.xml : %s", e)

        # Gradle (build.gradle)
        for gradle_name in ("build.gradle", "build.gradle.kts"):
            gradle = self.project_path / gradle_name
            if gradle.exists():
                try:
                    content = gradle.read_text(encoding="utf-8", errors="replace")
                    # implementation 'group:artifact:version'
                    for m in re.finditer(
                        r"(?:implementation|api|compile)\s+['\"]([^'\"]+)['\"]",
                        content
                    ):
                        parts = m.group(1).split(":")
                        if len(parts) >= 2:
                            deps.add(parts[1])  # artifactId
                            deps.add(parts[0])  # groupId
                except Exception as e:
                    logger.debug("Erreur lecture %s : %s", gradle_name, e)

        # Python (requirements.txt / pyproject.toml)
        req = self.project_path / "requirements.txt"
        if req.exists():
            try:
                for line in req.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        name = re.split(r"[>=<!\[]", line)[0].strip()
                        if name:
                            deps.add(name)
            except Exception:
                pass

        self._deps_cache = deps
        return deps

    # ─────────────────────────────────────────────────────────────────────────
    # Niveau 1b — Entités liées (même package)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_related_entities(self, conflict_file: str) -> str:
        """
        Lit les entités des fichiers du même package depuis project_indexer.

        Ex: si on résout UserService.java, on récupère les champs de User.java,
        les méthodes de UserController.java, etc.
        """
        # Trouver le répertoire du fichier
        abs_path = (self.project_path / conflict_file).resolve()
        parent_dir = abs_path.parent

        if not parent_dir.exists():
            return ""

        # Lister les fichiers Java/Python du même répertoire
        siblings = []
        for ext in (".java", ".py", ".ts", ".js"):
            siblings.extend(parent_dir.glob(f"*{ext}"))

        if not siblings:
            return ""

        lines = ["CLASSES/FILES IN SAME PACKAGE:"]
        for sibling in sorted(siblings)[:10]:
            if sibling.resolve() == abs_path:
                continue  # skip le fichier en conflit lui-même

            try:
                content = sibling.read_text(encoding="utf-8", errors="replace")
                # Extraire les champs et méthodes (parsing léger)
                entities = self._extract_quick_entities(content, sibling.suffix)
                if entities:
                    lines.append(f"\n  FILE: {sibling.name}")
                    for e in entities[:15]:
                        lines.append(f"    {e}")
            except Exception:
                continue

        return "\n".join(lines) if len(lines) > 1 else ""

    def _extract_quick_entities(self, content: str, suffix: str) -> List[str]:
        """
        Extraction rapide des champs/méthodes d'un fichier.
        
        Pas besoin du parser AST complet — on veut juste les noms
        pour que le LLM sache ce qui existe.
        """
        entities = []

        if suffix == ".java":
            # Classes
            for m in re.finditer(r"(?:public|private|protected)?\s*class\s+(\w+)", content):
                entities.append(f"class {m.group(1)}")

            # Champs (private String username;)
            for m in re.finditer(
                r"(?:private|protected|public)\s+([\w<>,\s]+?)\s+(\w+)\s*[;=]", content
            ):
                field_type = m.group(1).strip()
                field_name = m.group(2).strip()
                if field_name not in ("class", "void", "return", "if", "for"):
                    entities.append(f"field: {field_type} {field_name}")

            # Méthodes (public User findByUsername(String username))
            for m in re.finditer(
                r"(?:public|private|protected)\s+(?:static\s+)?(\w[\w<>,]*)\s+(\w+)\s*\(([^)]*)\)",
                content
            ):
                ret_type = m.group(1)
                name = m.group(2)
                params = m.group(3).strip()
                if name not in ("if", "for", "while", "switch"):
                    entities.append(f"method: {ret_type} {name}({params})")

        elif suffix == ".py":
            # Classes
            for m in re.finditer(r"class\s+(\w+)", content):
                entities.append(f"class {m.group(1)}")
            # Fonctions/méthodes
            for m in re.finditer(r"def\s+(\w+)\s*\(([^)]*)\)", content):
                entities.append(f"def {m.group(1)}({m.group(2)[:50]})")

        return entities

    # ─────────────────────────────────────────────────────────────────────────
    # Niveau 1c — Structure du fichier lui-même (depuis project_indexer cache)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_file_entities(self, conflict_file: str) -> str:
        """Lit les entités indexées du fichier depuis Redis MCP (project_indexer cache)."""
        try:
            redis = get_mcp_redis()
            ps_key = f"{KEY_PREFIX}ps:{key_hash(str(self.project_path))}"
            raw = redis.get(ps_key)
            if not raw:
                return ""

            snapshot = json.loads(raw)
            files_data = snapshot.get("files", {})
            if isinstance(files_data, str):
                files_data = json.loads(files_data)

            for fp, info in files_data.items():
                if conflict_file in fp or Path(fp).name == Path(conflict_file).name:
                    entities = info.get("entities", [])
                    if entities:
                        lines = [f"INDEXED ENTITIES FOR {Path(fp).name}:"]
                        for e in entities[:20]:
                            params = e.get("parameters", [])
                            p_str = ", ".join(params[:4])
                            lines.append(
                                f"  • {e['type']}: {e['name']}"
                                + (f"({p_str})" if params else "")
                            )
                        return "\n".join(lines)
        except Exception as e:
            logger.debug("Erreur lecture project snapshot Redis : %s", e)

        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Niveau 2a — Analyse Watch précédente
    # ─────────────────────────────────────────────────────────────────────────

    def _get_watch_analysis(self, conflict_file: str) -> str:
        """Lit l'analyse précédente depuis Redis MCP."""
        try:
            redis = get_mcp_redis()
            abs_path = str((self.project_path / conflict_file).resolve())
            redis_key = f"{KEY_PREFIX}fc:{key_hash(abs_path)}"
            analysis = redis.hget(redis_key, "analysis_text")
            if analysis:
                return f"  {analysis[:1000]}"
        except Exception:
            pass
        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Niveau 2b — Criticité du fichier
    # ─────────────────────────────────────────────────────────────────────────

    def _get_file_criticality(self, conflict_file: str) -> str:
        """Évalue la criticité du fichier (nombre de fichiers qui en dépendent)."""
        try:
            redis = get_mcp_redis()
            ps_key = f"{KEY_PREFIX}ps:{key_hash(str(self.project_path))}"
            raw = redis.get(ps_key)
            if not raw:
                return ""

            snapshot = json.loads(raw)
            files_data = snapshot.get("files", {})
            if isinstance(files_data, str):
                files_data = json.loads(files_data)

            for fp, info in files_data.items():
                if conflict_file in fp:
                    crit = info.get("criticality", 0)
                    if crit > 0:
                        return (
                            f"FILE CRITICALITY: {crit} other files depend on this file.\n"
                            f"  → Do NOT change public method signatures."
                        )
        except Exception:
            pass

        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Niveau 3 — Episode memory
    # ─────────────────────────────────────────────────────────────────────────

    def _get_episode_memory(self, conflict_file: str) -> str:
        """Lit les patterns récurrents depuis Redis MCP."""
        try:
            redis = get_mcp_redis()
            abs_path = str((self.project_path / conflict_file).resolve())
            scan_pattern = f"{KEY_PREFIX}em:{key_hash(abs_path)}:*"
            keys = redis.scan_keys(scan_pattern)

            if not keys:
                return ""

            results = []
            for k in keys:
                data = redis.hgetall(k)
                results.append((
                    data.get("pattern_type", ""),
                    data.get("severity", "MEDIUM"),
                    int(data.get("occurrence_count", "0")),
                ))

            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:5]

            lines = ["RECURRING PATTERNS (episode_memory):"]
            for pattern, sev, count in results:
                lines.append(f"  • [{sev}] {pattern} — seen {count}x")
            return "\n".join(lines)
        except Exception:
            return ""
