"""
test_discovery.py — Détection des fichiers de test et mapping src ↔ test.

Rôle :
  - Pour un fichier source donné, trouve le fichier de test correspondant
  - Détecte la convention de test du projet (pytest, jest, JUnit, etc.)
  - Calcule la couverture (entités testées / entités publiques)

Conventions supportées :
  Python   : tests/test_{module}.py  |  {module}_test.py  |  test_{module}.py
  JS/TS    : {module}.test.js  |  __tests__/{module}.test.js
  Java     : {module}Test.java  |  Test{Module}.java
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


_TEST_CONVENTIONS: Dict[str, List[Tuple[str, str]]] = {
    "python": [
        ("tests/test_{name}.py", "tests"),
        ("test_{name}.py", "."),
        ("{name}_test.py", "."),
        ("tests/tests_{name}.py", "tests"),
    ],
    "javascript": [
        ("{name}.test.js", "."),
        ("{name}.spec.js", "."),
        ("__tests__/{name}.test.js", "__tests__"),
    ],
    "typescript": [
        ("{name}.test.ts", "."),
        ("{name}.spec.ts", "."),
        ("__tests__/{name}.test.ts", "__tests__"),
    ],
    "java": [
        ("{name}Test.java", "."),
        ("Test{name_cap}.java", "."),
        ("tests/{name}Test.java", "tests"),
    ],
}

# Dossiers de test courants (à chercher si la convention n'est pas évidente)
_COMMON_TEST_DIRS: Set[str] = {
    "tests", "test", "__tests__", "specs", "src/test",
    "src/tests", "app/src/test", "src/androidTest",
}


@dataclass
class TestDiscoveryResult:
    """Résultat de la recherche de tests pour un fichier source."""

    source_file:        Path
    test_file:          Optional[Path] = None
    convention_used:    Optional[str] = None
    entities_tested:    List[str] = field(default_factory=list)
    entities_untested:  List[str] = field(default_factory=list)
    coverage_ratio:     float = 0.0   # 0.0–1.0
    test_framework:     Optional[str] = None  # pytest, jest, junit, unittest…


class TestDiscoveryService:
    """
    Service de découverte des tests.
    Détecte la convention utilisée dans le projet et mappe src ↔ test.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._convention_cache: Dict[str, Tuple[str, str]] = {}
        self._framework_cache: Optional[str] = None

    # ── API publique ─────────────────────────────────────────────────────────

    def find_test_for(self, source_file: Path) -> TestDiscoveryResult:
        """
        Trouve le fichier de test correspondant à `source_file`.
        Détecte automatiquement la convention du projet si c'est la première fois.
        """
        result = TestDiscoveryResult(source_file=source_file)

        lang = self._detect_language(source_file)
        if lang not in _TEST_CONVENTIONS:
            return result  

        # Détecter la convention active dans ce projet (cache)
        if lang not in self._convention_cache:
            self._convention_cache[lang] = self._detect_convention(lang)

        convention, test_dir_hint = self._convention_cache[lang]
        result.convention_used = convention

        # Construire le chemin candidat
        candidate = self._build_candidate(source_file, convention, test_dir_hint)
        if candidate and candidate.exists():
            result.test_file = candidate

        # Détecter le framework de test
        result.test_framework = self._detect_test_framework()

        return result

    def check_coverage(
        self,
        source_file: Path,
        source_entities: List[Dict[str, any]],
    ) -> TestDiscoveryResult:
        """
        Analyse le fichier de test associé et calcule la couverture
        symbolique (quels noms d'entités source apparaissent dans le test).

        C'est une heuristique rapide (0 token LLM) :
        - Lit le fichier de test
        - Compte combien d'entités publiques du source y sont référencées
        """
        result = self.find_test_for(source_file)

        if not result.test_file or not result.test_file.exists():
            # Aucun test trouvé → toutes les entités publiques sont untested
            result.entities_untested = [
                (e["name"] if isinstance(e, dict) else getattr(e, "name", ""))
                for e in source_entities
                if self._is_public_entity(e)
            ]
            result.coverage_ratio = 0.0
            return result

        try:
            test_code = result.test_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Impossible de lire %s : %s", result.test_file, e)
            result.entities_untested = [
                (e["name"] if isinstance(e, dict) else getattr(e, "name", ""))
                for e in source_entities
                if self._is_public_entity(e)
            ]
            return result

        tested = []
        untested = []
        for ent in source_entities:
            if not self._is_public_entity(ent):
                continue
            name = ent.get("name", "") if isinstance(ent, dict) else getattr(ent, "name", "")
            if name and name in test_code:
                tested.append(name)
            else:
                untested.append(name)

        result.entities_tested = tested
        result.entities_untested = untested
        total = len(tested) + len(untested)
        result.coverage_ratio = len(tested) / total if total > 0 else 1.0

        return result

    def has_test_file(self, source_file: Path) -> bool:
        """Vérifie rapidement si un fichier de test existe."""
        res = self.find_test_for(source_file)
        return res.test_file is not None and res.test_file.exists()



    def _detect_convention(self, lang: str) -> Tuple[str, str]:
        """
        Scanne le projet pour trouver la convention de test dominante.
        Retourne (pattern_template, base_dir).
        """
        conventions = _TEST_CONVENTIONS.get(lang, [])
        if not conventions:
            return ("{name}_test.py", ".")

        # Chercher un dossier de test
        test_dir = self._find_test_directory()

        # Compter les matches pour chaque convention
        best = conventions[0]
        best_score = 0

        for pattern, default_dir in conventions:
            score = self._score_convention(pattern, test_dir or default_dir)
            if score > best_score:
                best_score = score
                best = (pattern, test_dir or default_dir)

        logger.debug("Convention détectée pour %s : %s (score=%d)", lang, best[0], best_score)
        return best

    def _find_test_directory(self) -> Optional[str]:
        """Cherche un dossier de test à la racine ou dans src/."""
        for root in [self.project_path] + list(self.project_path.rglob("src")):
            for dirname in _COMMON_TEST_DIRS:
                candidate = root / dirname
                if candidate.is_dir() and any(candidate.iterdir()):
                    return str(candidate.relative_to(self.project_path)).replace("\\", "/")
        return None

    def _score_convention(self, pattern: str, base_dir: str) -> int:
        """Compte combien de fichiers matchent la convention."""
        count = 0
        base = self.project_path / base_dir
        if not base.exists():
            return 0

        # Heuristique : chercher *test* dans le dossier de base
        for f in base.rglob("*"):
            if f.is_file() and "test" in f.name.lower():
                count += 1
            if count >= 3:
                break
        return count

    def _build_candidate(
        self, source_file: Path, convention: str, test_dir_hint: str
    ) -> Optional[Path]:
        """Construit le chemin candidat du fichier de test."""
        name = source_file.stem  # sans extension
        name_cap = name[:1].upper() + name[1:] if name else name

        relative = source_file.relative_to(self.project_path)
        rel_dir = relative.parent

        # Si la convention contient un dossier explicite
        if "/" in convention:
            parts = convention.split("/")
            test_dir = self.project_path / test_dir_hint / parts[0]
            filename = parts[1].format(name=name, name_cap=name_cap)
            return test_dir / filename

        # Sinon : même dossier ou dossier de test
        filename = convention.format(name=name, name_cap=name_cap)
        candidate = self.project_path / test_dir_hint / rel_dir / filename
        if candidate.exists():
            return candidate

        # Fallback : même dossier que le source
        return source_file.parent / filename

    # ── Détection de framework ───────────────────────────────────────────────

    def _detect_test_framework(self) -> Optional[str]:
        """Détecte le framework de test utilisé (pytest, jest, JUnit…)."""
        # Ne pas utiliser le cache si c'est "unknown" — réessayer la détection
        if self._framework_cache and self._framework_cache != "unknown":
            return self._framework_cache

        root = self.project_path

        # Python
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            self._framework_cache = "pytest"
            return self._framework_cache
        if (root / "setup.cfg").exists():
            cfg = (root / "setup.cfg").read_text(errors="ignore")
            if "pytest" in cfg:
                self._framework_cache = "pytest"
                return self._framework_cache
        if (root / "requirements.txt").exists():
            req = (root / "requirements.txt").read_text(errors="ignore")
            if "pytest" in req:
                self._framework_cache = "pytest"
                return self._framework_cache
            if "unittest" in req:
                self._framework_cache = "unittest"
                return self._framework_cache

        # JavaScript / TypeScript
        pkg_json = root / "package.json"
        if pkg_json.exists():
            try:
                import json
                pkg = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "jest" in deps:
                    self._framework_cache = "jest"
                    return self._framework_cache
                if "mocha" in deps:
                    self._framework_cache = "mocha"
                    return self._framework_cache
                if "vitest" in deps:
                    self._framework_cache = "vitest"
                    return self._framework_cache
            except Exception:
                pass

        # Java — chercher pom.xml ou build.gradle (même en profondeur)
        for pom in root.rglob("pom.xml"):
            try:
                if "junit" in pom.read_text(errors="ignore").lower():
                    self._framework_cache = "junit"
                    return self._framework_cache
            except Exception:
                continue
        for gradle in root.rglob("build.gradle"):
            try:
                if "junit" in gradle.read_text(errors="ignore").lower():
                    self._framework_cache = "junit"
                    return self._framework_cache
            except Exception:
                continue

        # Détection via fichiers de test existants
        for test_file in root.rglob("*Test.java"):
            try:
                content = test_file.read_text(errors="ignore")
                if "org.junit" in content or "import org.junit" in content:
                    self._framework_cache = "junit"
                    return self._framework_cache
            except Exception:
                continue

        self._framework_cache = "unknown"
        return self._framework_cache

   
    @staticmethod
    def _detect_language(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
        }
        return mapping.get(ext, "unknown")

    @staticmethod
    def _is_public_entity(entity) -> bool:
        """Détermine si une entité doit être testée (publique, non-helper)."""
        # Gère à la fois les dict et les objets dataclass (CodeEntity)
        if isinstance(entity, dict):
            etype = entity.get("type", "")
            name = entity.get("name", "")
        else:
            etype = getattr(entity, "type", "")
            name = getattr(entity, "name", "")

        # Ignorer les privés / internes
        if name.startswith("_"):
            return False
        if etype not in ("function", "method", "class", "constructor"):
            return False

        # Ignorer les getters/setters triviaux
        if name.startswith("get") or name.startswith("set"):
            return False

        return True


# Instance singleton (initialisée dans Orchestrator.initialize())
test_discovery = TestDiscoveryService(Path("."))
