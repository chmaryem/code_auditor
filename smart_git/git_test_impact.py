"""
git_test_impact.py — F4 : Analyse d'impact sur les tests.

Rôle :
  Pour chaque fichier modifié dans le diff staged/branch, trouve les fichiers
  de test correspondants par deux stratégies complémentaires :
    1. Convention de nommage : src/auth.py → tests/test_auth.py
    2. Import grep : cherche les fichiers de test qui importent le module modifié

Architecture :
  - Analyse 100% locale (lecture fichiers + grep) — 0 token LLM
  - Accessible via le LangGraph (intent: test_impact)
  - Retourne la liste des tests impactés + ceux qui sont absents (coverage gap)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# ── Patterns de nommage des tests ─────────────────────────────────────────────

TEST_NAMING_PATTERNS = [
    # Python
    ("test_{stem}.py",       "{stem}"),
    ("{stem}_test.py",       "{stem}"),
    ("test_{stem}.py",       "{stem}"),
    # Java/Kotlin
    ("{Stem}Test.java",      "{Stem}"),
    ("{Stem}Spec.java",      "{Stem}"),
    ("{Stem}Tests.java",     "{Stem}"),
    # JS/TS
    ("{stem}.test.ts",       "{stem}"),
    ("{stem}.spec.ts",       "{stem}"),
    ("{stem}.test.js",       "{stem}"),
    ("{stem}.spec.js",       "{stem}"),
    ("__tests__/{stem}.ts",  "{stem}"),
    ("__tests__/{stem}.js",  "{stem}"),
]

# Répertoires de tests reconnus
TEST_DIR_NAMES = {"tests", "test", "__tests__", "spec", "specs", "src/test", "src/tests"}

# Extensions de fichiers source surveillées
SOURCE_EXTENSIONS = {".py", ".ts", ".js", ".tsx", ".jsx", ".java", ".kt", ".go"}


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class TestImpact:
    """Impact sur les tests pour un fichier source modifié."""
    source_file:      str
    test_files:       List[str] = field(default_factory=list)   # tests trouvés
    missing_tests:    bool = False    # aucun test trouvé → coverage gap
    discovery_method: List[str] = field(default_factory=list)   # naming | import | existing


@dataclass
class TestImpactReport:
    """Rapport complet d'analyse d'impact sur les tests."""
    impacts:         List[TestImpact] = field(default_factory=list)
    total_files:     int = 0
    covered_files:   int = 0
    uncovered_files: int = 0
    all_test_files:  List[str] = field(default_factory=list)   # liste dédupliquée
    error:           Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def coverage_ratio(self) -> float:
        if self.total_files == 0:
            return 1.0
        return self.covered_files / self.total_files

    @property
    def has_gaps(self) -> bool:
        return self.uncovered_files > 0


# ── Analyseur ─────────────────────────────────────────────────────────────────

class GitTestImpactAnalyzer:
    """
    Trouve les fichiers de test impactés par les modifications staged.

    Stratégies (appliquées dans l'ordre) :
      1. Convention de nommage — cherche test_<stem>.py, <stem>.test.ts, etc.
         dans tous les répertoires de tests du projet.
      2. Import grep — cherche les fichiers de test qui importent le module modifié.
         Ex: "from auth import" ou "import auth" ou "require('./auth')".
    """

    def analyze_staged(self, project_path: Path) -> TestImpactReport:
        """Analyse les fichiers stagés et retourne le rapport d'impact."""
        report = TestImpactReport()

        try:
            # Récupérer les fichiers modifiés stagés
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, cwd=str(project_path),
            )
            if result.returncode != 0:
                report.error = f"git diff failed: {result.stderr.strip()}"
                return report

            changed = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
            source_files = [f for f in changed if Path(f).suffix.lower() in SOURCE_EXTENSIONS]

            if not source_files:
                return report

            report.total_files = len(source_files)

            # Indexer les répertoires de tests existants
            test_dirs = self._find_test_dirs(project_path)

            # Analyser chaque fichier source
            all_tests: Set[str] = set()
            for src_file in source_files:
                impact = self._analyze_file(src_file, project_path, test_dirs)
                report.impacts.append(impact)
                all_tests.update(impact.test_files)

                if impact.test_files:
                    report.covered_files += 1
                else:
                    report.uncovered_files += 1
                    impact.missing_tests = True

            report.all_test_files = sorted(all_tests)

        except Exception as e:
            report.error = str(e)

        return report

    def analyze_files(self, source_files: List[str], project_path: Path) -> TestImpactReport:
        """Analyse une liste de fichiers source (mode branche/PR)."""
        report = TestImpactReport()

        try:
            src = [f for f in source_files if Path(f).suffix.lower() in SOURCE_EXTENSIONS]
            if not src:
                return report

            report.total_files = len(src)
            test_dirs = self._find_test_dirs(project_path)
            all_tests: Set[str] = set()

            for src_file in src:
                impact = self._analyze_file(src_file, project_path, test_dirs)
                report.impacts.append(impact)
                all_tests.update(impact.test_files)

                if impact.test_files:
                    report.covered_files += 1
                else:
                    report.uncovered_files += 1
                    impact.missing_tests = True

            report.all_test_files = sorted(all_tests)

        except Exception as e:
            report.error = str(e)

        return report

    # ── Analyse d'un fichier source ───────────────────────────────────────────

    def _analyze_file(
        self, src_file: str, project_path: Path, test_dirs: List[Path]
    ) -> TestImpact:
        impact = TestImpact(source_file=src_file)
        stem = Path(src_file).stem
        ext  = Path(src_file).suffix.lower()

        # Stratégie 1 : Convention de nommage
        naming_tests = self._find_by_naming(stem, ext, test_dirs, project_path)
        if naming_tests:
            impact.test_files.extend(naming_tests)
            impact.discovery_method.append("naming")

        # Stratégie 2 : Import grep
        import_tests = self._find_by_imports(src_file, stem, project_path)
        for t in import_tests:
            if t not in impact.test_files:
                impact.test_files.append(t)
        if import_tests:
            impact.discovery_method.append("import")

        return impact

    # ── Stratégie 1 : Convention de nommage ──────────────────────────────────

    def _find_by_naming(
        self, stem: str, ext: str, test_dirs: List[Path], project_path: Path
    ) -> List[str]:
        """Cherche les fichiers de test par convention de nommage."""
        found = []
        stem_cap = stem[0].upper() + stem[1:] if stem else stem

        # Patterns selon l'extension
        if ext == ".py":
            candidates = [f"test_{stem}.py", f"{stem}_test.py"]
        elif ext in (".ts", ".tsx"):
            candidates = [f"{stem}.test.ts", f"{stem}.spec.ts", f"{stem}.test.tsx"]
        elif ext in (".js", ".jsx"):
            candidates = [f"{stem}.test.js", f"{stem}.spec.js", f"{stem}.test.jsx"]
        elif ext == ".java":
            candidates = [f"{stem_cap}Test.java", f"{stem_cap}Tests.java", f"{stem_cap}Spec.java"]
        else:
            candidates = [f"test_{stem}{ext}", f"{stem}_test{ext}"]

        for test_dir in test_dirs:
            for candidate in candidates:
                test_path = test_dir / candidate
                if test_path.exists():
                    rel = str(test_path.relative_to(project_path)).replace("\\", "/")
                    found.append(rel)

                # Chercher aussi dans __tests__/ sous-répertoires
                for sub in test_dir.rglob(candidate):
                    rel = str(sub.relative_to(project_path)).replace("\\", "/")
                    if rel not in found:
                        found.append(rel)

        return found

    # ── Stratégie 2 : Import grep ─────────────────────────────────────────────

    def _find_by_imports(
        self, src_file: str, stem: str, project_path: Path
    ) -> List[str]:
        """
        Cherche les fichiers de test qui importent le module modifié.
        Utilise git grep pour rester dans le contexte du dépôt.
        """
        found = []

        # Patterns d'import selon le langage
        ext = Path(src_file).suffix.lower()
        if ext == ".py":
            patterns = [
                f"from {stem} import",
                f"import {stem}",
                f"from .{stem} import",
            ]
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            module_path = src_file.replace("\\", "/").replace(ext, "")
            patterns = [
                f"from './{stem}'",
                f"from \"./{stem}\"",
                f"require('./{stem}')",
                f"require(\"./{stem}\")",
                f"from '{module_path}'",
            ]
        elif ext == ".java":
            class_name = Path(src_file).stem
            patterns = [f"import.*{class_name}"]
        else:
            patterns = [f"import.*{stem}", f"require.*{stem}"]

        try:
            for pattern in patterns:
                result = subprocess.run(
                    ["git", "grep", "-l", "--", pattern],
                    capture_output=True, text=True, cwd=str(project_path),
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        f = line.strip().replace("\\", "/")
                        # Ne garder que les fichiers de test
                        if self._is_test_file(f) and f not in found and f != src_file:
                            found.append(f)
        except Exception:
            pass

        return found

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _find_test_dirs(self, project_path: Path) -> List[Path]:
        """Trouve tous les répertoires de tests dans le projet."""
        test_dirs = []
        for name in TEST_DIR_NAMES:
            p = project_path / name
            if p.is_dir():
                test_dirs.append(p)

        # Chercher récursivement (max 3 niveaux)
        for depth in range(3):
            for p in project_path.rglob("*"):
                if p.is_dir() and p.name in {"tests", "test", "__tests__"}:
                    if p not in test_dirs:
                        test_dirs.append(p)

        return test_dirs

    def _is_test_file(self, file_path: str) -> bool:
        """Retourne True si le fichier est un fichier de test."""
        p = file_path.replace("\\", "/").lower()
        return (
            "/test" in p or "/spec" in p or "/__tests__/" in p
            or p.endswith((".test.py", "_test.py", "test_.py"))
            or p.endswith((".test.ts", ".spec.ts", ".test.js", ".spec.js"))
            or bool(re.search(r"test[s]?[/\\]", p))
        )


# ── Point d'entrée public ─────────────────────────────────────────────────────

_analyzer = GitTestImpactAnalyzer()


def analyze_test_impact_staged(project_path: Path) -> TestImpactReport:
    """Analyse d'impact des tests pour les fichiers stagés."""
    return _analyzer.analyze_staged(project_path)


def analyze_test_impact_files(source_files: List[str], project_path: Path) -> TestImpactReport:
    """Analyse d'impact des tests pour une liste de fichiers."""
    return _analyzer.analyze_files(source_files, project_path)
