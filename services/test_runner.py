"""
test_runner.py — Exécution des tests générés et capture des erreurs runtime.

Rôle :
  Après génération LLM, exécuter les tests dans un subprocess isolé et
  retourner un rapport structuré. Si les tests échouent, le rapport contient
  le message d'erreur exact (ImportError, AssertionError, ligne) pour que
  TestGeneratorAgent puisse faire un retry LLM ciblé.

Langages supportés :
  - Python  : pytest --tb=short -x (stop au premier échec)
  - JS/TS   : npx jest --no-coverage (si package.json trouvé)
  - Java    : mvn test / gradle test (si build file trouvé)

Sécurité :
  - Timeout 60s max par exécution
  - Subprocess isolé (pas de shell=True pour éviter l'injection)
  - Répertoire de travail = project_path
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60


@dataclass
class TestRunResult:
    """Résultat de l'exécution d'un fichier de test."""

    success: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    error_summary: str = ""         # Message d'erreur condensé pour le retry LLM
    failing_tests: List[str] = field(default_factory=list)
    # Résultat par test : [{"name": str, "status": "pass"|"fail"|"skip", "error": str|None}, ...]
    # Alimenté pour Python (pytest -v) ; vide pour les langages non encore parsés → le
    # consommateur retombe sur le booléen success (comportement historique).
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    language: str = ""

    @property
    def has_import_error(self) -> bool:
        combined = self.stdout + self.stderr
        return "ImportError" in combined or "ModuleNotFoundError" in combined

    @property
    def has_attribute_error(self) -> bool:
        combined = self.stdout + self.stderr
        return "AttributeError" in combined

    @property
    def has_syntax_error(self) -> bool:
        combined = self.stdout + self.stderr
        return "SyntaxError" in combined


class TestRunner:
    """
    Exécute les tests générés dans un subprocess et retourne un rapport structuré.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path

    # ── API publique ─────────────────────────────────────────────────────────

    def run(self, test_file: Path, language: str) -> TestRunResult:
        """
        Exécute le fichier de test et retourne le résultat.

        Args:
            test_file: Chemin absolu du fichier de test à exécuter.
            language: 'python', 'javascript', 'typescript', 'java'

        Returns:
            TestRunResult avec success=True si tous les tests passent.
        """
        if not test_file.exists():
            return TestRunResult(
                success=False, returncode=-1,
                error_summary=f"Fichier de test introuvable : {test_file}",
                language=language,
            )

        logger.info("TestRunner: execution de %s (%s)", test_file.name, language)
        start = time.time()

        if language == "python":
            result = self._run_pytest(test_file)
        elif language in ("javascript", "typescript"):
            result = self._run_jest(test_file)
        elif language == "java":
            result = self._run_java(test_file)
        else:
            return TestRunResult(
                success=True, returncode=0,
                error_summary="Langage non supporté pour l'exécution — skipped",
                language=language,
            )

        result.duration_seconds = round(time.time() - start, 2)
        result.language = language
        if language == "python":
            result.test_results = self._parse_pytest_results(result.stdout)
        result.error_summary = self._build_error_summary(result)

        logger.info(
            "TestRunner: %s en %.1fs — %s",
            "SUCCES" if result.success else "ECHEC",
            result.duration_seconds,
            result.error_summary[:120] if not result.success else "OK",
        )
        return result

    # ── Python (pytest) ──────────────────────────────────────────────────────

    def _run_pytest(self, test_file: Path) -> TestRunResult:
        """Exécute pytest en mode verbeux (tous les tests, résultat par test)."""
        python_exe = self._find_python()
        cmd = [
            python_exe, "-m", "pytest",
            str(test_file),
            "-v",            # une ligne PASSED/FAILED par test → parsable
            "--tb=short",    # traceback court — assez pour le LLM
            "--no-header",
            # NB : pas de -x — on exécute TOUS les tests pour un rapport complet
            # par test (sinon les tests après le 1er échec ne tournent jamais).
        ]
        return self._subprocess(cmd)

    # ── Parsing du résultat par test (pytest -v) ─────────────────────────────

    _PYTEST_LINE_RE = re.compile(
        r"^(?P<file>.+?)::(?P<name>[\w\[\]\-.]+)\s+"
        r"(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
    )
    _PYTEST_SUMMARY_RE = re.compile(
        r"^(?:FAILED|ERROR)\s+.+?::(?P<name>[\w\[\]\-.]+)\s*-\s*(?P<msg>.+)$"
    )

    def _parse_pytest_results(self, stdout: str) -> List[Dict[str, Any]]:
        """
        Extrait le statut par test depuis la sortie `pytest -v`.
        Retourne [] si rien n'est parsable (ex. erreur de collection) → le
        consommateur retombe alors sur le booléen success (comportement legacy).
        """
        if not stdout:
            return []

        status_map = {
            "PASSED": "pass", "XPASS": "pass",
            "FAILED": "fail", "ERROR": "fail", "XFAIL": "fail",
            "SKIPPED": "skip",
        }

        results: List[Dict[str, Any]] = []
        seen: set = set()
        for line in stdout.splitlines():
            m = self._PYTEST_LINE_RE.match(line.strip())
            if not m:
                continue
            name = m.group("name")
            if name in seen:
                continue
            seen.add(name)
            results.append({
                "name": name,
                "status": status_map.get(m.group("status"), "fail"),
                "error": None,
            })

        # Enrichir le message d'échec par test depuis le "short test summary info"
        errors_by_name: Dict[str, str] = {}
        for line in stdout.splitlines():
            sm = self._PYTEST_SUMMARY_RE.match(line.strip())
            if sm:
                errors_by_name.setdefault(sm.group("name"), sm.group("msg").strip())
        for r in results:
            if r["status"] == "fail" and r["name"] in errors_by_name:
                r["error"] = errors_by_name[r["name"]]

        return results

    def _find_python(self) -> str:
        """Cherche le python du venv du projet, sinon sys.executable."""
        venv_python = self.project_path / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)
        venv_python_unix = self.project_path / ".venv" / "bin" / "python"
        if venv_python_unix.exists():
            return str(venv_python_unix)
        return sys.executable

    # ── JavaScript / TypeScript (jest) ──────────────────────────────────────

    def _run_jest(self, test_file: Path) -> TestRunResult:
        """Exécute jest sur le fichier de test."""
        pkg_json = self.project_path / "package.json"
        if not pkg_json.exists():
            return TestRunResult(
                success=False, returncode=-1,
                error_summary="package.json introuvable — jest non exécutable",
            )

        # Utiliser npx pour ne pas dépendre d'une installation globale
        cmd = ["npx", "--yes", "jest",
               str(test_file),
               "--no-coverage",
               "--forceExit",
               "--testTimeout=30000"]
        return self._subprocess(cmd)

    # ── Java (Maven / Gradle) ────────────────────────────────────────────────

    def _run_java(self, test_file: Path) -> TestRunResult:
        """Exécute Maven ou Gradle test sur le fichier de test."""
        # Extraire le nom de la classe de test depuis le chemin
        class_name = test_file.stem   # ex: UserServiceTest

        mvnw = self.project_path / "mvnw"
        pom = self.project_path / "pom.xml"
        gradlew = self.project_path / "gradlew"
        build_gradle = self.project_path / "build.gradle"

        if pom.exists():
            exe = str(mvnw) if mvnw.exists() else "mvn"
            cmd = [exe, "test", f"-Dtest={class_name}", "-pl", ".", "-am", "-q"]
        elif build_gradle.exists():
            exe = str(gradlew) if gradlew.exists() else "gradle"
            cmd = [exe, "test", f"--tests={class_name}", "-q"]
        else:
            return TestRunResult(
                success=False, returncode=-1,
                error_summary="Aucun pom.xml ou build.gradle trouvé — Java non exécutable",
            )

        return self._subprocess(cmd)

    # ── Subprocess générique ──────────────────────────────────────────────────

    def _subprocess(self, cmd: List[str]) -> TestRunResult:
        """Exécute une commande dans un subprocess avec timeout."""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(self.project_path),
            )
            return TestRunResult(
                success=proc.returncode == 0,
                returncode=proc.returncode,
                stdout=proc.stdout[-4000:],   # limiter à 4000 chars pour le contexte LLM
                stderr=proc.stderr[-2000:],
            )
        except subprocess.TimeoutExpired:
            return TestRunResult(
                success=False, returncode=-1,
                error_summary=f"Timeout ({TIMEOUT_SECONDS}s) — tests trop lents ou boucle infinie",
            )
        except FileNotFoundError as e:
            return TestRunResult(
                success=False, returncode=-1,
                error_summary=f"Commande introuvable : {e}",
            )
        except Exception as e:
            return TestRunResult(
                success=False, returncode=-1,
                error_summary=f"Erreur subprocess : {e}",
            )

    # ── Analyse du rapport d'erreur ──────────────────────────────────────────

    def _build_error_summary(self, result: TestRunResult) -> str:
        """
        Construit un résumé d'erreur condensé et exploitable par le LLM.
        Extrait les lignes les plus informatives du rapport pytest/jest.
        """
        if result.success:
            return "Tous les tests passent"

        combined = (result.stdout + "\n" + result.stderr).strip()
        if not combined:
            return f"Échec sans sortie (returncode={result.returncode})"

        lines = combined.splitlines()
        relevant: List[str] = []

        # Lignes prioritaires : erreurs typées
        error_keywords = (
            "Error:", "Exception:", "FAILED", "ERROR",
            "ImportError", "ModuleNotFoundError",
            "AttributeError", "TypeError", "SyntaxError",
            "AssertionError", "assert ",
            "E  ",           # format pytest short tb
            "COMPILATION ERROR",  # Maven
            "error TS",           # TypeScript
        )

        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in error_keywords):
                relevant.append(stripped)

        # Si on n'a rien trouvé, prendre les 15 dernières lignes
        if not relevant:
            relevant = [l.strip() for l in lines[-15:] if l.strip()]

        # Dédupliquer et limiter
        seen = set()
        unique = []
        for l in relevant:
            if l not in seen:
                seen.add(l)
                unique.append(l)
            if len(unique) >= 20:
                break

        return "\n".join(unique)
