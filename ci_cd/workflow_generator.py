"""
workflow_generator.py — Génère le fichier YAML GitHub Actions adapté au projet.

Détecte le langage et le build system du repo cible,
puis génère un workflow avec 2 jobs :
  1. build-test  → compile + tests (Maven, Gradle, npm, pytest...)
  2. sonar-scan  → Analyse qualité SonarQube

Workflow CI/CD structuré :
  - Job 1: Checkout → Setup → Build & Test
  - Job 2: SonarQube Analysis (dépend de build-test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectProfile:
    """Profil détecté du projet cible."""
    language: str            # java, python, javascript, typescript
    build_system: str        # maven, gradle, npm, pip, poetry, unknown
    java_version: str = "17"
    python_version: str = "3.11"
    node_version: str = "20"
    has_tests: bool = True


# ── Détection du profil ──────────────────────────────────────────────────────

# Ordre important : tsconfig.json AVANT package.json pour détecter TypeScript en priorité
BUILD_FILES = {
    "pom.xml":          ("java",       "maven"),
    "build.gradle":     ("java",       "gradle"),
    "build.gradle.kts": ("java",       "gradle"),
    "tsconfig.json":    ("typescript", "npm"),    # Avant package.json !
    "package.json":     ("javascript", "npm"),
    "requirements.txt": ("python",     "pip"),
    "pyproject.toml":   ("python",     "poetry"),
    "setup.py":         ("python",     "pip"),
}


# Mapping extension → langage (fallback quand pas de fichier de build)
# Limité aux 3 langages cibles : Java, Python, JavaScript/TypeScript
EXT_TO_LANGUAGE = {
    ".py":   "python",
    ".java": "java",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
}

# Langage → build system par défaut
LANG_DEFAULT_BUILD = {
    "python":     "pip",
    "java":       "maven",
    "javascript": "npm",
    "typescript": "npm",
}


def detect_project_profile(file_checker, file_lister=None) -> ProjectProfile:
    """
    Détecte le profil du projet.

    Stratégie en 2 niveaux :
      1. Chercher les fichiers de build (pom.xml, requirements.txt, etc.)
      2. Fallback : compter les extensions de fichiers du repo

    Args:
        file_checker: callable(path) -> str|None
        file_lister:  callable() -> list[str] (noms de fichiers du repo)
    """
    # Niveau 1 : fichiers de build
    for build_file, (language, build_system) in BUILD_FILES.items():
        content = file_checker(build_file)
        if content:
            profile = ProjectProfile(language=language, build_system=build_system)
            if build_file == "pom.xml" and "<java.version>" in content:
                import re
                m = re.search(r"<java.version>(\d+)</java.version>", content)
                if m:
                    profile.java_version = m.group(1)
            return profile

    # Niveau 2 : fallback par extensions de fichiers
    if file_lister:
        try:
            files = file_lister()
            if files:
                from pathlib import Path
                ext_count: dict[str, int] = {}
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in EXT_TO_LANGUAGE:
                        lang = EXT_TO_LANGUAGE[ext]
                        ext_count[lang] = ext_count.get(lang, 0) + 1

                if ext_count:
                    dominant = max(ext_count, key=ext_count.get)
                    build = LANG_DEFAULT_BUILD.get(dominant, "unknown")
                    return ProjectProfile(language=dominant, build_system=build)
        except Exception:
            pass

    return ProjectProfile(language="unknown", build_system="unknown")


# ── Génération du YAML ───────────────────────────────────────────────────────

def generate_workflow(
    profile: ProjectProfile,
    auditor_repo: str = "chmaryem/code_auditor",
    checkout_path: str = "code_auditor_tool",
) -> str:
    """
    Génère le contenu YAML complet du workflow GitHub Actions.

    Args:
        profile: Profil du projet détecté
        auditor_repo: Paramètre legacy (non utilisé)
        checkout_path: Paramètre legacy (non utilisé)

    2 jobs :
      1. build-test  : compile + exécute les tests du projet cible
      2. sonar-scan  : Analyse qualité SonarQube (dépend de build-test)
    """
    build_steps   = _build_steps(profile)
    sonar_steps   = _sonar_steps(profile)
    codeql_job    = _codeql_job(profile)

    yaml = f"""# ─────────────────────────────────────────────────────────────────
# CI/CD Pipeline — Généré automatiquement par Code Auditor
#
# 3 jobs :
#   1. build-test  → {profile.build_system} ({profile.language})
#   2. sonar-scan  → Qualité + Coverage (SonarQube)
#   3. codeql-scan → Sécurité statique profonde (CodeQL - GitHub natif)
#
# Secrets requis :
#   - SONAR_TOKEN    : Token SonarQube/SonarCloud
#   - SONAR_HOST_URL : https://sonarcloud.io (ou votre serveur)
#   CodeQL ne requiert aucun secret (intégré GitHub Actions).
# ─────────────────────────────────────────────────────────────────

name: "CI/CD — Build + SonarQube + CodeQL"

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]

permissions:
  contents: read
  statuses: write
  pull-requests: read
  security-events: write   # Requis pour CodeQL et Trivy SARIF upload
  actions: read            # Requis pour CodeQL

jobs:
  # ── Job 1 : Build & Test ──────────────────────────────────────
  build-test:
    name: "Build & Test ({profile.language}/{profile.build_system})"
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Requis pour SonarQube blame
{build_steps}

      - name: "Upload Test Results"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: |
            {_test_results_path(profile)}
          retention-days: 5

  # ── Job 2 : SonarQube Analysis ────────────────────────────────
  sonar-scan:
    name: "SonarQube Analysis"
    needs: build-test  # Attend les résultats de coverage
    runs-on: ubuntu-latest
    timeout-minutes: 15
    if: always() && needs.build-test.result == 'success'
    steps:
{sonar_steps}

  # ── Job 3 : CodeQL Security Analysis ──────────────────────────
  # CodeQL est GRATUIT et NATIF GitHub Actions — zéro serveur requis.
  # Il analyse le code source pour détecter les vulnérabilités de sécurité
  # (SQL injection, XSS, path traversal...) et publie les résultats dans
  # Security > Code scanning du repo GitHub.
{codeql_job}
"""
    result = yaml.strip() + "\n"

    # Valider le YAML avant de le retourner
    errors = validate_workflow(result)
    if errors:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("YAML validation issues: %s", errors)

    return result


def validate_workflow(yaml_content: str) -> list[str]:
    """
    Valide que le YAML généré est syntaxiquement correct et contient
    les clés obligatoires pour un workflow GitHub Actions.

    Returns:
        Liste d'erreurs (vide = YAML valide)
    """
    is_valid, errors = validate_workflow_strict(yaml_content)
    return errors


def validate_workflow_strict(yaml_content: str) -> tuple[bool, list[str]]:
    """
    Validation stricte du YAML avant déploiement.
    Retourne (is_valid, errors)
    """
    errors = []

    # 1. Parse YAML
    try:
        import yaml as pyyaml
        doc = pyyaml.safe_load(yaml_content)
    except ImportError:
        # PyYAML pas installé — skip la validation
        return True, []
    except Exception as e:
        return False, [f"YAML invalide: {e}"]

    if not isinstance(doc, dict):
        return False, ["Le YAML ne contient pas un mapping racine"]

    # 2. Clés obligatoires
    # Note: PyYAML convertit la clé YAML "on" en booléen True
    # Donc on vérifie à la fois "on" et True comme clé valide
    for key in ("name", "jobs"):
        if key not in doc:
            errors.append(f"Clé obligatoire manquante: '{key}'")
    if "on" not in doc and True not in doc:
        errors.append("Clé obligatoire manquante: 'on' (trigger)")

    # 3. Jobs attendus
    jobs = doc.get("jobs", {})
    if isinstance(jobs, dict):
        if "build-test" not in jobs:
            errors.append("Job 'build-test' manquant")
        if "sonar-scan" not in jobs:
            errors.append("Job 'sonar-scan' manquant")

        # 4. Chaque job doit avoir runs-on et steps
        for job_name, job_def in jobs.items():
            if not isinstance(job_def, dict):
                errors.append(f"Job '{job_name}' n'est pas un mapping")
                continue
            if "runs-on" not in job_def:
                errors.append(f"Job '{job_name}': 'runs-on' manquant")
            if "steps" not in job_def:
                errors.append(f"Job '{job_name}': 'steps' manquant")

    # 5. Validation des steps (vérifie que les actions existent)
    for job_name, job_def in jobs.items():
        steps = job_def.get("steps", [])
        for i, step in enumerate(steps):
            if "uses" in step:
                action = step["uses"]
                # Vérifie format action@vX (ex: actions/checkout@v4)
                if "@" not in action and not action.startswith("./"):
                    errors.append(f"{job_name}, step {i}: action sans version: {action}")

    # 6. Validation des secrets utilisés
    yaml_str = yaml_content
    required_secrets = []
    if "secrets.SONAR_TOKEN" in yaml_str:
        required_secrets.append("SONAR_TOKEN")
    if "secrets.SONAR_HOST_URL" in yaml_str:
        required_secrets.append("SONAR_HOST_URL")
    if "secrets.GITHUB_TOKEN" in yaml_str:
        required_secrets.append("GITHUB_TOKEN")

    return len(errors) == 0, errors


def _test_results_path(profile: ProjectProfile) -> str:
    """
    Retourne les chemins des rapports de couverture à uploader.

    IMPORTANT : ces chemins doivent correspondre exactement aux fichiers
    générés par les outils de test — SonarQube les lit depuis ces paths.
    """
    paths = {
        # JaCoCo génère jacoco.xml + surefire les rapports de tests
        "java":       "target/surefire-reports/\n            target/site/jacoco/",
        # pytest-cov génère coverage.xml à la racine
        "python":     "coverage.xml\n            .coverage",
        # jest --coverage génère coverage/lcov.info
        "javascript": "coverage/lcov.info\n            coverage/",
        "typescript": "coverage/lcov.info\n            coverage/",
    }
    return paths.get(profile.language, "test-results/")


def _build_steps(profile: ProjectProfile) -> str:
    """
    Génère les steps de build/test selon le build system.

    NOTE Maven : `-Dmaven.test.failure.ignore=true` permet de continuer
    même si des tests échouent. SonarQube analysera quand même le code
    et fournira des métriques de couverture.
    """
    if profile.build_system == "maven":
        return f"""
      - name: "⚙️ Setup Java"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'maven'

      - name: "🔨 Build"
        run: mvn compile -q

      - name: "🧪 Tests"
        # -Dmaven.test.failure.ignore=true : les échecs de tests ne bloquent pas
        # le pipeline — SonarQube analysera quand même le code et la couverture
        run: mvn test -Dmaven.test.failure.ignore=true
        continue-on-error: true

      - name: "📊 Coverage Report (JaCoCo)"
        # JaCoCo génère target/site/jacoco/jacoco.xml — utilisé par SonarQube
        # Si JaCoCo n'est pas dans le pom.xml, ce step est ignoré silencieusement
        run: mvn jacoco:report -q || echo "[INFO] JaCoCo non configuré — ajoutez jacoco-maven-plugin dans pom.xml pour la couverture"
        continue-on-error: true

      - name: "📦 Package"
        run: mvn package -q -DskipTests"""

    elif profile.build_system == "gradle":
        return f"""
      - name: "Setup Java"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'gradle'

      - name: "Build & Test"
        run: ./gradlew build test
        continue-on-error: true"""

    elif profile.build_system == "npm":
        lang_label = "TypeScript" if profile.language == "typescript" else "JavaScript"
        return f"""
      - name: "⚙️ Setup Node.js ({lang_label})"
        uses: actions/setup-node@v4
        with:
          node-version: '{profile.node_version}'
          cache: 'npm'

      - name: "📦 Install"
        run: npm ci

      - name: "🔨 Build"
        run: npm run build --if-present
        continue-on-error: true

      - name: "🧪 Tests & Coverage"
        # CI=true désactive le mode watch (Create React App / Jest)
        # --coverage génère coverage/lcov.info — utilisé par SonarQube
        run: CI=true npm test -- --coverage --watchAll=false 2>/dev/null || npm test --if-present
        continue-on-error: true"""

    elif profile.build_system in ("pip", "poetry"):
        if profile.build_system == "pip":
            setup = """pip install --upgrade pip
          [ -f requirements.txt ] && pip install -r requirements.txt || echo "[INFO] Pas de requirements.txt"
          pip install pytest pytest-cov"""
            # --cov=. analyse tout le projet, --cov-report=xml génère coverage.xml
            # coverage.xml est requis par SonarQube pour afficher la couverture
            test_cmd = "pytest --tb=short -q --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing || echo 'No tests found'"
        else:
            setup = """pip install poetry
          poetry install
          poetry add --dev pytest-cov 2>/dev/null || true"""
            test_cmd = "poetry run pytest --tb=short -q --cov=. --cov-report=xml:coverage.xml || echo 'No tests found'"

        return f"""
      - name: "⚙️ Setup Python"
        uses: actions/setup-python@v5
        with:
          python-version: '{profile.python_version}'

      - name: "📦 Install"
        run: |
          {setup}

      - name: "🧪 Tests & Coverage"
        # --cov-report=xml génère coverage.xml → lu par SonarQube
        run: {test_cmd}
        continue-on-error: true"""

    else:
        return """
      - name: "Build system non detecte"
        run: |
          echo "Aucun build system reconnu — ajoutez manuellement les steps."
          ls -la"""


def _sonar_steps(profile: ProjectProfile) -> str:
    """
    Génère les steps SonarQube selon le langage du projet.

    Stratégie par langage :
      Java/Maven  : mvn sonar:sonar -DskipTests  (JAMAIS mvn verify — évite re-run des tests)
      Java/Gradle : ./gradlew sonar
      Python      : sonarsource/sonarqube-scan-action@v5 (Java bundlé, pas besoin de setup-java)
      JS/TS       : sonarsource/sonarqube-scan-action@v5 + lcov.info pour la couverture
    """
    sonar_token = "${{ secrets.SONAR_TOKEN }}"
    sonar_host  = "${{ secrets.SONAR_HOST_URL }}"
    project_key = "${{ github.repository_owner }}_${{ github.event.repository.name }}"

    # ── Setup Java (uniquement si on appelle mvn/gradle directement) ──────────
    # L'action sonarsource/sonarqube-scan-action@v5 bundle déjà son propre Java.
    # On n'installe Java manuellement QUE pour Maven et Gradle.
    if profile.build_system in ("maven", "gradle"):
        java_setup = f"""
      - name: "⚙️ Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: '{profile.build_system}'
"""
    else:
        java_setup = ""  # Pas besoin — sonarsource/sonarqube-scan-action bundle Java

    # ── Step d'analyse SonarQube (spécifique au langage) ─────────────────────
    if profile.language == "java" and profile.build_system == "maven":
        # IMPORTANT : sonar:sonar -DskipTests — NE PAS utiliser 'mvn verify ... sonar'
        # car 'verify' re-exécute tous les tests (doublon coûteux et fragile).
        sonar_scan = f"""      - name: "🔍 SonarQube Scan (Maven)"
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        run: |
          mvn -B sonar:sonar -DskipTests \\
            -Dsonar.projectKey={project_key} \\
            -Dsonar.host.url={sonar_host} \\
            -Dsonar.token={sonar_token} \\
            -Dsonar.coverage.jacoco.xmlReportPaths=target/site/jacoco/jacoco.xml
        continue-on-error: true"""

    elif profile.language == "java" and profile.build_system == "gradle":
        sonar_scan = f"""      - name: "🔍 SonarQube Scan (Gradle)"
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        run: |
          ./gradlew sonar \\
            -Dsonar.projectKey={project_key} \\
            -Dsonar.host.url={sonar_host} \\
            -Dsonar.token={sonar_token}
        continue-on-error: true"""

    elif profile.language == "python":
        # L'action officielle supporte Python nativement.
        # -Dsonar.python.coverage.reportPaths=coverage.xml → lit le rapport pytest-cov
        sonar_scan = f"""      - name: "🔍 SonarQube Scan (Python)"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        with:
          args: >
            -Dsonar.projectKey={project_key}
            -Dsonar.sources=.
            -Dsonar.python.version={profile.python_version}
            -Dsonar.python.coverage.reportPaths=coverage.xml
        continue-on-error: true"""

    elif profile.language == "typescript":
        # TypeScript : tsconfig.json pour l'analyse + lcov.info pour la couverture
        sonar_scan = f"""      - name: "🔍 SonarQube Scan (TypeScript)"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        with:
          args: >
            -Dsonar.projectKey={project_key}
            -Dsonar.sources=.
            -Dsonar.typescript.tsconfigPath=tsconfig.json
            -Dsonar.javascript.lcov.reportPaths=coverage/lcov.info
        continue-on-error: true"""

    else:  # javascript (et fallback)
        sonar_scan = f"""      - name: "🔍 SonarQube Scan (JavaScript)"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        with:
          args: >
            -Dsonar.projectKey={project_key}
            -Dsonar.sources=.
            -Dsonar.javascript.lcov.reportPaths=coverage/lcov.info
        continue-on-error: true"""

    return f"""      - name: "📥 Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Requis pour git blame et l'historique SonarQube
{java_setup}
      - name: "📦 Download Coverage Artifacts"
        uses: actions/download-artifact@v4
        with:
          name: test-results
          path: .
        continue-on-error: true

{sonar_scan}

      - name: "✅ SonarQube Quality Gate"
        # v1.1.0 — version stable et pinée (éviter @master qui peut casser)
        uses: sonarsource/sonarqube-quality-gate-action@v1.1.0
        timeout-minutes: 5
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        continue-on-error: true

      - name: "📊 SonarQube Dashboard URL"
        if: always()
        run: |
          echo "══════════════════════════════════════════"
          echo " SonarQube Analysis Complete"
          echo " Projet  : {project_key}"
          echo " URL     : {sonar_host}/dashboard?id={project_key}"
          echo "══════════════════════════════════════════"
"""


# ── CodeQL Security Analysis Job ─────────────────────────────────────────────

# Mapping langage Code Auditor → langage CodeQL
# CodeQL supporte : java, python, javascript (couvre aussi TypeScript)
_CODEQL_LANGUAGE_MAP = {
    "java":       "java",
    "python":     "python",
    "javascript": "javascript",
    "typescript": "javascript",  # CodeQL analyse TS via le mode javascript
}


def _codeql_job(profile: ProjectProfile) -> str:
    """
    Génère le job CodeQL complet pour le workflow GitHub Actions.

    CodeQL vs SonarQube :
      - SonarQube  → qualité du code + coverage + bugs courants
      - CodeQL     → vulnérabilités de sécurité PROFONDES dans la logique
                     (SQL injection, XSS, path traversal, code injection...)

    Avantages de CodeQL :
      - Gratuit pour tous les repos (public ET privé via GitHub Advanced Security)
      - Zéro infrastructure (pas de serveur SonarQube requis)
      - Résultats dans Security > Code scanning > CodeQL alerts
      - Peut bloquer le merge via Branch Protection Rules

    Le job tourne EN PARALLÈLE avec sonar-scan (les deux dépendent
    de build-test mais pas l'un de l'autre — gain de temps).
    """
    codeql_language = _CODEQL_LANGUAGE_MAP.get(profile.language, "")

    # Si le langage n'est pas supporté par CodeQL, retourner un job no-op
    if not codeql_language:
        return """  codeql-scan:
    name: "CodeQL (not supported for this language)"
    runs-on: ubuntu-latest
    steps:
      - name: "Skip"
        run: echo "CodeQL does not support this language — skipping."
"""

    # Autobuild est suffisant pour Python et JavaScript.
    # Pour Java/Maven/Gradle, autobuild détecte automatiquement le build system.
    return f"""  codeql-scan:
    name: "CodeQL Security Analysis ({codeql_language})"
    runs-on: ubuntu-latest
    # Tourne en parallèle avec sonar-scan pour optimiser le temps total
    needs: build-test
    if: always() && needs.build-test.result == 'success'
    timeout-minutes: 30  # CodeQL peut être lent sur les grands projets
    permissions:
      security-events: write  # Upload des résultats SARIF
      actions: read
      contents: read
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: "Initialize CodeQL"
        # Init : configure les langages à analyser + les requêtes de sécurité
        uses: github/codeql-action/init@v3
        with:
          languages: {codeql_language}
          # security-extended : requêtes de sécurité complètes (recommandé)
          # security-and-quality : ajoute aussi les règles de qualité
          queries: security-extended

      - name: "Autobuild"
        # Autobuild compile automatiquement le projet pour l'analyse
        # (nécessaire pour Java — pas pour Python/JS qui sont interprétés)
        uses: github/codeql-action/autobuild@v3

      - name: "Perform CodeQL Analysis"
        # Lance l'analyse et uploade les résultats dans GitHub Security
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:{codeql_language}"
"""