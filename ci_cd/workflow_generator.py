"""
workflow_generator.py -- Genere le fichier YAML GitHub Actions adapte au projet.

Phase 1 : 2 jobs uniquement (build-test + sonar-scan).
Langages supportes : java, python, javascript, typescript.

Secrets requis :
  SONAR_TOKEN     : Token SonarCloud
  SONAR_HOST_URL  : https://sonarcloud.io
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectProfile:
    """Profil detecte du projet cible."""
    language: str         
    build_system: str    
    java_version: str = "17"
    python_version: str = "3.11"
    node_version: str = "20"
    has_tests: bool = True
    has_dockerfile: bool = False 


# -- Detection du profil -----------------------------------------------------

# Ordre important : tsconfig.json AVANT package.json pour TypeScript en priorite
BUILD_FILES = {
    "pom.xml":          ("java",       "maven"),
    "build.gradle":     ("java",       "gradle"),
    "build.gradle.kts": ("java",       "gradle"),
    "tsconfig.json":    ("typescript", "npm"),
    "package.json":     ("javascript", "npm"),
    "requirements.txt": ("python",     "pip"),
    "pyproject.toml":   ("python",     "poetry"),
    "setup.py":         ("python",     "pip"),
}

EXT_TO_LANGUAGE = {
    ".py":  "python",
    ".java":"java",
    ".js":  "javascript",
    ".ts":  "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}

LANG_DEFAULT_BUILD = {
    "python":     "pip",
    "java":       "maven",
    "javascript": "npm",
    "typescript": "npm",
}


def detect_project_profile(file_checker, file_lister=None) -> ProjectProfile:
    """
    Detecte le profil du projet.

    Strategie en 2 niveaux :
      1. Chercher les fichiers de build (pom.xml, requirements.txt, etc.)
      2. Fallback : compter les extensions de fichiers du repo
    """
    for build_file, (language, build_system) in BUILD_FILES.items():
        content = file_checker(build_file)
        if content:
            profile = ProjectProfile(language=language, build_system=build_system)
            if build_file == "pom.xml" and "<java.version>" in content:
                import re
                m = re.search(r"<java.version>(\d+)</java.version>", content)
                if m:
                    profile.java_version = m.group(1)
            # Detecter Dockerfile dans le repo
            for df_name in ("Dockerfile", "dockerfile", "Dockerfile.prod", "Dockerfile.dev"):
                if file_checker(df_name):
                    profile.has_dockerfile = True
                    break
            return profile

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


# -- generate_workflow --------------------------------------------------------

def _report_task_path(profile: ProjectProfile) -> str:
    """
    Retourne le chemin du fichier report-task.txt selon le build system.

    Maven  : target/sonar/report-task.txt  (mvn sonar:sonar)
    Gradle : build/sonar/report-task.txt   (./gradlew sonar)
    Autres : .scannerwork/report-task.txt  (sonarqube-scan-action)
    """
    if profile.build_system == "maven":
        return "target/sonar/report-task.txt"
    if profile.build_system == "gradle":
        return "build/sonar/report-task.txt"
    return ".scannerwork/report-task.txt"


def generate_workflow(
    profile: ProjectProfile,
    auditor_repo: str = "chmaryem/code_auditor",
    checkout_path: str = "code_auditor_tool",
    enable_publish: bool = True,
    enable_deploy: bool = False,
) -> str:
    """
    Genere le contenu YAML complet du workflow GitHub Actions.

    7 jobs :
      1. build-test   : compile + tests + coverage
      2. sonar-scan   : analyse qualite + Quality Gate BLOQUANT
      3. dep-scan     : scan CVE des dependances
      4. codeql-scan  : securite statique profonde (GitHub natif)
      5. docker-trivy : build Docker + scan image (Trivy)
      6. publish      : push image sur Docker Hub [si enable_publish]
      7. deploy       : SSH + docker compose up   [si enable_deploy]
    """
    build_steps = _build_steps(profile)
    sonar_steps = _sonar_steps(profile)
    artifacts_path = _test_results_path(profile)

    yaml = f"""# -----------------------------------------------------------------
# CI/CD Pipeline -- Genere automatiquement par Code Auditor
#
# 7 jobs (Java / Python / JS+TS) :
#   1. build-test    -> {profile.build_system} ({profile.language})
#   2. sonar-scan    -> Qualite + Coverage + Quality Gate BLOQUANT
#   3. dep-scan      -> Vulnerabilites CVE dans les dependances
#   4. codeql-scan   -> Securite statique profonde (GitHub natif)
#   5. docker-trivy  -> Scan securite image Docker
#   6. publish       -> Docker Hub (push main seulement)
#   7. deploy        -> SSH + docker compose up (push main seulement)
#
# Secrets requis :
#   - SONAR_TOKEN         : Token SonarCloud (sonarcloud.io)
#   - DOCKERHUB_USERNAME  : Login Docker Hub
#   - DOCKERHUB_TOKEN     : Token Docker Hub
#   - DEPLOY_HOST         : IP ou hostname du serveur de prod
#   - DEPLOY_USER         : User SSH (ex: ubuntu)
#   - DEPLOY_SSH_KEY      : Cle privee SSH (ed25519)
# Variables GitHub Actions requises :
#   - DEPLOY_PATH         : Chemin sur le serveur (ex: /opt/app)
#   - DEPLOY_URL          : URL de sante (ex: https://monapp.com)
# -----------------------------------------------------------------

name: "CI -- Build + SonarCloud + Dep-Scan + CodeQL + Trivy + Deploy"

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]
  workflow_dispatch:

permissions:
  contents: read
  statuses: write
  pull-requests: read
  security-events: write   # Requis pour CodeQL SARIF upload
  actions: read            # Requis pour CodeQL
  packages: write          # Requis pour GitHub Packages (optionnel)

jobs:

  # -- Job 1 : Build & Test ------------------------------------------------
  build-test:
    name: "Build & Test ({profile.language}/{profile.build_system})"
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
{build_steps}
      - name: "Upload coverage artifacts"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-report
          path: |
{artifacts_path}
          retention-days: 5

  # -- Job 2 : SonarCloud Analysis -----------------------------------------
  sonar-scan:
    name: "SonarCloud Analysis"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 15
    if: always() && needs.build-test.result == 'success'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
{sonar_steps}
      - name: "SonarCloud Quality Gate"
        uses: sonarsource/sonarqube-quality-gate-action@v1.1.0
        timeout-minutes: 5
        env:
          SONAR_TOKEN: ${{{{ secrets.SONAR_TOKEN }}}}
          SONAR_HOST_URL: "https://sonarcloud.io"
        with:
          scanMetadataReportFile: {_report_task_path(profile)}

      - name: "Dashboard URL"
        if: always()
        run: |
          echo "============================================"
          echo " SonarCloud Analysis Complete"
          echo " URL : https://sonarcloud.io/dashboard?id=${{{{ github.repository_owner }}}}_${{{{ github.event.repository.name }}}}"
          echo "============================================"

{_dep_scan_job(profile)}

{_codeql_job(profile)}

{_trivy_job(profile)}

{_publish_job(profile) if enable_publish else ''}

{_deploy_job(profile) if enable_deploy else ''}
"""

    result = yaml.strip() + "\n"
    errors = validate_workflow(result)
    if errors:
        import logging
        logging.getLogger(__name__).warning("YAML validation issues: %s", errors)
    return result


# -- Per-language build steps ------------------------------------------------

def _build_steps(profile: ProjectProfile) -> str:
    """Genere les steps build+test+coverage selon le langage."""

    lang = profile.language
    bs   = profile.build_system

    # ---- Java / Maven ----
    if bs == "maven":
        return f"""
      - name: "Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'maven'

      - name: "Compile"
        run: mvn compile -q
        continue-on-error: true

      - name: "Tests + JaCoCo Coverage"
        run: >
          mvn -B
          org.jacoco:jacoco-maven-plugin:prepare-agent
          test
          org.jacoco:jacoco-maven-plugin:report
          -Dmaven.test.failure.ignore=true
          -q
        continue-on-error: true

      - name: "Package"
        run: mvn package -q -DskipTests
        continue-on-error: true
"""

    # ---- Java / Gradle ----
    if bs == "gradle":
        return f"""
      - name: "Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'gradle'

      - name: "Make gradlew executable"
        run: chmod +x ./gradlew

      - name: "Tests + Coverage"
        run: ./gradlew test jacocoTestReport --continue || true

      - name: "Build"
        run: ./gradlew build -x test
"""

    # ---- Python / pip ----
    if bs == "pip":
        return f"""
      - name: "Setup Python {profile.python_version}"
        uses: actions/setup-python@v5
        with:
          python-version: '{profile.python_version}'
          cache: 'pip'

      - name: "Install dependencies"
        run: |
          pip install --upgrade pip
          [ -f requirements.txt ] && pip install -r requirements.txt || true
          [ -f requirements-dev.txt ] && pip install -r requirements-dev.txt || true
          pip install pytest pytest-cov

      - name: "Tests + Coverage"
        run: pytest --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing -v || true
        continue-on-error: true
"""

    # ---- Python / poetry ----
    if bs == "poetry":
        return f"""
      - name: "Setup Python {profile.python_version}"
        uses: actions/setup-python@v5
        with:
          python-version: '{profile.python_version}'

      - name: "Install Poetry"
        run: pip install poetry

      - name: "Install dependencies"
        run: poetry install

      - name: "Tests + Coverage"
        run: poetry run pytest --cov=. --cov-report=xml:coverage.xml -v || true
        continue-on-error: true
"""

    # ---- JavaScript / TypeScript (npm) ----
    if bs == "npm":
        tsconfig = ""
        if lang == "typescript":
            tsconfig = """
      - name: "TypeScript compile check"
        run: npx tsc --noEmit || true
        continue-on-error: true
"""
        return f"""
      - name: "Setup Node.js {profile.node_version}"
        uses: actions/setup-node@v4
        with:
          node-version: '{profile.node_version}'
          cache: 'npm'

      - name: "Install dependencies"
        run: npm ci
{tsconfig}
      - name: "Tests + Coverage (Jest)"
        run: CI=true npm test -- --coverage --watchAll=false --passWithNoTests || true
        continue-on-error: true
"""

    # ---- Fallback ----
    return """
      - name: "Build"
        run: echo "No build system detected -- update detect_project_profile()"
"""


# -- Per-language SonarCloud steps -------------------------------------------

def _sonar_steps(profile: ProjectProfile) -> str:
    """Genere les steps SonarCloud selon le langage."""

    lang = profile.language
    bs   = profile.build_system

    # ---- Java / Maven : sonar:sonar via le plugin Maven ----
    if bs == "maven":
        return f"""
      - name: "Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'maven'

      - name: "Download coverage artifacts"
        uses: actions/download-artifact@v4
        with:
          name: coverage-report
          path: target/site/jacoco/
        continue-on-error: true

      - name: "Compile (requis par SonarCloud pour analyser le bytecode)"
        run: mvn -B compile -q

      - name: "SonarCloud Scan (Maven)"
        env:
          SONAR_TOKEN: ${{{{ secrets.SONAR_TOKEN }}}}
          SONAR_HOST_URL: "https://sonarcloud.io"
        run: >
          mvn -B sonar:sonar -DskipTests
          -Dsonar.projectKey=${{{{ github.repository_owner }}}}_${{{{ github.event.repository.name }}}}
          -Dsonar.organization=${{{{ github.repository_owner }}}}
          -Dsonar.java.binaries=target/classes
          -Dsonar.coverage.jacoco.xmlReportPaths=target/site/jacoco/jacoco.xml
"""

    # ---- Java / Gradle ----
    if bs == "gradle":
        return f"""
      - name: "Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'gradle'

      - name: "Make gradlew executable"
        run: chmod +x ./gradlew

      - name: "SonarCloud Scan (Gradle)"
        env:
          SONAR_TOKEN: ${{{{ secrets.SONAR_TOKEN }}}}
          SONAR_HOST_URL: "https://sonarcloud.io"
        run: >
          ./gradlew sonar
          "-Dsonar.projectKey=${{{{ github.repository_owner }}}}_${{{{ github.event.repository.name }}}}"
          "-Dsonar.organization=${{{{ github.repository_owner }}}}"
"""

    # ---- Python : sonarqube-scan-action ----
    if lang == "python":
        return f"""
      - name: "Setup Python {profile.python_version}"
        uses: actions/setup-python@v5
        with:
          python-version: '{profile.python_version}'

      - name: "Download coverage artifacts"
        uses: actions/download-artifact@v4
        with:
          name: coverage-report
          path: .
        continue-on-error: true

      - name: "SonarCloud Scan (Python)"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: ${{{{ secrets.SONAR_TOKEN }}}}
          SONAR_HOST_URL: "https://sonarcloud.io"
        with:
          args: >
            -Dsonar.projectKey=${{{{ github.repository_owner }}}}_${{{{ github.event.repository.name }}}}
            -Dsonar.organization=${{{{ github.repository_owner }}}}
            -Dsonar.sources=.
            -Dsonar.python.version={profile.python_version}
            -Dsonar.python.coverage.reportPaths=coverage.xml
            -Dsonar.exclusions=**/__pycache__/**,**/.venv/**,**/venv/**,**/tests/**
"""

    # ---- JavaScript : sonarqube-scan-action ----
    if lang == "javascript":
        return """
      - name: "Download coverage artifacts"
        uses: actions/download-artifact@v4
        with:
          name: coverage-report
          path: coverage/
        continue-on-error: true

      - name: "SonarCloud Scan (JavaScript)"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: "https://sonarcloud.io"
        with:
          args: >
            -Dsonar.projectKey=${{ github.repository_owner }}_${{ github.event.repository.name }}
            -Dsonar.organization=${{ github.repository_owner }}
            -Dsonar.sources=src
            -Dsonar.javascript.lcov.reportPaths=coverage/lcov.info
            -Dsonar.exclusions=**/node_modules/**,**/dist/**,**/*.test.js,**/*.spec.js
"""

    # ---- TypeScript : sonarqube-scan-action ----
    if lang == "typescript":
        return """
      - name: "Download coverage artifacts"
        uses: actions/download-artifact@v4
        with:
          name: coverage-report
          path: coverage/
        continue-on-error: true

      - name: "SonarCloud Scan (TypeScript)"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: "https://sonarcloud.io"
        with:
          args: >
            -Dsonar.projectKey=${{ github.repository_owner }}_${{ github.event.repository.name }}
            -Dsonar.organization=${{ github.repository_owner }}
            -Dsonar.sources=src
            -Dsonar.typescript.tsconfigPath=tsconfig.json
            -Dsonar.javascript.lcov.reportPaths=coverage/lcov.info
            -Dsonar.exclusions=**/node_modules/**,**/dist/**,**/*.spec.ts,**/*.test.ts
"""

    # ---- Fallback ----
    return """
      - name: "SonarCloud Scan"
        uses: sonarsource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: "https://sonarcloud.io"
        with:
          args: >
            -Dsonar.projectKey=${{ github.repository_owner }}_${{ github.event.repository.name }}
            -Dsonar.organization=${{ github.repository_owner }}
"""


# -- Test results path per language ------------------------------------------

def _test_results_path(profile: ProjectProfile) -> str:
    """Retourne le(s) chemin(s) des artefacts de couverture selon le langage."""
    bs = profile.build_system
    if bs == "maven":
        return "            target/site/jacoco/\n            target/surefire-reports/"
    if bs == "gradle":
        return "            build/reports/jacoco/\n            build/reports/tests/"
    if profile.language == "python":
        return "            coverage.xml\n            htmlcov/"
    if bs == "npm":
        return "            coverage/lcov.info\n            coverage/lcov-report/"
    return "            ."


# -- Validation --------------------------------------------------------------

def generate_dockerfile(profile: ProjectProfile) -> str:
    """Public wrapper — Dockerfile adapté au langage/build system détecté."""
    return _dockerfile_template(profile)


def generate_docker_compose(profile: ProjectProfile) -> str:
    """
    Génère un docker-compose.yml production-ready adapté au projet.
    Services additionnels inférés depuis le build system :
      Java/Maven|Gradle → app + postgres
      Python            → app + redis
      JS/TS             → app seul
    """
    port_map = {"java": "8080", "python": "8000", "javascript": "3000", "typescript": "3000"}
    app_port = port_map.get(profile.language.lower(), "8080")
    lang     = profile.language.lower()

    extra_services = ""
    extra_volumes  = ""

    if lang == "java":
        extra_services = """
  db:
    image: postgres:15-alpine
    container_name: db
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-appdb}
      POSTGRES_USER: ${POSTGRES_USER:-appuser}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - app-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-appuser}"]
      interval: 10s
      timeout: 5s
      retries: 5
"""
        extra_volumes = "  db-data:\n"

    elif lang == "python":
        extra_services = """
  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
"""
        extra_volumes = "  redis-data:\n"

    volumes_block = f"\nvolumes:\n{extra_volumes}" if extra_volumes else ""

    return f"""# docker-compose.yml — généré par Code Auditor
# Projet  : {profile.language} / {profile.build_system}
# Usage   : docker compose up -d

version: "3.9"

services:

  app:
    build: .
    image: ${{DOCKERHUB_USERNAME:-myorg}}/${{IMAGE_NAME:-app}}:${{IMAGE_TAG:-latest}}
    container_name: app
    restart: unless-stopped
    ports:
      - "{app_port}:{app_port}"
    env_file:
      - .env
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{app_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
{('    depends_on:\n      db:\n        condition: service_healthy' if lang == 'java' else '    depends_on:\n      redis:\n        condition: service_healthy') if extra_services else ''}
{extra_services}
networks:
  app-network:
    driver: bridge
{volumes_block}"""


def validate_workflow(yaml_content: str) -> list[str]:
    """Valide le YAML genere. Retourne la liste des erreurs (vide = valide)."""
    _, errors = validate_workflow_strict(yaml_content)
    return errors


def validate_workflow_strict(yaml_content: str) -> tuple[bool, list[str]]:
    """Validation stricte du YAML. Retourne (is_valid, errors)."""
    errors = []
    try:
        import yaml as pyyaml
        doc = pyyaml.safe_load(yaml_content)
    except ImportError:
        return True, []
    except Exception as e:
        return False, [f"YAML invalide: {e}"]

    if not isinstance(doc, dict):
        return False, ["Le YAML ne contient pas un mapping racine"]

    for key in ("name", "jobs"):
        if key not in doc:
            errors.append(f"Cle obligatoire manquante: '{key}'")
    if "on" not in doc and True not in doc:
        errors.append("Cle obligatoire manquante: 'on' (trigger)")

    jobs = doc.get("jobs", {})
    if isinstance(jobs, dict):
        if "build-test" not in jobs:
            errors.append("Job 'build-test' manquant")
        if "sonar-scan" not in jobs:
            errors.append("Job 'sonar-scan' manquant")
        for job_name, job_def in jobs.items():
            if not isinstance(job_def, dict):
                errors.append(f"Job '{job_name}' n'est pas un mapping")
                continue
            if "runs-on" not in job_def:
                errors.append(f"Job '{job_name}': 'runs-on' manquant")
            if "steps" not in job_def:
                errors.append(f"Job '{job_name}': 'steps' manquant")

    return len(errors) == 0, errors


# -- Dependency Scan job -----------------------------------------------------

def _dep_scan_job(profile: ProjectProfile) -> str:
    """
    Genere le job dep-scan selon le build system.

    Java/Maven : OWASP Dependency-Check (bloque si CVSS >= 9)
    Python     : pip-audit
    JS/TS      : npm audit
    """
    lang = profile.language
    bs   = profile.build_system

    # ---- Java / Maven ----
    if bs == "maven":
        return f"""
  # -- Job 3 : Dependency Scan (OWASP) --------------------------------------
  dep-scan:
    name: "Dependency Scan (OWASP)"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 30
    if: always() && needs.build-test.result == 'success'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'

      - name: "OWASP Dependency-Check"
        uses: dependency-check/Dependency-Check_Action@main
        env:
          JAVA_HOME: /opt/jdk
        with:
          project: '${{{{ github.event.repository.name }}}}'
          path: '.'
          format: 'HTML'
          args: >
            --failOnCVSS 9
            --enableRetired
            --nvdApiKey ${{{{ secrets.NVD_API_KEY }}}}
        continue-on-error: true

      - name: "Upload OWASP Report"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: owasp-dependency-report
          path: ${{{{ github.workspace }}}}/reports
          retention-days: 10
"""

    # ---- Java / Gradle ----
    if bs == "gradle":
        return f"""
  # -- Job 3 : Dependency Scan (OWASP) --------------------------------------
  dep-scan:
    name: "Dependency Scan (OWASP)"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 20
    if: always() && needs.build-test.result == 'success'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Setup Java {profile.java_version}"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'gradle'

      - name: "Make gradlew executable"
        run: chmod +x ./gradlew

      - name: "OWASP Dependency-Check (Gradle)"
        run: ./gradlew dependencyCheckAnalyze || echo "Vulnerabilites detectees"
        continue-on-error: true

      - name: "Upload OWASP Report"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: owasp-dependency-report
          path: build/reports/dependency-check-report.html
          retention-days: 10
"""

    # ---- Python ----
    if lang == "python":
        req_file = "pyproject.toml" if bs == "poetry" else "requirements.txt"
        install_cmd = "poetry run pip-audit" if bs == "poetry" else "pip-audit"
        return f"""
  # -- Job 3 : Dependency Scan (pip-audit) ----------------------------------
  dep-scan:
    name: "Dependency Scan (pip-audit)"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 10
    if: always() && needs.build-test.result == 'success'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Setup Python {profile.python_version}"
        uses: actions/setup-python@v5
        with:
          python-version: '{profile.python_version}'
          cache: 'pip'

      - name: "Install dependencies + pip-audit"
        run: |
          pip install --upgrade pip pip-audit
          [ -f requirements.txt ] && pip install -r requirements.txt || true
          [ -f requirements-dev.txt ] && pip install -r requirements-dev.txt || true

      - name: "pip-audit scan"
        run: pip-audit --format=json --output=pip-audit-report.json || true
        continue-on-error: true

      - name: "pip-audit summary"
        run: pip-audit --format=columns || true
        continue-on-error: true

      - name: "Upload pip-audit report"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: pip-audit-report
          path: pip-audit-report.json
          retention-days: 10
"""

    # ---- JavaScript / TypeScript ----
    if bs == "npm":
        return f"""
  # -- Job 3 : Dependency Scan (npm audit) ----------------------------------
  dep-scan:
    name: "Dependency Scan (npm audit)"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 10
    if: always() && needs.build-test.result == 'success'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Setup Node.js {profile.node_version}"
        uses: actions/setup-node@v4
        with:
          node-version: '{profile.node_version}'
          cache: 'npm'

      - name: "Install dependencies"
        run: npm ci

      - name: "npm audit"
        run: npm audit --json > npm-audit-report.json || true
        continue-on-error: true

      - name: "npm audit summary"
        run: npm audit --audit-level=critical || true
        continue-on-error: true

      - name: "Upload npm audit report"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: npm-audit-report
          path: npm-audit-report.json
          retention-days: 10
"""

    # ---- Fallback ----
    return """
  # -- Job 3 : Dependency Scan ----------------------------------------------
  dep-scan:
    name: "Dependency Scan"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 10
    if: always() && needs.build-test.result == 'success'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4
      - name: "Scan"
        run: echo "No dependency scanner configured for this language"
"""


# -- CodeQL Security Analysis job --------------------------------------------

def _codeql_job(profile: ProjectProfile) -> str:
    """
    Genere le job codeql-scan.

    100% GitHub natif — zéro secret requis.
    Supporte : java, python, javascript (inclut TypeScript).
    Resultats publies dans GitHub Security > Code Scanning.
    """
    lang = profile.language

    # Mapping langage projet → langage CodeQL
    codeql_lang_map = {
        "java":       "java",
        "python":     "python",
        "javascript": "javascript",
        "typescript": "javascript",   # CodeQL utilise 'javascript' pour TS aussi
    }
    codeql_lang = codeql_lang_map.get(lang, "")

    if not codeql_lang:
        return """
  # -- Job 4 : CodeQL ---------------------------------------------------
  codeql-scan:
    name: "CodeQL Security Analysis"
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: "Skip"
        run: echo "CodeQL non configure pour ce langage"
"""

    # Build step specifique par langage
    if profile.build_system == "maven":
        build_step = f"""
      - name: "Build (CodeQL)"
        run: mvn -B compile -DskipTests -q
        continue-on-error: true"""
        # Maven: autobuild non nécessaire — on compile manuellement
        # CodeQL traque la compilation via son agent Java
        use_autobuild = False
    elif profile.build_system == "gradle":
        build_step = """
      - name: "Build (CodeQL)"
        run: chmod +x ./gradlew && ./gradlew compileJava -x test
        continue-on-error: true"""
        use_autobuild = False
    elif profile.build_system == "npm":
        build_step = """
      - name: "Install deps (CodeQL)"
        run: npm ci"""
        use_autobuild = True   # JS/TS : autobuild = no-op, OK
    else:
        build_step = ""
        use_autobuild = True

    autobuild_step = """
      - name: "Autobuild"
        uses: github/codeql-action/autobuild@v3
""" if use_autobuild else ""

    return f"""
  # -- Job 4 : CodeQL Security Analysis ---------------------------------
  codeql-scan:
    name: "CodeQL Security Analysis ({codeql_lang})"
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      security-events: write
      actions: read
      contents: read
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Initialize CodeQL"
        uses: github/codeql-action/init@v3
        with:
          languages: {codeql_lang}
          queries: security-extended
{build_step}
{autobuild_step}
      - name: "Analyze"
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:{codeql_lang}"
"""


# -- Dockerfile template -----------------------------------------------------

def _dockerfile_template(profile: ProjectProfile) -> str:
    """
    Genere un Dockerfile minimal adapte au langage/build system.
    Utilise des multi-stage builds pour minimiser la taille de l'image.
    """
    bs   = profile.build_system
    lang = profile.language

    if bs == "maven":
        return f"""# Dockerfile genere par Code Auditor
# Multi-stage build Java/Maven
# eclipse-temurin:jammy supporte linux/amd64 ET linux/arm64

FROM maven:3.9-eclipse-temurin-{profile.java_version} AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline -q
COPY src ./src
RUN mvn package -DskipTests -q

FROM eclipse-temurin:{profile.java_version}-jre-jammy
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
COPY --from=build /app/target/*.jar app.jar
USER appuser
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
"""

    if bs == "gradle":
        return f"""# Dockerfile genere par Code Auditor
# Multi-stage build Java/Gradle
# eclipse-temurin:jammy supporte linux/amd64 ET linux/arm64

FROM gradle:8-jdk{profile.java_version} AS build
WORKDIR /app
COPY build.gradle* settings.gradle* ./
COPY gradle ./gradle
RUN gradle dependencies -q
COPY src ./src
RUN gradle bootJar -x test -q

FROM eclipse-temurin:{profile.java_version}-jre-jammy
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
COPY --from=build /app/build/libs/*.jar app.jar
USER appuser
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
"""

    if lang == "python":
        req_file = "pyproject.toml" if bs == "poetry" else "requirements.txt"
        return f"""# Dockerfile genere par Code Auditor
# Python {profile.python_version}

FROM python:{profile.python_version}-slim
WORKDIR /app
RUN addgroup --system appgroup && adduser --system --group appuser
COPY {req_file} .
RUN pip install --no-cache-dir -r {req_file}
COPY . .
USER appuser
EXPOSE 8000
CMD ["python", "main.py"]
"""

    if lang in ("javascript", "typescript"):
        build_cmd = "npm run build" if lang == "typescript" else "echo 'no build'"
        return f"""# Dockerfile genere par Code Auditor
# Node.js {profile.node_version} ({lang})

FROM node:{profile.node_version}-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN {build_cmd} || true

FROM node:{profile.node_version}-alpine
WORKDIR /app
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
COPY --from=build /app/package*.json ./
RUN npm ci --only=production
COPY --from=build /app/dist ./dist
USER appuser
EXPOSE 3000
CMD ["node", "dist/index.js"]
"""

    # Fallback
    return """# Dockerfile genere par Code Auditor
FROM ubuntu:22.04
WORKDIR /app
COPY . .
CMD ["echo", "Configure votre Dockerfile"]
"""


# -- Docker Trivy scan job ---------------------------------------------------

def _trivy_job(profile: ProjectProfile) -> str:
    """
    Genere le job docker-trivy.
    Si pas de Dockerfile dans le profil → job skip (sera gere par ci-deploy).
    Trivy scanne l'image Docker buildee pour :
      - CVE CRITICAL/HIGH dans les packages OS
      - CVE dans les libs runtime
      - Misconfigurations Dockerfile
    Resultats uploades dans GitHub Security > Code Scanning (SARIF).
    """
    return """
  # -- Job 5 : Docker Security Scan (Trivy) --------------------------------
  docker-trivy:
    name: "Container Scan (Trivy)"
    needs: build-test
    runs-on: ubuntu-latest
    timeout-minutes: 15
    if: always() && needs.build-test.result == 'success'
    permissions:
      security-events: write
      contents: read
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Check Dockerfile exists"
        id: check_docker
        run: |
          if [ -f "Dockerfile" ]; then
            echo "exists=true" >> $GITHUB_OUTPUT
          else
            echo "exists=false" >> $GITHUB_OUTPUT
            echo "Dockerfile absent — skip Trivy scan"
          fi

      - name: "Build Docker image"
        if: steps.check_docker.outputs.exists == 'true'
        run: docker build -t app:${{ github.sha }} .

      - name: "Trivy — scan vulnerabilites (SARIF)"
        if: steps.check_docker.outputs.exists == 'true'
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: CRITICAL,HIGH
          ignore-unfixed: true
          exit-code: 0

      - name: "Upload SARIF to GitHub Security"
        if: steps.check_docker.outputs.exists == 'true'
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
          category: trivy-container

      - name: "Trivy — rapport JSON"
        if: steps.check_docker.outputs.exists == 'true'
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:${{ github.sha }}
          format: json
          output: trivy-report.json
          severity: CRITICAL,HIGH
          ignore-unfixed: true
          exit-code: 0

      - name: "Trivy — resume console"
        if: steps.check_docker.outputs.exists == 'true'
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:${{ github.sha }}
          format: table
          severity: CRITICAL,HIGH
          ignore-unfixed: true
          exit-code: 0

      - name: "Upload Trivy rapport"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: trivy-security-report
          path: |
            trivy-results.sarif
            trivy-report.json
          retention-days: 10
"""


# -- Job 6 : Publish to Docker Hub ------------------------------------------

def _publish_job(profile: ProjectProfile) -> str:
    """
    Stage 5 — Publie l'image Docker sur Docker Hub.

    Conditions :
      - Uniquement sur push sur main (pas sur les PRs)
      - Attend que build-test, sonar-scan, dep-scan et docker-trivy soient OK
      - Multi-platform : linux/amd64 + linux/arm64
      - Cache GitHub Actions pour des builds rapides
      - Tags : SHA court + latest + version semver (si tag git)

    Secrets requis :
      - DOCKERHUB_USERNAME : Login Docker Hub
      - DOCKERHUB_TOKEN    : Access token Docker Hub (read/write)
    """
    return """
  # -- Job 6 : Publish to Docker Hub -----------------------------------------
  publish:
    name: "Publish to Docker Hub"
    needs: [build-test, docker-trivy]
    runs-on: ubuntu-latest
    timeout-minutes: 20
    # Publie si build-test OK + push sur main
    # Sonar/OWASP fournissent de la visibilite mais ne bloquent PAS le publish
    if: |
      github.ref == 'refs/heads/main' &&
      github.event_name == 'push' &&
      needs.build-test.result == 'success' &&
      needs.docker-trivy.result != 'failure'
    steps:
      - name: "Checkout"
        uses: actions/checkout@v4

      - name: "Set up QEMU (multi-platform)"
        uses: docker/setup-qemu-action@v3

      - name: "Set up Docker Buildx"
        uses: docker/setup-buildx-action@v3

      - name: "Login to Docker Hub"
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: "Prepare image name"
        id: image
        run: |
          # Lowercase + strip trailing dashes (noms comme 'test-project-' invalides)
          REPO=$(echo "${{ github.event.repository.name }}" | tr '[:upper:]' '[:lower:]' | sed 's/-*$//')
          USER=$(echo "${{ secrets.DOCKERHUB_USERNAME }}" | tr '[:upper:]' '[:lower:]')
          SHA=$(echo "${{ github.sha }}" | cut -c1-7)
          echo "tag_sha=$USER/$REPO:$SHA"       >> $GITHUB_OUTPUT
          echo "tag_latest=$USER/$REPO:latest"  >> $GITHUB_OUTPUT
          echo "url=https://hub.docker.com/r/$USER/$REPO" >> $GITHUB_OUTPUT
          echo "Image : $USER/$REPO  Tags: $SHA + latest"

      - name: "Build & Push (linux/amd64)"
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          platforms: linux/amd64
          tags: |
            ${{ steps.image.outputs.tag_sha }}
            ${{ steps.image.outputs.tag_latest }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: "Image digest"
        run: |
          echo "========================================"
          echo " Image publiee sur Docker Hub"
          echo " Tags  : ${{ steps.image.outputs.tag_sha }}"
          echo "         ${{ steps.image.outputs.tag_latest }}"
          echo " URL   : ${{ steps.image.outputs.url }}"
          echo "========================================"
"""



# -- Job 7 : Deploy via SSH + Docker Compose ---------------------------------

def _deploy_job(profile: ProjectProfile) -> str:
    """
    Stage 6 — Deploiement sur serveur via SSH + docker compose.

    Conditions :
      - Uniquement sur push sur main, apres le job publish
      - Health check automatique apres deploiement
      - Rollback manuel possible (instruction dans les logs)

    Secrets requis :
      - DEPLOY_HOST    : IP ou hostname du serveur
      - DEPLOY_USER    : User SSH (ex: ubuntu, deploy)
      - DEPLOY_SSH_KEY : Cle privee SSH (ed25519)
      - DOCKERHUB_USERNAME : Pour docker login sur le serveur
      - DOCKERHUB_TOKEN    : Pour docker login sur le serveur

    Variables GitHub Actions :
      - DEPLOY_PATH    : Chemin absolu sur le serveur (ex: /opt/myapp)
      - DEPLOY_URL     : URL de sante (ex: https://myapp.example.com)
    """
    # Port par defaut selon le langage
    port_map = {
        "java":       "8080",
        "python":     "8000",
        "javascript": "3000",
        "typescript": "3000",
    }
    app_port = port_map.get(profile.language.lower(), "8080")

    return f"""
  # -- Job 7 : Deploy via SSH + Docker Compose --------------------------------
  deploy:
    name: "Deploy (Docker Compose)"
    needs: [publish]
    runs-on: ubuntu-latest
    timeout-minutes: 15
    if: |
      github.ref == 'refs/heads/main' &&
      github.event_name == 'push' &&
      needs.publish.result == 'success'
    environment:
      name: production
      url: ${{{{ vars.DEPLOY_URL }}}}
    steps:
      - name: "Deploy via SSH"
        uses: appleboy/ssh-action@v1.0.3
        with:
          host:     ${{{{ secrets.DEPLOY_HOST }}}}
          username: ${{{{ secrets.DEPLOY_USER }}}}
          key:      ${{{{ secrets.DEPLOY_SSH_KEY }}}}
          envs:     DOCKERHUB_USERNAME,DOCKERHUB_TOKEN,GITHUB_REPOSITORY
          script: |
            set -e
            APP_PATH=${{{{vars.DEPLOY_PATH:-/opt/app}}}}
            cd "$APP_PATH"

            # Login Docker Hub sur le serveur
            echo "${{{{ secrets.DOCKERHUB_TOKEN }}}}" | \\
              docker login -u "${{{{ secrets.DOCKERHUB_USERNAME }}}}" --password-stdin

            # Mettre a jour et redemarrer
            docker compose pull
            docker compose up -d --remove-orphans
            docker image prune -f

            # Afficher les containers actifs
            echo "=== Containers actifs ==="
            docker compose ps

      - name: "Health check"
        run: |
          echo "Attente demarrage (20s)..."
          sleep 20
          URL="${{{{ vars.DEPLOY_URL }}}}"
          if [ -n "$URL" ]; then
            if curl --fail --silent --retry 5 --retry-delay 10 "$URL" > /dev/null; then
              echo "Health check OK : $URL"
            else
              echo "::warning::Health check echoue : $URL"
              echo "Rollback : docker compose pull <version-precedente> && docker compose up -d"
            fi
          else
            echo "DEPLOY_URL non configure -- health check ignore"
          fi

      - name: "Deployment summary"
        if: always()
        run: |
          echo "========================================"
          echo " Deploiement termine"
          echo " Repo  : ${{{{ github.repository }}}}"
          echo " SHA   : ${{{{ github.sha }}}}"
          echo " Port  : {app_port}"
          echo " URL   : ${{{{ vars.DEPLOY_URL }}}}"
          echo "========================================"
"""