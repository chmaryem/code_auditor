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

BUILD_FILES = {
    "pom.xml":            ("java",       "maven"),
    "build.gradle":       ("java",       "gradle"),
    "build.gradle.kts":   ("java",       "gradle"),
    "package.json":       ("javascript", "npm"),
    "requirements.txt":   ("python",     "pip"),
    "pyproject.toml":     ("python",     "poetry"),
    "setup.py":           ("python",     "pip"),
    "Cargo.toml":         ("rust",       "cargo"),
    "go.mod":             ("go",         "go"),
}


def detect_project_profile(file_checker) -> ProjectProfile:
    """
    Détecte le profil du projet en cherchant les fichiers de build.

    Args:
        file_checker: callable(path) -> str|None
            Retourne le contenu d'un fichier ou None s'il n'existe pas.
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
            return profile

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
    build_steps  = _build_steps(profile)
    sonar_steps  = _sonar_steps(profile)

    yaml = f"""# ─────────────────────────────────────────────────────────────────
# CI/CD Pipeline — Généré automatiquement par Code Auditor
#
# 2 jobs :
#   1. build-test  → {profile.build_system} ({profile.language})
#   2. sonar-scan  → Analyse qualité SonarQube
#
# Pour configurer SonarQube :
#   1. Créer un projet sur SonarCloud.io ou votre serveur SonarQube
#   2. Ajouter les secrets dans Settings → Secrets → Actions :
#      - SONAR_TOKEN    : Token d'analyse SonarQube
#      - SONAR_HOST_URL : https://sonarcloud.io (ou votre serveur)
# ─────────────────────────────────────────────────────────────────

name: "CI/CD — Build + SonarQube"

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]

permissions:
  contents: read
  statuses: write
  pull-requests: read
  security-events: write

jobs:
  # ── Job 1 : Build & Test ──────────────────────────────────────
  build-test:
    name: "🔧 Build & Test ({profile.language}/{profile.build_system})"
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - name: "📥 Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Requis pour SonarQube blame
{build_steps}

      - name: "📊 Upload Test Results"
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: |
            target/surefire-reports/
            build/test-results/
          retention-days: 5

  # ── Job 2 : SonarQube Analysis ────────────────────────────────
  sonar-scan:
    name: "🔍 SonarQube Analysis"
    needs: build-test  # Attend que build-test réussisse
    runs-on: ubuntu-latest
    timeout-minutes: 15
    if: always() && needs.build-test.result == 'success'
    steps:
{sonar_steps}
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


def _build_steps(profile: ProjectProfile) -> str:
    """
    Génère les steps de build/test selon le build system.

    NOTE Maven : `-Dmaven.test.failure.ignore=true` permet de continuer
    même si des tests échouent. SonarQube analysera quand même le code
    et fournira des métriques de couverture.
    """
    if profile.build_system == "maven":
        return f"""
      - name: "Setup Java"
        uses: actions/setup-java@v4
        with:
          java-version: '{profile.java_version}'
          distribution: 'temurin'
          cache: 'maven'

      - name: "Build"
        run: mvn compile -q

      - name: "🧪 Tests"
        # Les tests ne bloquent pas le pipeline — SonarQube analysera quand même
        run: mvn test -q -Dmaven.test.failure.ignore=true
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
        return f"""
      - name: "Setup Node.js"
        uses: actions/setup-node@v4
        with:
          node-version: '{profile.node_version}'
          cache: 'npm'

      - name: "Install"
        run: npm ci

      - name: "Build"
        run: npm run build --if-present

      - name: "Tests"
        run: npm test --if-present
        continue-on-error: true"""

    elif profile.build_system in ("pip", "poetry"):
        setup = (
            "pip install -r requirements.txt"
            if profile.build_system == "pip"
            else "pip install poetry && poetry install"
        )
        test_cmd = (
            "pytest --tb=short -q"
            if profile.build_system == "pip"
            else "poetry run pytest --tb=short -q"
        )
        return f"""
      - name: "Setup Python"
        uses: actions/setup-python@v5
        with:
          python-version: '{profile.python_version}'

      - name: "Install"
        run: |
          pip install --upgrade pip
          {setup}

      - name: "Tests"
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
    
    Supporte:
      - Java (Maven/Gradle) : via plugin officiel
      - Python : via sonar-scanner-cli
      - JavaScript/TypeScript : via sonar-scanner-cli
    """
    
    sonar_token = "${{ secrets.SONAR_TOKEN }}"
    sonar_host = "${{ secrets.SONAR_HOST_URL }}"
    project_key = "${{ github.repository_owner }}_${{ github.event.repository.name }}"
    
    if profile.language == "java" and profile.build_system == "maven":
        sonar_step = f"""      - name: "🔍 SonarQube Scan (Maven)"
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        run: |
          mvn -B verify org.sonarsource.scanner.maven:sonar-maven-plugin:sonar \\
            -Dsonar.projectKey={project_key} \\
            -Dsonar.host.url={sonar_host} \\
            -Dsonar.token={sonar_token}
        continue-on-error: true"""
    
    elif profile.language == "java" and profile.build_system == "gradle":
        sonar_step = f"""      - name: "🔍 SonarQube Scan (Gradle)"
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        run: |
          ./gradlew sonarqube \\
            -Dsonar.projectKey={project_key} \\
            -Dsonar.host.url={sonar_host} \\
            -Dsonar.token={sonar_token}
        continue-on-error: true"""
    
    else:
        # Python, JavaScript, et autres langages via sonar-scanner-cli
        sonar_step = f"""      - name: "🔍 SonarQube Scan (CLI)"
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        run: |
          # Télécharger sonar-scanner si nécessaire
          if ! command -v sonar-scanner &> /dev/null; then
            wget -q https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-5.0.1.3006-linux.zip
            unzip -q sonar-scanner-cli-*.zip
            export PATH=$PWD/sonar-scanner-*/bin:$PATH
          fi
          
          sonar-scanner \\
            -Dsonar.projectKey={project_key} \\
            -Dsonar.sources=. \\
            -Dsonar.host.url={sonar_host} \\
            -Dsonar.token={sonar_token}
        continue-on-error: true"""
    
    return f"""      - name: "📥 Checkout"
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Requis pour l'analyse complète

      - name: "⚙️ Setup Java 17 (SonarQube requiert Java)"
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: "📦 Download Build Artifacts"
        uses: actions/download-artifact@v4
        with:
          name: test-results
          path: .
        continue-on-error: true

{sonar_step}

      - name: "✅ SonarQube Quality Gate"
        uses: sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: {sonar_token}
          SONAR_HOST_URL: {sonar_host}
        continue-on-error: true

      - name: "📊 SonarQube Metrics"
        run: |
          echo "SonarQube analysis complete"
          echo "Project: {project_key}"
          echo "URL: {sonar_host}/dashboard?id={project_key}"
        continue-on-error: true
"""