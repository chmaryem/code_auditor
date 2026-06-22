"""
tests/test_workflow_generator.py

Tests unitaires pour ci_cd/workflow_generator.py.
Aucun appel réseau — tout est testé en mémoire.

Lancer : pytest tests/test_workflow_generator.py -v
"""

import pytest
import yaml

from ci_cd.workflow_generator import (
    ProjectProfile,
    detect_project_profile,
    generate_workflow,
    generate_dockerfile,
    generate_docker_compose,
    validate_workflow_strict,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _file_checker_for(*present_files):
    """Retourne un file_checker qui simule la présence de fichiers donnés."""
    def checker(path: str) -> str | None:
        return "dummy_content" if path in present_files else None
    return checker


PROFILES = {
    "java_maven":   ProjectProfile(language="java",       build_system="maven"),
    "java_gradle":  ProjectProfile(language="java",       build_system="gradle"),
    "python":       ProjectProfile(language="python",     build_system="pip"),
    "javascript":   ProjectProfile(language="javascript", build_system="npm"),
    "typescript":   ProjectProfile(language="typescript", build_system="npm"),
}


# ── detect_project_profile ────────────────────────────────────────────────────

class TestDetectProjectProfile:

    def test_detects_java_maven_from_pom(self):
        checker = _file_checker_for("pom.xml")
        profile = detect_project_profile(checker)
        assert profile.language == "java"
        assert profile.build_system == "maven"

    def test_detects_java_gradle_from_build_gradle(self):
        checker = _file_checker_for("build.gradle")
        profile = detect_project_profile(checker)
        assert profile.language == "java"
        assert profile.build_system == "gradle"

    def test_detects_python_pip_from_requirements(self):
        checker = _file_checker_for("requirements.txt")
        profile = detect_project_profile(checker)
        assert profile.language == "python"
        assert profile.build_system == "pip"

    def test_detects_python_poetry_from_pyproject(self):
        checker = _file_checker_for("pyproject.toml")
        profile = detect_project_profile(checker)
        assert profile.language == "python"
        assert profile.build_system == "poetry"

    def test_detects_typescript_before_javascript(self):
        # tsconfig.json doit être prioritaire sur package.json
        checker = _file_checker_for("tsconfig.json", "package.json")
        profile = detect_project_profile(checker)
        assert profile.language == "typescript"

    def test_detects_javascript_from_package_json(self):
        checker = _file_checker_for("package.json")
        profile = detect_project_profile(checker)
        assert profile.language == "javascript"
        assert profile.build_system == "npm"

    def test_returns_unknown_when_no_build_file(self):
        checker = _file_checker_for()
        profile = detect_project_profile(checker)
        assert profile.language == "unknown"

    def test_detects_dockerfile_present(self):
        checker = _file_checker_for("pom.xml", "Dockerfile")
        profile = detect_project_profile(checker)
        assert profile.has_dockerfile is True

    def test_no_dockerfile_when_absent(self):
        checker = _file_checker_for("pom.xml")
        profile = detect_project_profile(checker)
        assert profile.has_dockerfile is False

    def test_extracts_java_version_from_pom(self):
        def checker(path):
            if path == "pom.xml":
                return "<properties><java.version>21</java.version></properties>"
            return None
        profile = detect_project_profile(checker)
        assert profile.java_version == "21"

    def test_fallback_to_file_extension_scan(self):
        checker = _file_checker_for()  # aucun build file
        files = ["src/main.py", "utils/helper.py", "app.py"]
        profile = detect_project_profile(checker, file_lister=lambda: files)
        assert profile.language == "python"


# ── generate_workflow ─────────────────────────────────────────────────────────

class TestGenerateWorkflow:

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_output_is_valid_yaml(self, profile_key):
        profile = PROFILES[profile_key]
        yaml_str = generate_workflow(profile)
        doc = yaml.safe_load(yaml_str)
        assert isinstance(doc, dict), f"YAML invalide pour {profile_key}"

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_has_required_top_level_keys(self, profile_key):
        profile = PROFILES[profile_key]
        doc = yaml.safe_load(generate_workflow(profile))
        assert "name" in doc
        assert "jobs" in doc
        assert "on" in doc or True in doc  # 'on' est parsé en True par PyYAML

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_has_build_test_job(self, profile_key):
        doc = yaml.safe_load(generate_workflow(PROFILES[profile_key]))
        assert "build-test" in doc["jobs"]

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_has_sonar_scan_job(self, profile_key):
        doc = yaml.safe_load(generate_workflow(PROFILES[profile_key]))
        assert "sonar-scan" in doc["jobs"]

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_every_job_has_runs_on(self, profile_key):
        doc = yaml.safe_load(generate_workflow(PROFILES[profile_key]))
        for job_name, job_def in doc["jobs"].items():
            assert "runs-on" in job_def, f"Job '{job_name}' manque 'runs-on'"

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_every_job_has_steps(self, profile_key):
        doc = yaml.safe_load(generate_workflow(PROFILES[profile_key]))
        for job_name, job_def in doc["jobs"].items():
            assert "steps" in job_def, f"Job '{job_name}' manque 'steps'"
            assert len(job_def["steps"]) > 0, f"Job '{job_name}' n'a aucune step"

    def test_validate_workflow_strict_passes_on_generated(self):
        for key, profile in PROFILES.items():
            yaml_str = generate_workflow(profile)
            valid, errors = validate_workflow_strict(yaml_str)
            assert valid, f"validate_workflow_strict a rejeté le workflow {key}: {errors}"


# ── generate_docker_compose ───────────────────────────────────────────────────

class TestGenerateDockerCompose:

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_output_is_valid_yaml(self, profile_key):
        compose = generate_docker_compose(PROFILES[profile_key])
        doc = yaml.safe_load(compose)
        assert isinstance(doc, dict)
        assert "services" in doc

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_app_service_always_present(self, profile_key):
        doc = yaml.safe_load(generate_docker_compose(PROFILES[profile_key]))
        assert "app" in doc["services"]

    def test_java_has_postgres_service(self):
        doc = yaml.safe_load(generate_docker_compose(PROFILES["java_maven"]))
        assert "db" in doc["services"]
        assert "postgres" in doc["services"]["db"]["image"]

    def test_python_has_redis_service(self):
        doc = yaml.safe_load(generate_docker_compose(PROFILES["python"]))
        assert "redis" in doc["services"]

    def test_javascript_has_no_extra_services(self):
        doc = yaml.safe_load(generate_docker_compose(PROFILES["javascript"]))
        assert "db" not in doc["services"]
        assert "redis" not in doc["services"]

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_depends_on_never_empty_list(self, profile_key):
        """Régression : depends_on: [] était du YAML invalide pour Docker Compose."""
        doc = yaml.safe_load(generate_docker_compose(PROFILES[profile_key]))
        for svc_name, svc in doc["services"].items():
            deps = svc.get("depends_on")
            if deps is not None:
                assert len(deps) > 0, (
                    f"Service '{svc_name}' dans {profile_key} a un depends_on vide"
                )

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_app_service_has_healthcheck(self, profile_key):
        doc = yaml.safe_load(generate_docker_compose(PROFILES[profile_key]))
        assert "healthcheck" in doc["services"]["app"]

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_networks_defined(self, profile_key):
        doc = yaml.safe_load(generate_docker_compose(PROFILES[profile_key]))
        assert "networks" in doc


# ── generate_dockerfile ───────────────────────────────────────────────────────

class TestGenerateDockerfile:

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_not_empty(self, profile_key):
        df = generate_dockerfile(PROFILES[profile_key])
        assert len(df.strip()) > 0

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_contains_from_instruction(self, profile_key):
        df = generate_dockerfile(PROFILES[profile_key])
        non_comment_lines = [l for l in df.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert non_comment_lines, f"Dockerfile {profile_key} vide après suppression des commentaires"
        assert non_comment_lines[0].startswith("FROM"), (
            f"Première instruction non-commentaire de {profile_key} n'est pas FROM : {non_comment_lines[0]}"
        )

    @pytest.mark.parametrize("profile_key", PROFILES)
    def test_does_not_run_as_root(self, profile_key):
        """Sécurité : le container ne doit pas tourner en root."""
        df = generate_dockerfile(PROFILES[profile_key])
        has_user_directive = "USER " in df
        has_user_creation = any(cmd in df for cmd in ("useradd", "adduser", "addgroup"))
        assert has_user_directive, f"Dockerfile {profile_key} manque la directive USER"
        assert has_user_creation, f"Dockerfile {profile_key} ne crée pas de user non-root"

    def test_java_maven_uses_multistage(self):
        df = generate_dockerfile(PROFILES["java_maven"])
        from_count = df.count("\nFROM ") + (1 if df.startswith("FROM ") else 0)
        assert from_count >= 2, "Dockerfile Java/Maven devrait être multi-stage (build + runtime)"

    def test_python_installs_requirements(self):
        df = generate_dockerfile(PROFILES["python"])
        assert "requirements.txt" in df

    def test_node_installs_npm_dependencies(self):
        df = generate_dockerfile(PROFILES["javascript"])
        assert "npm" in df or "yarn" in df or "pnpm" in df


# ── validate_workflow_strict ──────────────────────────────────────────────────

class TestValidateWorkflowStrict:

    def test_valid_minimal_workflow(self):
        content = """
name: CI
on: [push]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
  sonar-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""
        valid, errors = validate_workflow_strict(content)
        assert valid
        assert errors == []

    def test_rejects_invalid_yaml_syntax(self):
        content = "name: CI\n  bad_indent: :\n"
        valid, errors = validate_workflow_strict(content)
        assert not valid
        assert len(errors) > 0

    def test_rejects_missing_jobs_key(self):
        content = "name: CI\non: [push]\n"
        valid, errors = validate_workflow_strict(content)
        assert not valid
        assert any("jobs" in e for e in errors)

    def test_rejects_missing_name_key(self):
        content = "on: [push]\njobs:\n  build-test:\n    runs-on: ubuntu-latest\n    steps: []\n"
        valid, errors = validate_workflow_strict(content)
        assert not valid
        assert any("name" in e for e in errors)

    def test_rejects_job_without_runs_on(self):
        content = """
name: CI
on: [push]
jobs:
  build-test:
    steps:
      - uses: actions/checkout@v4
  sonar-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""
        valid, errors = validate_workflow_strict(content)
        assert not valid
        assert any("runs-on" in e for e in errors)

    def test_rejects_missing_build_test_job(self):
        content = """
name: CI
on: [push]
jobs:
  sonar-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""
        valid, errors = validate_workflow_strict(content)
        assert not valid
        assert any("build-test" in e for e in errors)

    def test_empty_string_is_invalid(self):
        valid, errors = validate_workflow_strict("")
        assert not valid

    def test_returns_tuple(self):
        result = validate_workflow_strict("name: CI\non: [push]\njobs: {}")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)
