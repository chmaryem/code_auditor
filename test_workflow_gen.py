"""Validate generated YAML for all 3 languages."""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

for line in (Path(".env")).read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from ci_cd.workflow_generator import generate_workflow, ProjectProfile

# ── Test 3 langages ───────────────────────────────────────────────────────────
profiles = [
    ProjectProfile(language="java",       build_system="maven", has_dockerfile=True),
    ProjectProfile(language="python",     build_system="pip",   has_dockerfile=True),
    ProjectProfile(language="javascript", build_system="npm",   has_dockerfile=True),
]

KEYWORDS = [
    "publish",
    "deploy",
    "DOCKERHUB_USERNAME",
    "DOCKERHUB_TOKEN",
    "docker compose",
    "health check",
    "appleboy/ssh-action",
    "docker/build-push-action",
    "Prepare image name",      # Remplace docker/metadata-action
    "sed 's/-*$//'",           # Sanitize trailing dashes (fix du bug /test-project-)
]

all_ok = True
for p in profiles:
    print(f"\n=== {p.language}/{p.build_system} ===")
    yaml = generate_workflow(p, enable_publish=True, enable_deploy=True)
    print(f"  Taille YAML : {len(yaml)} chars")

    for kw in KEYWORDS:
        found = kw.lower() in yaml.lower()
        status = "OK" if found else "MANQUANT"
        if not found:
            all_ok = False
        print(f"  {status:8} : {kw}")

print("\n" + ("=== TOUS LES TESTS OK ===" if all_ok else "=== CERTAINS TESTS ECHOUES ==="))
