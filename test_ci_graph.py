"""
test_ci_graph.py — Test end-to-end du CIGraph sur le dernier run de test-project-

Usage:
    python test_ci_graph.py
"""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 0. Load .env FIRST (avant tout import de services) ───────────────────────
print("\n[0/5] Chargement .env...")
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    print("  ✓ .env chargé")

token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
print(f"  Token GitHub : {'✓ présent (' + token[:4] + '...)' if token else '✗ MANQUANT'}")

REPO       = "chmaryem/test-project-"
OWNER      = "chmaryem"
REPO_NAME  = "test-project-"
PROJECT_KEY = "chmaryem_test-project-"

# ── 1. Reset Redis seen key ───────────────────────────────────────────────────
print("\n[1/5] Reset Redis seen key...")
try:
    from services.mcp_redis_service import get_mcp_redis
    redis = get_mcp_redis()
    seen_key = f"ci:poll:seen:{REPO}"
    redis.delete(seen_key)
    print(f"  ✓ Clé '{seen_key}' supprimée")
except Exception as e:
    print(f"  ! Redis non disponible: {e} (skip)")

# ── 2. Récupérer le dernier run via REST API ──────────────────────────────────
print("\n[2/5] Récupération des runs GitHub Actions (REST)...")
if not token:
    print("  ✗ Token manquant — impossible de continuer")
    sys.exit(1)

import urllib.request, urllib.parse
try:
    params = urllib.parse.urlencode({"status": "completed", "per_page": "5"})
    url = f"https://api.github.com/repos/{OWNER}/{REPO_NAME}/actions/runs?{params}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    runs = data.get("workflow_runs", [])
    print(f"  ✓ {len(runs)} runs trouvés")
except Exception as e:
    print(f"  ✗ Erreur REST API: {e}")
    runs = []

if not runs:
    print("  ✗ Aucun run — vérifiez le repo et le token")
    sys.exit(1)

# Afficher les runs disponibles
print("\n  Runs disponibles :")
for i, r in enumerate(runs[:5]):
    print(f"    [{i}] #{r['id']} | {r.get('name','?')[:40]:40} | {r.get('conclusion','?'):10} | {r.get('head_branch','?')}")

# Prendre le plus récent
latest     = runs[0]
run_id     = str(latest["id"])
conclusion = latest.get("conclusion", "")
head_sha   = latest.get("head_sha", "")
prs        = [pr["number"] for pr in latest.get("pull_requests", [])]
pr_number  = prs[0] if prs else None

print(f"\n  → Run sélectionné : #{run_id} | conclusion={conclusion} | sha={head_sha[:8]}")

# ── 3. Invoquer le CIGraph ────────────────────────────────────────────────────
print(f"\n[3/5] Invocation CIGraph pour run #{run_id}...")
try:
    from langchain_agents.graphs.ci_graph import invoke_ci_run
    result = invoke_ci_run(
        run_id               = run_id,
        repo                 = REPO,
        owner                = OWNER,
        project_key          = PROJECT_KEY,
        pr_number            = pr_number,
        head_sha             = head_sha,
        run_conclusion       = conclusion,
        run_duration_seconds = 0,
    )
    print("  ✓ CIGraph exécuté")
except Exception as e:
    print(f"  ✗ CIGraph erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── 4. Afficher les résultats ─────────────────────────────────────────────────
print("\n[4/5] Résultats CIGraph :")
print("=" * 65)
print(f"  Outcome          : {result.get('outcome', 'N/A')}")
print(f"  Failure type     : {result.get('failure_type', 'N/A')}")
print(f"  Severity         : {result.get('severity', 'N/A')}")
print(f"  Notification     : {result.get('notification_level', 'N/A')}")
print(f"  Confidence Redis : {result.get('confidence', 0):.0%}")
print(f"  Comment posted   : {result.get('comment_posted', False)}")
print(f"  Indexed in Redis : {result.get('indexed', False)}")

sonar = result.get("sonar_gate", {})
if sonar:
    status = sonar.get("status", "N/A")
    badge  = "🟢 OK" if status == "OK" else "🔴 FAILED" if status == "ERROR" else "🟡 " + status
    print(f"\n  SonarCloud Gate  : {badge}")
    for cond in sonar.get("conditions", [])[:3]:
        if cond.get("status") != "OK":
            print(f"    ✗ {cond.get('metric')}: {cond.get('actualValue')} (seuil: {cond.get('errorThreshold')})")

root = result.get("root_cause")
if root:
    print(f"\n  Root Cause:\n    {root[:400]}")

fix = result.get("suggested_fix")
if fix:
    print(f"\n  Suggested Fix:\n    {fix[:400]}")

similar = result.get("similar_fixes", [])
print(f"\n  Similar in Redis : {len(similar)} match(es)")

print("=" * 65)

# ── 5. Sauvegarder le résultat JSON ──────────────────────────────────────────
print("\n[5/5] Sauvegarde résultat...")
out = Path(__file__).parent / "ci_graph_result.json"
try:
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Résultat sauvegardé → {out.name}")
except Exception as e:
    print(f"  ! Sauvegarde échouée: {e}")

print("\n Test end-to-end terminé !\n")
