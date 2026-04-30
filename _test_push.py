"""
_test_push.py — Diagnostic des runs GitHub Actions.

CORRECTIONS v2 :
  - Endpoint /logs → télécharge le ZIP de logs (nécessite actions:read scope)
    Si 403 : affiche un message clair sur le scope manquant au lieu de crasher
  - Alternative : affiche les annotations (erreurs) depuis l'endpoint /annotations
    qui lui NE requiert pas de scope spécial (repo:read suffit)
  - Affiche les étapes avec le message d'erreur quand disponible
"""

import sys, json, urllib.request, urllib.error, os, zipfile, io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
if not token:
    print("ERREUR: GITHUB_PERSONAL_ACCESS_TOKEN non defini dans .env")
    sys.exit(1)

owner, repo = "chmaryem", "test-project-"


def api(endpoint, accept="application/vnd.github.v3+json"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/{endpoint}",
        headers={
            "Authorization": f"token {token}",
            "Accept": accept,
        },
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def api_raw(endpoint):
    """Retourne les bytes bruts (pour les ZIP de logs)."""
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/{endpoint}",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


# ── 1. Lister les runs récents ────────────────────────────────────────────────
print("=" * 60)
print("RUNS RECENTS")
print("=" * 60)

runs = api("actions/runs?per_page=3")["workflow_runs"]
for r in runs:
    print(f"Run #{r['id']} | {r['status']:10} | {str(r['conclusion']):10} | {r['name']}")

if not runs:
    print("Aucun run trouve.")
    sys.exit(0)

# ── 2. Jobs du dernier run ────────────────────────────────────────────────────
run_id = runs[0]["id"]
print(f"\nDETAIL RUN #{run_id}")
print("=" * 60)

jobs = api(f"actions/runs/{run_id}/jobs")["jobs"]
for j in jobs:
    status_icon = "OK" if j["conclusion"] == "success" else "FAIL"
    print(f"\n[{status_icon}] Job: {j['name']}")
    for step in j.get("steps", []):
        s_icon = "OK" if step["conclusion"] == "success" else (step["conclusion"] or "?")
        print(f"  [{s_icon:8}] {step['name']}")

# ── 3. Annotations (erreurs) — ne nécessite pas actions:read ─────────────────
print("\n" + "=" * 60)
print("ANNOTATIONS (ERREURS DETECTEES)")
print("=" * 60)

for j in jobs:
    if j["conclusion"] == "failure":
        print(f"\n-- {j['name']} --")
        try:
            annotations = api(f"check-runs/{j['id']}/annotations")
            if annotations:
                for ann in annotations[:10]:
                    print(f"  [{ann.get('annotation_level','?').upper()}] "
                          f"L{ann.get('start_line','?')}: {ann.get('message','')[:200]}")
            else:
                print("  Aucune annotation disponible.")
        except urllib.error.HTTPError as e:
            print(f"  Annotations non disponibles (HTTP {e.code})")

# ── 4. Logs bruts (nécessite scope actions:read sur le token) ────────────────
print("\n" + "=" * 60)
print("LOGS BRUTS (dernier job en echec)")
print("=" * 60)
print("NOTE: Les logs bruts necessitent le scope 'actions:read' sur votre token.")
print("Si vous avez une erreur 403 ci-dessous, allez dans :")
print("  github.com > Settings > Developer settings > Personal access tokens")
print("  Cochez : 'workflow' ou 'actions' (selon le type de token)")
print()

for j in jobs:
    if j["conclusion"] == "failure":
        print(f"=== LOGS: {j['name']} ===")
        try:
            log_url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{j['id']}/logs"
            req = urllib.request.Request(
                log_url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()

            # Les logs peuvent être un ZIP ou du texte brut
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
                for name in zf.namelist():
                    lines = zf.read(name).decode("utf-8", errors="replace").splitlines()
                    for line in lines[-80:]:  # 80 dernières lignes
                        print(line)
            except zipfile.BadZipFile:
                # C'est du texte brut (redirection suivie)
                lines = raw.decode("utf-8", errors="replace").splitlines()
                for line in lines[-80:]:
                    print(line)

        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"  ERREUR 403 — Scope manquant sur votre token.")
                print(f"  Solution : ajoutez le scope 'workflow' à votre PAT.")
                print(f"  En attendant, consultez les logs directement sur GitHub :")
                print(f"  https://github.com/{owner}/{repo}/actions/runs/{run_id}")
            else:
                body = e.read().decode("utf-8", errors="replace")[:200]
                print(f"  Erreur HTTP {e.code}: {body}")
        except Exception as e:
            print(f"  Erreur: {e}")
        print()