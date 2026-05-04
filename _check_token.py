"""Diagnostic rapide du token GitHub."""
import os
from dotenv import load_dotenv
load_dotenv()

t1 = os.environ.get("GITHUB_TOKEN", "")
t2 = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
token = t1 or t2

print(f"GITHUB_TOKEN:                  {'[' + t1[:8] + '...] (' + str(len(t1)) + ' chars)' if t1 else '(vide)'}")
print(f"GITHUB_PERSONAL_ACCESS_TOKEN:  {'[' + t2[:8] + '...] (' + str(len(t2)) + ' chars)' if t2 else '(vide)'}")
print()

if not token:
    print("AUCUN TOKEN TROUVE !")
    print("Ajoutez dans votre .env :")
    print('  GITHUB_PERSONAL_ACCESS_TOKEN=ghp_votre_token_ici')
elif token.startswith("AIza"):
    print("ERREUR : C'est une cle Google API, pas un token GitHub !")
    print("Un token GitHub commence par ghp_ ou github_pat_")
elif token.startswith("ghp_") or token.startswith("github_pat_"):
    print("Format OK. Test de connexion...")
    import urllib.request, json
    try:
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            print(f"Connecte en tant que: {data.get('login')}")
    except Exception as e:
        print(f"Echec: {e}")
else:
    print(f"Format inconnu (commence par: {token[:4]})")
    print("Un token GitHub doit commencer par ghp_ ou github_pat_")
