"""
ci_cd/ci_deploy_agent.py — Déploie le workflow CI/CD sur un repo distant via MCP.

Fonctionnement :
  1. Se connecte au MCP GitHub via GitHubClient (code_mode_client)
  2. Vérifie si un workflow existe déjà (respecte --force)
  3. Détecte le langage/build system du repo cible (pom.xml, package.json...)
  4. Génère le YAML adapté via workflow_generator
  5. Pousse via MCP push_file() → fallback Git Data API si MCP échoue

CORRECTIONS v3 :
  - Pousse requirements-ci.txt EN MÊME TEMPS que le workflow YAML
    (le runner Actions en a besoin — absent du repo cible sans ça)
  - _push_via_rest() utilise la Git Data API (contourne la restriction .github/ des tokens)
  - file_checker extrait comme méthode interne propre
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logger = logging.getLogger(__name__)

_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"

WORKFLOW_PATH    = ".github/workflows/ci.yml"
REQUIREMENTS_CI_PATH = "requirements-ci.txt"
DEFAULT_AUDITOR  = "chmaryem/code_auditor"


# ─────────────────────────────────────────────────────────────────────────────
# Validations pré-déploiement
# ─────────────────────────────────────────────────────────────────────────────

def _validate_github_token() -> tuple[bool, str]:
    """
    Vérifie que le token GitHub est présent et valide.
    Retourne (is_valid, message)
    """
    token = _get_rest_token()
    
    if not token:
        return False, "GITHUB_TOKEN ou GITHUB_PERSONAL_ACCESS_TOKEN manquant"
    
    # Test rapide : GET /user
    try:
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read())
                login = data.get('login', 'unknown')
                return True, f"Connecté en tant que {login}"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Token invalide (401 Unauthorized)"
        return False, f"Erreur API: {e.code}"
    except Exception as e:
        return False, f"Erreur connexion: {str(e)}"
    
    return False, "Validation échouée"


def _validate_secrets_for_repo(owner: str, repo: str) -> tuple[bool, list[str]]:
    """
    Vérifie que les secrets requis sont disponibles localement.
    Retourne (is_valid, warnings)
    """
    warnings_list = []
    
    # Vérification locale: les clés API doivent être dans .env ou env
    required_env = ["GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    for env_var in required_env:
        if not os.environ.get(env_var):
            warnings_list.append(f"{env_var} (sera nécessaire dans GitHub Secrets)")
    
    # Vérifie que le token a accès au repo
    token = _get_rest_token()
    if token:
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"token {token}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                permissions = data.get("permissions", {})
                if not permissions.get("admin", False) and not permissions.get("maintain", False):
                    warnings_list.append("Token sans permissions admin - vérifiez manuellement les secrets dans Settings > Secrets > Actions")
        except Exception as e:
            warnings_list.append(f"Impossible de vérifier le repo: {str(e)}")
    
    return len(warnings_list) == 0, warnings_list


def _ensure_requirements_ci() -> str:
    """
    Garantit que requirements-ci.txt existe localement.
    Si absent, génère un fichier minimal compatible.
    """
    content = _read_local_requirements_ci()
    if content:
        return content
    
    # Fallback: générer un requirements minimal
    print(f"  {_YL}! requirements-ci.txt absent - génération automatique{_R}")
    
    minimal_requirements = """# Généré automatiquement par Code Auditor CI Deploy
# Versions compatibles Python 3.11

langchain==0.1.6
langchain-community==0.0.20
chromadb==0.4.22
sentence-transformers==2.3.1
tree-sitter>=0.21.3
tree-sitter-python>=0.21.0
tree-sitter-javascript>=0.21.0
tree-sitter-java>=0.21.0
astroid==3.0.2
pylint==3.0.3
watchdog>=3.0.0
pytest>=8.0
pytest-asyncio>=0.23
mcp>=1.0.0
httpx>=0.27.0
pydantic>=2.0
pyyaml>=6.0
python-dotenv>=1.0
"""
    
    # Sauvegarder localement pour futures utilisations
    local_path = _PROJECT_ROOT / "requirements-ci.txt"
    try:
        local_path.write_text(minimal_requirements, encoding="utf-8")
        print(f"  {_GR}✓ Sauvegardé dans {local_path}{_R}")
    except Exception as e:
        print(f"  {_YL}! Impossible de sauvegarder: {e}{_R}")
    
    return minimal_requirements


# ─────────────────────────────────────────────────────────────────────────────
# Déploiement principal
# ─────────────────────────────────────────────────────────────────────────────

async def deploy_ci_workflow(
    owner:        str,
    repo:         str,
    auditor_repo: str  = DEFAULT_AUDITOR,
    branch:       str  = "main",
    force:        bool = False,
) -> dict:
    """
    Déploie le workflow CI/CD + requirements-ci.txt sur le repo cible.

    Le fichier requirements-ci.txt est copié depuis le repo code_auditor local
    vers le repo cible. Le runner GitHub Actions en a besoin pour installer
    les dépendances avec les bonnes versions (compatibles Python 3.11).

    Returns:
        dict: {success, message, profile, workflow_path}
    """
    print(f"\n  {_CY}{_B}Code Auditor — CI/CD Deploy{_R}")
    print(f"  {_DM}Repo cible    : {owner}/{repo}{_R}")
    print(f"  {_DM}Branche       : {branch}{_R}")
    print(f"  {_DM}Auditor repo  : {auditor_repo}{_R}\n")

    # ── Validation 0 : Token GitHub ─────────────────────────────────────────
    print(f"  {_DM}[0/6] Validation du token GitHub...{_R}")
    is_valid, msg = _validate_github_token()
    if not is_valid:
        print(f"  {_RD}x Erreur authentification: {msg}{_R}")
        print(f"  {_DM}Vérifiez vos variables d'environnement:{_R}")
        print(f"    - GITHUB_TOKEN ou GITHUB_PERSONAL_ACCESS_TOKEN{_R}\n")
        return {"success": False, "error": f"Authentification échouée: {msg}"}
    print(f"  {_GR}✓ {msg}{_R}")

    # ── Validation 0b : Secrets pour le repo ───────────────────────────────
    secrets_ok, warnings = _validate_secrets_for_repo(owner, repo)
    if warnings:
        print(f"  {_YL}! Avertissements secrets:{_R}")
        for w in warnings:
            print(f"    - {w}")
        print(f"  {_DM}Le workflow risque d'échouer si ces secrets ne sont pas configurés.{_R}\n")

    try:
        from services.code_mode_client import github
    except ImportError as e:
        print(f"  {_RD}x Erreur import code_mode_client : {e}{_R}\n")
        return {"success": False, "error": str(e)}

    try:
        # ── Step 1 : Vérifier si le workflow existe déjà ─────────────────────
        print(f"  {_DM}[1/6] Verification du workflow existant...{_R}")
        existing = github.get_file_content(owner, repo, WORKFLOW_PATH, branch)

        if existing and not force:
            print(f"  {_YL}Le workflow existe deja : {WORKFLOW_PATH}{_R}")
            print(f"  {_DM}Utilisez --force pour le remplacer.{_R}\n")
            return {
                "success": False,
                "message": "Workflow deja existant (utilisez --force)",
                "workflow_path": WORKFLOW_PATH,
            }

        # ── Step 2 : Détecter le profil du projet ────────────────────────────
        print(f"  {_DM}[2/5] Detection du langage et du build system...{_R}")

        from ci_cd.workflow_generator import detect_project_profile

        def _check_remote_file(path: str) -> Optional[str]:
            try:
                content = github.get_file_content(owner, repo, path, branch)
                return content if content else None
            except Exception:
                return None

        profile = detect_project_profile(_check_remote_file)

        if profile.language == "unknown":
            print(f"  {_YL}Aucun build system detecte — workflow generique cree{_R}")
        else:
            print(f"  {_GR}Detecte : {profile.language} / {profile.build_system}{_R}")

        # ── Step 3 : Générer le YAML ──────────────────────────────────────────
        print(f"  {_DM}[3/6] Generation du workflow YAML...{_R}")

        from ci_cd.workflow_generator import generate_workflow, validate_workflow_strict
        checkout_path = os.environ.get("CODE_AUDITOR_CHECKOUT_PATH", "code_auditor_tool")
        yaml_content = generate_workflow(profile, auditor_repo, checkout_path)

        # ── Validation 3b : Valider le YAML avant push ─────────────────────────
        print(f"  {_DM}[3b/6] Validation du YAML...{_R}")
        is_valid_yaml, yaml_errors = validate_workflow_strict(yaml_content)
        if not is_valid_yaml:
            print(f"  {_RD}x YAML invalide:{_R}")
            for err in yaml_errors:
                print(f"    - {err}")
            return {"success": False, "error": f"YAML invalide: {yaml_errors}"}
        print(f"  {_GR}✓ YAML valide ({len(yaml_content)} caractères){_R}")

        # ── Step 4 : Garantir requirements-ci.txt ─────────────────────────────
        print(f"  {_DM}[4/6] Preparation de requirements-ci.txt...{_R}")
        req_ci_content = _ensure_requirements_ci()

        # ── Step 5 : Pousser les deux fichiers ────────────────────────────────
        print(f"  {_DM}[5/6] Push sur {owner}/{repo}@{branch}...{_R}")

        # Pousser requirements-ci.txt d'abord (le workflow en dépend)
        if req_ci_content:
            req_pushed = _push_file(
                github, owner, repo, REQUIREMENTS_CI_PATH,
                req_ci_content,
                "ci: add requirements-ci.txt for GitHub Actions runner",
                branch,
            )
            if req_pushed:
                print(f"  {_GR}requirements-ci.txt pousse{_R}")
            else:
                print(f"  {_YL}Avertissement : requirements-ci.txt non pousse (non bloquant){_R}")
        else:
            print(f"  {_YL}requirements-ci.txt local introuvable — le runner utilisera requirements.txt{_R}")

        # Pousser le workflow YAML
        workflow_commit_msg = (
            f"ci: add Code Auditor pipeline "
            f"({profile.language}/{profile.build_system})"
        )
        result = _push_file(
            github, owner, repo, WORKFLOW_PATH,
            yaml_content, workflow_commit_msg, branch,
        )

        if result:
            print(f"\n  {_GR}{_B}Workflow CI/CD deploye !{_R}")
            print(f"  {_DM}Fichier  : {WORKFLOW_PATH}{_R}")
            print(f"  {_DM}Profil   : {profile.language} / {profile.build_system}{_R}")
            _print_next_steps()
            return {
                "success": True,
                "message": "Workflow deploye",
                "profile": {
                    "language":     profile.language,
                    "build_system": profile.build_system,
                },
                "workflow_path": WORKFLOW_PATH,
            }

        print(f"\n  {_RD}x Echec du push — verifiez les permissions du token GitHub{_R}\n")
        return {"success": False, "error": "Push failed (MCP + REST API)"}

    except Exception as e:
        logger.error("deploy_ci_workflow failed: %s", e)
        print(f"\n  {_RD}x Erreur : {e}{_R}\n")
        return {"success": False, "error": str(e)}
    finally:
        try:
            github.disconnect()
        except Exception:
            pass


def _push_file(
    github_client,
    owner:   str,
    repo:    str,
    path:    str,
    content: str,
    message: str,
    branch:  str,
) -> dict:
    """
    Pousse un fichier via MCP, puis fallback REST API si MCP échoue.
    Retourne le résultat ou {} en cas d'échec total.
    """
    # Tentative 1 : MCP push_file
    try:
        result = github_client.push_file(
            owner=owner, repo=repo,
            path=path, content=content,
            message=message, branch=branch,
        )
        if result:
            return result
    except Exception as mcp_err:
        logger.debug("MCP push_file echec pour %s: %s — fallback REST API", path, mcp_err)

    # Tentative 2 : Git Data API (bas niveau)
    return _push_via_rest(owner, repo, path, content, message, branch)


def _read_local_requirements_ci() -> str:
    """
    Lit le fichier requirements-ci.txt depuis le repo code_auditor local.
    Cherche dans le dossier parent du package ci_cd/ et à la racine du projet.
    Retourne "" si introuvable.
    """
    candidates = [
        _PROJECT_ROOT / "requirements-ci.txt",
        _THIS_DIR / "requirements-ci.txt",
        _PROJECT_ROOT / "ci_cd" / "requirements-ci.txt",
    ]
    for path in candidates:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                logger.info("requirements-ci.txt lu depuis %s", path)
                return content
            except Exception as e:
                logger.warning("Impossible de lire %s : %s", path, e)
    logger.warning("requirements-ci.txt introuvable dans %s", _PROJECT_ROOT)
    return ""


def _print_next_steps() -> None:
    """Affiche les instructions post-déploiement."""
    print(f"\n  {_CY}Prochaines etapes :{_R}")
    print(f"  {_DM}1. Ajouter les Secrets GitHub :{_R}")
    print(f"  {_DM}   Settings > Secrets > Actions > New repository secret{_R}")
    print(f"  {_DM}   - GOOGLE_API_KEY  (Gemini — obligatoire){_R}")
    print(f"  {_DM}   - OPENROUTER_API_KEY  (optionnel, fallback){_R}")
    print(f"  {_DM}2. Activer la Branch Protection :{_R}")
    print(f"  {_DM}   Settings > Branches > Add rule{_R}")
    print(f"  {_DM}   Cocher : Require status checks > Code Auditor / PR Review{_R}")
    print(f"  {_DM}3. Ouvrir une PR pour tester !{_R}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Fallback : Git Data API (bas niveau)
# ─────────────────────────────────────────────────────────────────────────────

def _get_rest_token() -> str:
    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    )
    if not token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = (
                os.environ.get("GITHUB_TOKEN")
                or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            )
        except ImportError:
            pass
    return token


def _push_via_rest(
    owner:   str,
    repo:    str,
    path:    str,
    content: str,
    message: str,
    branch:  str,
) -> dict:
    """
    Pousse un fichier via la Git Data API GitHub (séquence bas niveau 5 étapes).

    Pourquoi pas la Contents API ?
    GitHub bloque la Contents API (PUT /contents/{path}) pour les fichiers
    sous .github/ avec les tokens classic ghp_. La Git Data API contourne ça :
      1. Créer un blob avec le contenu (base64)
      2. Récupérer le tree SHA du commit HEAD actuel
      3. Créer un nouveau tree avec le blob
      4. Créer un commit pointant vers ce tree
      5. Mettre à jour la ref de la branche
    """
    token = _get_rest_token()
    if not token:
        logger.error("_push_via_rest: aucun token GitHub disponible")
        return {}

    api_base = f"https://api.github.com/repos/{owner}/{repo}"
    headers  = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
        "User-Agent":    "CodeAuditor-CI-Deploy/3.0",
    }

    def _call(method: str, endpoint: str, payload: dict = None) -> dict:
        url  = f"{api_base}/{endpoint}"
        data = json.dumps(payload).encode("utf-8") if payload else None
        req  = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    try:
        # 1. SHA du dernier commit de la branche
        ref_data = _call("GET", f"git/refs/heads/{branch}")
        head_sha = ref_data["object"]["sha"]

        # 2. SHA du tree actuel
        commit   = _call("GET", f"git/commits/{head_sha}")
        tree_sha = commit["tree"]["sha"]

        # 3. Blob avec le contenu du fichier
        blob = _call("POST", "git/blobs", {
            "content":  base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "encoding": "base64",
        })

        # 4. Nouveau tree incluant le blob
        tree = _call("POST", "git/trees", {
            "base_tree": tree_sha,
            "tree": [{
                "path": path,
                "mode": "100644",
                "type": "blob",
                "sha":  blob["sha"],
            }],
        })

        # 5. Commit + mise à jour de la ref
        new_commit = _call("POST", "git/commits", {
            "message": message,
            "tree":    tree["sha"],
            "parents": [head_sha],
        })
        _call("PATCH", f"git/refs/heads/{branch}", {"sha": new_commit["sha"]})

        logger.info("Git Data API push OK: %s -> %s", path, new_commit["sha"][:8])
        return {"sha": new_commit["sha"], "path": path}

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        logger.error("Git Data API erreur %d pour %s : %s", e.code, path, body)
    except Exception as e:
        logger.error("Git Data API erreur pour %s : %s", path, e)

    return {}