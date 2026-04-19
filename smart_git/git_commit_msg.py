"""
git_commit_msg.py — Génération automatique de messages de commit via LLM.

╔══════════════════════════════════════════════════════════════════════╗
║  prepare-commit-msg hook                                             ║
║                                                                      ║
║  Appelé par Git APRÈS le pre-commit hook et AVANT que l'éditeur      ║
║  de message ne s'ouvre.                                              ║
║                                                                      ║
║  Flow complet :                                                      ║
║    git commit                                                        ║
║      ├── pre-commit hook → vérifie qualité → bloque ou autorise      ║
║      └── prepare-commit-msg hook (CE FICHIER)                        ║
║           1. Lit git diff --staged                                   ║
║           2. Envoie au LLM Gemini                                    ║
║           3. Génère un message Conventional Commits                  ║
║           4. Écrit le message dans le fichier de commit               ║
║           5. L'utilisateur voit le message dans son éditeur           ║
║              → Il accepte, modifie, ou supprime                      ║
║                                                                      ║
║  Format : Conventional Commits                                       ║
║    feat(scope): description                                          ║
║    fix(scope): description                                           ║
║    refactor(scope): description                                      ║
║    security(scope): description                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# ── ANSI ─────────────────────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_CY = "\033[96m"
_DM = "\033[2m"
_RD = "\033[91m"


# ─────────────────────────────────────────────────────────────────────────────
# Récupération du diff staged
# ─────────────────────────────────────────────────────────────────────────────

def _get_staged_diff(project_path: Path) -> str:
    """Récupère le diff des fichiers staged (git diff --staged)."""
    result = subprocess.run(
        ["git", "diff", "--staged", "--stat"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(project_path)
    )
    stat = result.stdout.strip() if result.returncode == 0 else ""

    result = subprocess.run(
        ["git", "diff", "--staged", "--no-color"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(project_path)
    )
    diff = result.stdout.strip() if result.returncode == 0 else ""

    return stat, diff


def _get_staged_files(project_path: Path) -> list:
    """Liste les fichiers staged."""
    result = subprocess.run(
        ["git", "diff", "--staged", "--name-only"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(project_path)
    )
    if result.returncode != 0:
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]



# Génération du message via LLM
def generate_commit_message(project_path: Path) -> str:
    """
    Génère un message de commit basé sur les changements staged.

    Utilise Gemini via langchain_google_genai (même SDK que le reste du projet).
    """
    stat, diff = _get_staged_diff(project_path)
    files = _get_staged_files(project_path)

    if not diff:
        return ""

    # Tronquer le diff si trop long (budget tokens)
    diff_truncated = diff[:4000]
    if len(diff) > 4000:
        diff_truncated += f"\n\n... ({len(diff) - 4000} chars truncated)"

    prompt = f"""You are a Git expert generating a commit message following Conventional Commits format.

STAGED FILES:
{chr(10).join(f'  • {f}' for f in files)}

DIFF STATS:
{stat}

DIFF CONTENT:
```
{diff_truncated}
```

RULES:
1. Use Conventional Commits format: type(scope): description
2. Types: feat, fix, refactor, security, test, docs, style, chore, perf
3. Scope = the main module/class affected (lowercase, short)
4. Description = imperative mood, lowercase, no period, max 50 chars
5. Add a blank line then 2-3 bullet points explaining the key changes
6. Write in English
7. Be specific: "add password validation" NOT "update code"

EXAMPLES:
  feat(auth): add bcrypt password hashing
  fix(user): prevent sql injection in findByUsername
  refactor(service): extract validation logic to helper
  security(auth): replace md5 with sha256 for token generation

OUTPUT FORMAT (exactly this, no markdown fences):
type(scope): short description

- bullet point 1
- bullet point 2

GENERATE THE COMMIT MESSAGE:"""

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv

        load_dotenv(Path(_project_root) / ".env")
        api_key = os.getenv("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            max_output_tokens=256,
            temperature=0.3,
        )
        response = llm.invoke(prompt)
        message = response.content.strip()

        # Nettoyer les blocs markdown si présents
        if message.startswith("```"):
            lines = message.splitlines()
            if lines[-1].strip() == "```":
                message = "\n".join(lines[1:-1])
            else:
                message = "\n".join(lines[1:])

        return message

    except Exception as e:
        return ""


# Hook prepare-commit-msg

def run_prepare_commit_msg(project_path: Path, commit_msg_file: str) -> int:
    """
    Point d'entrée du hook prepare-commit-msg.

    Appelé par Git avec le chemin du fichier de message comme argument.
    Le contenu de ce fichier est ce que l'utilisateur verra dans son éditeur.
    """
    msg_path = Path(commit_msg_file)

    # Ne pas générer si l'utilisateur a déjà fourni un message (-m "message")
    if msg_path.exists():
        existing = msg_path.read_text(encoding="utf-8", errors="replace").strip()
        # Git met un contenu par défaut vide ou avec des commentaires #
        real_content = "\n".join(
            l for l in existing.splitlines() if not l.startswith("#")
        ).strip()
        if real_content:
            return 0  # L'utilisateur a déjà écrit un message → ne pas écraser

    print(f"\n  {_CY}{_B}Code Auditor — Suggestion de message de commit{_R}")
    print(f"  {_DM}Analyse du diff staged...{_R}", end="", flush=True)

    message = generate_commit_message(project_path)

    if not message:
        print(f"  {_DM}(pas de suggestion disponible){_R}\n")
        return 0

    print(f"  {_GR}✓{_R}\n")
    print(f"  {_B}Message suggéré :{_R}")
    print(f"  ┌{'─' * 58}┐")
    for line in message.splitlines():
        print(f"  │ {line:<56} │")
    print(f"  └{'─' * 58}┘")
    print(f"\n  {_DM}Ce message sera pré-rempli dans votre éditeur.{_R}")
    print(f"  {_DM}Modifiez-le ou sauvegardez tel quel pour accepter.{_R}\n")

    # Écrire le message dans le fichier de commit
    # Les commentaires Git existants sont préservés en dessous
    existing_comments = ""
    if msg_path.exists():
        existing_comments = "\n".join(
            l for l in msg_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.startswith("#")
        )

    final = message
    if existing_comments:
        final += "\n\n" + existing_comments

    msg_path.write_text(final, encoding="utf-8")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Code Auditor — Commit Message Generator")
    parser.add_argument("--project", type=str, default=".", help="Chemin du projet git")
    parser.add_argument("--msg-file", type=str, default="", help="Fichier de message commit")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()

    if args.msg_file:
        sys.exit(run_prepare_commit_msg(project_path, args.msg_file))
    else:
        # Mode standalone : juste afficher le message suggéré
        message = generate_commit_message(project_path)
        if message:
            print(message)
        else:
            print("Aucun changement staged ou erreur LLM.")
