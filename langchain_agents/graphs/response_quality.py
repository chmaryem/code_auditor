"""
response_quality.py — Deterministic quality guard for agent answers.

A 0-LLM-cost safety net used by the CIGraph / CDGraph consolidate nodes. It
does NOT judge whether an answer is *semantically correct* (that would need an
extra LLM call); it checks the answer's structural reliability:

  - complete  : root cause and fix are both present and non-trivial,
  - concrete  : the fix contains actionable content (commands / code / paths),
                not just generic advice,
  - grounded  : the root cause references terms actually seen in the run
                context (logs / issues), OR the agent actually called tools —
                a lightweight anti-hallucination heuristic.

It returns a 0–1 score and a list of flags. Callers may lower the confidence,
surface the flags, or (below CRITICAL_THRESHOLD) fall back to a deterministic
analysis instead of trusting the LLM answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# Below this score the LLM answer is considered unreliable and the caller
# should prefer a deterministic fallback.
CRITICAL_THRESHOLD = 0.30

_CMD_HINTS = (
    "mvn", "npm", "pip", "docker", "git", "gradle", "yarn", "apt", "./",
    "sudo", "export", "kubectl", "helm", "chmod", "curl", "wget", "python",
    "node", "sonar", "trivy", "codeql",
)
_PATH_HINTS = (
    ".xml", ".yml", ".yaml", ".java", ".py", ".js", ".ts", ".json",
    ".gradle", ".properties", ".txt", ".toml", "/",
)
_STOP = set(
    "the a an and or of to in is are was were this that with for from into "
    "your you will can may should must have has does not but they their there "
    "which when what where because due more most less than then also".split()
)


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-zA-Z0-9_]{4,}", (text or "").lower())
        if w not in _STOP
    }


def _is_concrete_fix(fix: str) -> bool:
    """A fix is 'concrete' if it carries actionable content, not just advice."""
    if not fix or len(fix.strip()) < 20:
        return False
    low = fix.lower()
    if "```" in fix:                       # a code block
        return True
    if any(h in low for h in _CMD_HINTS):  # a shell/build command
        return True
    if any(h in low for h in _PATH_HINTS): # a file path / config file
        return True
    return False


def _grounding_ratio(root_cause: str, context_text: str) -> float:
    rc = _tokens(root_cause)
    if not rc:
        return 0.0
    ctx = _tokens(context_text)
    if not ctx:
        return 0.0
    return len(rc & ctx) / max(1, len(rc))


def assess_quality(
    root_cause: str,
    suggested_fix: str,
    context_text: str = "",
    used_tools: bool = False,
) -> Dict[str, Any]:
    """
    Score an agent answer deterministically. No LLM call.

    Returns:
        {score: float 0-1, flags: list[str],
         complete: bool, concrete: bool, grounded: bool}
    """
    flags: List[str] = []

    complete = bool(
        root_cause and len(root_cause.strip()) >= 30
        and suggested_fix and len(suggested_fix.strip()) >= 15
    )
    if not complete:
        flags.append("incomplete")

    concrete = _is_concrete_fix(suggested_fix)
    if not concrete:
        flags.append("fix_generic")

    grounded = bool(used_tools) or _grounding_ratio(root_cause, context_text) >= 0.12
    if not grounded:
        flags.append("not_grounded")

    score = (0.40 if complete else 0.0) \
          + (0.35 if concrete else 0.0) \
          + (0.25 if grounded else 0.0)

    return {
        "score": round(score, 2),
        "flags": flags,
        "complete": complete,
        "concrete": concrete,
        "grounded": grounded,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Chat answers — deterministic quality guard (0 LLM)
# ══════════════════════════════════════════════════════════════════════════════

# En dessous de ce score, la réponse du chat mérite d'être signalée (badge/flags).
CHAT_WARN_THRESHOLD = 0.70

# Marqueurs d'échec LLM connus (repli « clés API », indisponibilité…). Une réponse
# qui les contient n'est pas une vraie réponse → score 0.
_LLM_FAILURE_MARKERS = (
    "llm unavailable", "check api keys", "vérifie tes clés",
    "n'ai pas pu appeler le llm", "could not call the llm",
    "[chat_", "[complete_fn]", "[new_class]",
)


def assess_chat_quality(
    response: str,
    context_text: str = "",
    used_tools: bool = False,
    has_project_context: bool = True,
) -> Dict[str, Any]:
    """
    Évalue DÉTERMINISTIQUEMENT (aucun appel LLM) la fiabilité d'une réponse du chat.

    Ne juge PAS la justesse sémantique (cela demanderait un LLM). Vérifie :
      - substantive : réponse non vide, non triviale, et pas un stub d'échec LLM ;
      - grounded    : anti-hallucination léger — la réponse réutilise des termes du
                      contexte fourni (fichier/RAG/signaux) OU des outils ont été
                      appelés. En mode général (aucun contexte projet), non applicable ;
      - specific    : la réponse porte du concret (code, chemin, backticks) ou est
                      suffisamment détaillée.

    Retourne {score 0-1, flags, grounded, specific}.
    """
    text = (response or "").strip()
    low = text.lower()

    # ── Garde dur : réponse vide / stub d'échec LLM → score 0 ──────────────────
    if len(text) < 15 or any(m in low for m in _LLM_FAILURE_MARKERS):
        return {"score": 0.0, "flags": ["empty_or_failed"],
                "grounded": False, "specific": False}

    flags: List[str] = []

    # ── Grounded (anti-hallucination) — pertinent seulement avec contexte projet ─
    if not has_project_context:
        grounded = True  # général/chitchat : rien à halluciner contre le projet
    else:
        grounded = bool(used_tools) or _grounding_ratio(text, context_text) >= 0.10
        if not grounded:
            flags.append("not_grounded")

    # ── Specific : du concret ou du détail ────────────────────────────────────
    specific = (
        len(text) >= 120
        or "```" in text
        or "`" in text
        or any(h in low for h in _PATH_HINTS)
    )
    if not specific:
        flags.append("thin")

    score = 0.50 + (0.30 if grounded else 0.0) + (0.20 if specific else 0.0)
    return {
        "score": round(score, 2),
        "flags": flags,
        "grounded": grounded,
        "specific": specific,
    }
