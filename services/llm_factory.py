from __future__ import annotations

import logging
import os
import time
import threading
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Délais de backoff en cas de quota 429 (secondes) - 4 tentatives pour le modèle
_BACKOFF = [10, 30, 60, 120]

# Erreurs qui déclenchent le backoff ou le passage au provider suivant
_QUOTA_ERRORS = ("429", "RESOURCE_EXHAUSTED", "quota", "RateLimitError", "rate_limit", "503", "524")
_AUTH_ERRORS  = ("401", "403", "AuthenticationError", "invalid_api_key", "API_KEY_INVALID")

def _is_quota_error(err: str) -> bool:
    return any(k in err for k in _QUOTA_ERRORS)

def _is_auth_error(err: str) -> bool:
    return any(k in err for k in _AUTH_ERRORS)


# ── HTTP-direct callers (utilisés par invoke_with_fallback UNIQUEMENT) ────────
# Raison : langchain_openai/langchain_google_genai déclenchent un import
# PyTorch au chargement → fbgemm.dll manquant sur Windows → crash.
# Ces 2 fonctions font le même appel REST en urllib stdlib (0 dépendance).
# Les _build_*_llm() restent intacts pour le multi-agent LangGraph.

def _call_openrouter_http(prompt: str, api_key: str, model: str, max_tokens: int = 2048) -> str:
    import json, urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/code-auditor",
            "X-Title":       "Code Auditor",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"OpenRouter: réponse vide — {data}")
    return choices[0]["message"]["content"]


def _call_gemini_http(prompt: str, api_key: str, model: str = "gemini-2.0-flash", max_tokens: int = 2048) -> str:
    import json, urllib.request
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini: réponse vide — {data}")
    return candidates[0]["content"]["parts"][0]["text"]



def _build_openrouter_llm(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
):
    """Construit un LLM via OpenRouter (API compatible OpenAI).
    Utilisé pour MiniMax M2.5 et tout autre modèle disponible sur OpenRouter.
    """
    from langchain_openai import ChatOpenAI
    from config import config

    api_key = config.api.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY non défini")

    resolved_model = model or config.api.openrouter_model

    return ChatOpenAI(
        model           = resolved_model,
        api_key         = api_key,
        base_url        = "https://openrouter.ai/api/v1",
        temperature     = temperature,
        max_tokens      = max_tokens,
        default_headers = {
            "HTTP-Referer": "https://github.com/code-auditor",
            "X-Title":      "Code Auditor",
        },
    ), resolved_model 


def _build_gemini_llm(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
):
  
    from langchain_google_genai import ChatGoogleGenerativeAI
    from config import config

    api_key = config.api.gemini_api_key or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY non défini")

    return ChatGoogleGenerativeAI(
        model                           = model or config.api.gemini_model,
        google_api_key                  = api_key,
        temperature                     = temperature,
        max_output_tokens               = max_tokens,
        convert_system_message_to_human = True,
    )

def get_primary_llm(temperature: float = 0.0, max_tokens: int = 8192):
   
    llm, model_name = _build_openrouter_llm(temperature=temperature, max_tokens=max_tokens)
    return llm


def invoke_with_fallback(
    prompt: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    label: str = "LLM",
) -> Optional[str]:
    """
    Cascade OpenRouter → Gemini pour le CI Analyzer et la couche IA.

    Stratégie :
      1. Essaie d'abord via HTTP direct (_call_*_http) — 0 dépendance PyTorch.
         → Résout le crash fbgemm.dll sur Windows.
      2. Si HTTP échoue (erreur réseau, etc.), tente LangChain comme fallback.
         → Garde la compatibilité avec le multi-agent watch mode.
    """
    from config import config

    # Normaliser le prompt en str (compatible HumanMessage LangChain)
    if hasattr(prompt, "content"):
        prompt_text = str(prompt.content)
    elif not isinstance(prompt, str):
        prompt_text = str(prompt)
    else:
        prompt_text = prompt

    # ── 1. OpenRouter via HTTP (zéro PyTorch) ────────────────────────────────
    api_key = config.api.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if api_key:
        model      = config.api.openrouter_model or "mistralai/mistral-7b-instruct:free"
        short_name = model.split("/")[-1].replace(":free", "")
        for attempt in range(4):
            try:
                print(f"    [{short_name}] {label} — appel HTTP (tentative {attempt + 1}/4)...")
                text = _call_openrouter_http(prompt_text, api_key, model, max_tokens)
                logger.info("[%s] réponse via OpenRouter/%s (HTTP)", label, short_name)
                return text
            except Exception as e:
                err = str(e)
                if _is_quota_error(err):
                    if attempt < 3:
                        wait = _BACKOFF[attempt]
                        print(f"   ⚠️  [{short_name}] quota — attente {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error("[%s] OpenRouter épuisé après 4 tentatives", label); break
                elif _is_auth_error(err):
                    logger.error("[%s] OpenRouter clé invalide", label); break
                else:
                    logger.warning("[%s] OpenRouter HTTP erreur: %s", label, err[:150]); break

    # ── 2. Gemini via HTTP (zéro PyTorch) ────────────────────────────────────
    api_key = config.api.gemini_api_key or os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        model = config.api.gemini_model or "gemini-2.0-flash"
        for attempt in range(4):
            try:
                print(f"    [Gemini] {label} — appel HTTP (tentative {attempt + 1}/4)...")
                text = _call_gemini_http(prompt_text, api_key, model, max_tokens)
                logger.info("[%s] réponse via Gemini (HTTP)", label)
                return text
            except Exception as e:
                err = str(e)
                if _is_quota_error(err):
                    if attempt < 3:
                        wait = _BACKOFF[attempt]
                        print(f"   ⚠️  [Gemini] quota — attente {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error("[%s] Gemini épuisé après 4 tentatives", label); break
                elif _is_auth_error(err):
                    logger.error("[%s] Gemini clé invalide", label); break
                else:
                    logger.warning("[%s] Gemini HTTP erreur: %s", label, err[:150]); break

    logger.error("[%s] Tous les providers LLM ont échoué", label)
    return None


def build_llm_cascade_for_agent(
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> list:
    """Retourne la cascade [(name, llm)] pour les agents."""
    cascade = []
    # OpenRouter (premier modèle de la rotation)
    try:
        llm, model_name = _build_openrouter_llm(temperature=temperature, max_tokens=max_tokens)
        short = model_name.split("/")[-1].replace(":free", "")
        cascade.append((f"OpenRouter/{short}", llm))
    except ValueError:
        pass
    # Gemini fallback
    try:
        llm = _build_gemini_llm(temperature=temperature, max_tokens=max_tokens)
        cascade.append(("Gemini", llm))
    except ValueError:
        pass
    return cascade
