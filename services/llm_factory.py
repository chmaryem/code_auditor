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
    """Construit un LLM Google Gemini."""
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
    """Retourne le LLM primaire (OpenRouter avec rotation)."""
    llm, model_name = _build_openrouter_llm(temperature=temperature, max_tokens=max_tokens)
    return llm


def invoke_with_fallback(
    prompt: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    label: str = "LLM",
) -> Optional[str]:

    from config import config

    # ── Phase 1 : OpenRouter (modèle unique avec backoff) ───────────────────
    api_key = config.api.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if api_key:
        model = config.api.openrouter_model
        short_name = model.split("/")[-1].replace(":free", "")
        
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model           = model,
                api_key         = api_key,
                base_url        = "https://openrouter.ai/api/v1",
                temperature     = temperature,
                max_tokens      = max_tokens,
                default_headers = {
                    "HTTP-Referer": "https://github.com/code-auditor",
                    "X-Title":      "Code Auditor",
                },
            )
            
            for attempt in range(4):
                try:
                    print(f"    [{short_name}] {label} — appel LLM (tentative {attempt + 1}/4)...")
                    response = llm.invoke(prompt)
                    text = response.content if hasattr(response, "content") else str(response)
                    logger.info("[%s] réponse reçue via %s", label, short_name)
                    return text
                except Exception as e:
                    err = str(e)
                    if _is_quota_error(err):
                        if attempt < 3:
                            wait = _BACKOFF[attempt]
                            print(f"   ⚠️  [{short_name}] erreur/quota — attente {wait}s...")
                            time.sleep(wait)
                        else:
                            logger.error("[%s] %s épuisé après 4 tentatives", label, short_name)
                            break
                    elif _is_auth_error(err):
                        logger.error("[%s] %s clé API invalide", label, short_name)
                        break
                    else:
                        logger.error("[%s] %s erreur inattendue: %s", label, short_name, err[:120])
                        break
        except Exception as e:
            logger.debug("[%s] OpenRouter build failed: %s", label, e)

    # ── Phase 2 : Gemini (fallback) ─────────────────────────────────────────
    try:
        llm = _build_gemini_llm(temperature=temperature, max_tokens=max_tokens)
    except ValueError as missing:
        logger.debug("[%s] Gemini ignoré : %s", label, missing)
        logger.error("[%s] Tous les providers LLM ont échoué", label)
        return None

    for attempt in range(4):
        try:
            print(f"    [Gemini] {label} — appel LLM (tentative {attempt + 1}/4)...")
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            logger.info("[%s] réponse reçue via Gemini", label)
            return text

        except Exception as e:
            err = str(e)
            if _is_quota_error(err):
                if attempt < 3:
                    wait = _BACKOFF[attempt]
                    print(f"   ⚠️  [Gemini] quota 429 — attente {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error("[%s] Gemini quota épuisé après 4 tentatives", label)
                    break
            elif _is_auth_error(err):
                logger.error("[%s] Gemini clé API invalide", label)
                break
            else:
                logger.error("[%s] Gemini erreur: %s", label, err[:120])
                break

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
