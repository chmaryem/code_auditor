from __future__ import annotations

import logging
import os
import time
import threading
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


_BACKOFF = [10, 30, 60, 120]

_QUOTA_ERRORS = ("429", "RESOURCE_EXHAUSTED", "quota", "RateLimitError", "rate_limit", "503", "524")
_AUTH_ERRORS  = ("401", "403", "AuthenticationError", "invalid_api_key", "API_KEY_INVALID")

def _is_quota_error(err: str) -> bool:
    return any(k in err for k in _QUOTA_ERRORS)

def _is_auth_error(err: str) -> bool:
    return any(k in err for k in _AUTH_ERRORS)

# Cooldown OpenRouter — si rate-limited, on skip pendant 60s pour aller direct sur Groq
_openrouter_state: dict = {"cooling_until": 0.0}



def _call_openrouter_http(prompt: str, api_key: str, model: str, max_tokens: int = 2048) -> str:
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Code Auditor",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(15, 180),
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()
    choices = data.get("choices", [])

    if not choices:
        raise ValueError(f"OpenRouter: réponse vide — {data}")

    return choices[0]["message"]["content"]


def _call_groq_http(prompt: str, api_key: str, model: str, max_tokens: int = 2048) -> str:
    """Groq — API compatible OpenAI, free et très rapide.
    Sert de fallback entre OpenRouter et Gemini.
    """
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(15, 180),
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Groq HTTP {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()
    choices = data.get("choices", [])

    if not choices:
        raise ValueError(f"Groq: réponse vide — {data}")

    return choices[0]["message"]["content"]


def _call_gemini_http(prompt: str, api_key: str, model: str = "gemini-2.5-flash", max_tokens: int = 2048) -> str:
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.0,
        },
    }

    response = requests.post(
        url,
        json=payload,
        timeout=(15, 180),
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Gemini HTTP {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()
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
        max_retries     = 0,     # fail-fast : pas de backoff interne SDK → bascule immédiate sur le fallback (Groq)
        timeout         = 30,    # coupe les appels qui traînent au lieu d'attendre 18s+30s
        default_headers = {
            "HTTP-Referer": "https://github.com/code-auditor",
            "X-Title":      "Code Auditor",
        },
    ), resolved_model


def _build_groq_llm(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
):
    """Construit un LLM via Groq (API compatible OpenAI). Fallback free et rapide."""
    from langchain_openai import ChatOpenAI
    from config import config

    api_key = config.api.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY non défini")

    resolved_model = model or config.api.groq_model

    return ChatOpenAI(
        model       = resolved_model,
        api_key     = api_key,
        base_url    = "https://api.groq.com/openai/v1",
        temperature = temperature,
        max_tokens  = max_tokens,
        max_retries = 0,     # fail-fast : bascule immédiate sur le fallback suivant (Gemini)
        timeout     = 30,
    )


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
        if time.time() < _openrouter_state["cooling_until"]:
            remaining = round(_openrouter_state["cooling_until"] - time.time())
            logger.debug("[%s] OpenRouter cooldown actif (%ds restants) → Groq direct", label, remaining)
        else:
            model      = config.api.openrouter_model or "mistralai/mistral-7b-instruct:free"
            short_name = model.split("/")[-1].replace(":free", "")
            # 1 seule tentative : les modèles :free sont throttlés en amont (pool partagé) ;
            # inutile d'attendre 100s en backoff — on bascule tout de suite sur Groq.
            try:
                print(f"    [{short_name}] {label} — appel HTTP...")
                text = _call_openrouter_http(prompt_text, api_key, model, max_tokens)
                logger.info("[%s] réponse via OpenRouter/%s (HTTP)", label, short_name)
                return text
            except Exception as e:
                err = str(e)
                if _is_quota_error(err):
                    _openrouter_state["cooling_until"] = time.time() + 60
                    logger.info("[%s] OpenRouter rate-limited → cooldown 60s → Groq", label)
                elif _is_auth_error(err):
                    logger.error("[%s] OpenRouter clé invalide", label)
                else:
                    logger.warning("[%s] OpenRouter HTTP erreur: %s", label, err[:150])

    # ── 2. Groq via HTTP (fallback free, très rapide) ────────────────────────
    api_key = config.api.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if api_key:
        model = config.api.groq_model or "llama-3.3-70b-versatile"
        for attempt in range(4):
            try:
                print(f"    [Groq/{model}] {label} — appel HTTP (tentative {attempt + 1}/4)...")
                text = _call_groq_http(prompt_text, api_key, model, min(max_tokens, 8192))
                logger.info("[%s] réponse via Groq/%s (HTTP)", label, model)
                return text
            except Exception as e:
                err = str(e)
                if _is_quota_error(err):
                    if attempt < 3:
                        wait = _BACKOFF[attempt]
                        print(f"   ⚠️  [Groq] quota — attente {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error("[%s] Groq épuisé après 4 tentatives", label); break
                elif _is_auth_error(err):
                    logger.error("[%s] Groq clé invalide", label); break
                else:
                    logger.warning("[%s] Groq HTTP erreur: %s", label, err[:150]); break

    # ── 3. Gemini via HTTP (zéro PyTorch) ────────────────────────────────────
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
    from config import config

    cascade = []
    # OpenRouter (premier modèle de la rotation)
    try:
        llm, model_name = _build_openrouter_llm(temperature=temperature, max_tokens=max_tokens)
        short = model_name.split("/")[-1].replace(":free", "")
        cascade.append((f"OpenRouter/{short}", llm))
    except ValueError:
        pass
    # Groq fallback (free, rapide)
    try:
        llm = _build_groq_llm(temperature=temperature, max_tokens=max_tokens)
        cascade.append((f"Groq/{config.api.groq_model}", llm))
    except ValueError:
        pass
    # Gemini fallback
    try:
        llm = _build_gemini_llm(temperature=temperature, max_tokens=max_tokens)
        cascade.append(("Gemini", llm))
    except ValueError:
        pass
    return cascade


# ── P2 · Routage par complexité ─────────────────────────────────────────────

# Budget de tokens en SORTIE par niveau : une question "fast" n'a pas besoin
# d'une réponse de 8k tokens → on économise sans dégrader la qualité.
_MAX_TOKENS_BY_LEVEL = {"fast": 1024, "context": 4096, "deep": 8192}


def build_llm_for_level(
    level: str = "context",
    temperature: float = 0.0,
    max_tokens: int | None = None,
):
    """Construit un LLM dont le modèle PRIMAIRE dépend du niveau de complexité
    (fast | context | deep) décidé par le Decision Agent.

    Les deux autres providers restent ajoutés en fallback → la fiabilité est
    identique à l'ancien comportement, seul le modèle qui répond en premier change.

    Le mapping niveau → (provider, model) est centralisé dans
    config.api.model_for_level() : c'est le seul endroit à éditer pour changer
    de modèles (ou via .env : FAST_PROVIDER/FAST_MODEL/CONTEXT_*/DEEP_*).
    """
    from config import config

    level = level if level in ("fast", "context", "deep") else "context"
    provider, model = config.api.model_for_level(level)
    if max_tokens is None:
        max_tokens = _MAX_TOKENS_BY_LEVEL.get(level, 4096)

    def _build(p: str, m: str):
        if p == "groq":
            return _build_groq_llm(model=m or None, temperature=temperature, max_tokens=max_tokens)
        if p == "gemini":
            return _build_gemini_llm(model=m or None, temperature=temperature, max_tokens=max_tokens)
        # défaut : openrouter (renvoie un tuple (llm, model_name))
        return _build_openrouter_llm(model=m or None, temperature=temperature, max_tokens=max_tokens)[0]

    # Primaire = provider du niveau ; les autres deviennent fallback (sans forcer
    # de modèle particulier → ils utilisent leur modèle par défaut).
    order = [provider] + [p for p in ("groq", "openrouter", "gemini") if p != provider]
    llms = []
    for p in order:
        try:
            llms.append(_build(p, model if p == provider else ""))
        except Exception as e:
            logger.debug("build_llm_for_level(%s): provider %s indisponible: %s", level, p, e)

    if not llms:
        logger.warning("build_llm_for_level(%s): aucun provider LLM disponible", level)
        return None

    primary = llms[0]
    return primary.with_fallbacks(llms[1:]) if len(llms) > 1 else primary
