"""
lc_chat_decision_agent.py — LLM-powered Decision Agent for ChatGraph.

Architecture v2 :
  - Primary : Gemini Flash / GPT-4o-mini LLM call (~50 tokens, <0.5s)
  - Fallback : deterministic regex (original v1 logic, kept as safety net)

The LLM call returns a structured JSON plan with:
  - intent       : explain | complete_fn | new_class | git_question |
                   ci_question | test_generation | question | code_generation
  - target_file  : file hint from message
  - target_symbol: function/class name
  - context_level: fast | context | deep
  - needs_git    : bool
  - needs_ci     : bool
  - needs_generation: bool
  - confidence   : 0.0-1.0
  - reason       : one-line explanation

Improvements over v1:
  - Understands ambiguous messages that fool keyword matching
  - Uses cursor_line + active_function + selected_text for precision
  - Uses last 3 history turns for follow-up detection
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Salutations / small-talk : détectées en amont pour éviter le template projet.
_GREETING_RE = re.compile(
    r"^\s*(bonjour|salut|coucou|hello|hi|hey|yo|bonsoir|merci|thanks?|thx|"
    r"ok|okay|d'accord|au revoir|bye|ciao|à bientôt|good\s?(morning|evening))\b",
    re.IGNORECASE,
)

# ── LLM Decision Prompt ───────────────────────────────────────────────────────

_DECISION_PROMPT = """\
You are a routing agent for Code Auditor AI, an expert developer assistant embedded in an IDE.
Your ONLY job is to classify the developer's message and return a JSON routing plan.

Developer message: {message}

Active file: {target_file}
Active function (cursor): {active_function}
Selected text: {selected_text}
Cursor line: {cursor_line}

Last 3 conversation turns:
{history_snippet}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "intent": "<one of: explain|complete_fn|new_class|code_generation|git_question|ci_question|project_state|question>",
  "target_file": "<filename or empty string>",
  "target_symbol": "<function or class name, or empty string>",
  "generation_target": "<name to generate, or empty string>",
  "context_level": "<fast|context|deep>",
  "needs_git": <true|false>,
  "needs_ci": <true|false>,
  "needs_rag": <true|false>,
  "needs_generation": <true|false>,
  "confidence": <0.0 to 1.0>,
  "reason": "<one sentence>"
}}

Intent guide:
- explain        : user asks what code does, how it works, summarize a file or function
- complete_fn    : user asks to complete/implement an existing function stub
- new_class      : user asks to create a new class from scratch
- code_generation: user EXPLICITLY asks to write/generate/create NEW code, implement a feature,
                   or produce a code snippet (e.g. "write a function to...", "generate X", "create Y")
                   NEVER use for analysis/review tasks even if file context is needed
- git_question   : about commits, branches, merges, PRs, conflicts, diffs, what changed
- ci_question    : about CI/CD pipeline, builds, deployments, GitHub Actions, SonarCloud, quality gate
- project_state  : holistic project health — "can I deploy?", "is my project ready?",
                   "what should I fix next?", "show me all issues", combining git + CI + security
- question       : general Q&A, analysis, code review — includes "find bugs", "check for issues",
                   "security review", "analyze", "what's wrong with", "review this code",
                   "find vulnerabilities", "detect problems", "audit", and any explain/analysis
                   request that is NOT about writing new code

context_level guide:
- fast    : simple explain of 1 file, no external context needed
- context : needs file content + dependencies + RAG knowledge base
- deep    : multi-file reasoning, or needs git + CI data

RAG usage criteria — set needs_rag=true ONLY when:
- intent is explain, code_generation, complete_fn, new_class, or question about code structure
- AND a target_file is present or the question references specific code
- Set needs_rag=false for: git_question, ci_question, project_state, simple general questions,
  and any fast-level explain (context_level=fast handles those with file code alone)
"""


def _build_history_snippet(history: List[Dict[str, Any]]) -> str:
    """Last 3 turns formatted for the LLM prompt."""
    if not history:
        return "(no previous conversation)"
    lines = []
    for turn in history[-3:]:
        role = turn.get("role", "?")
        content = (turn.get("content", "") or "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _call_decision_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """Call a fast LLM (Gemini Flash preferred) to get routing JSON."""
    try:
        from services.llm_factory import invoke_with_fallback
        raw = invoke_with_fallback(prompt, label="chat_decision", max_tokens=256)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        return json.loads(raw)
    except Exception as e:
        logger.debug("LLM decision failed: %s", e)
        return None


class LCChatDecisionAgent:
    """
    LLM-powered Decision Agent (v2).

    Primary path  : LLM call → structured JSON plan
    Fallback path : regex/keyword deterministic routing (v1)
    """

    SUPPORTED_LANGS = ["python", "java", "javascript", "typescript"]

    def decide(
        self,
        user_message: str,
        target_file: str = "",
        base_intent: str = "question",
        intent_params: Dict[str, Any] | None = None,
        conversation_history: List[Dict[str, Any]] | None = None,
        # ── New cursor context ──────────────────────────────────
        cursor_line: int = 0,
        active_function: str = "",
        selected_text: str = "",
    ) -> Dict[str, Any]:
        msg_raw = user_message or ""
        params  = intent_params or {}
        history = conversation_history or []

        # ── Small-talk / salutations : court-circuit avant tout appel LLM ─────
        # Évite le template projet « Direct answer → Evidence from codebase » sur
        # un simple « bonjour ». Réponse courte et directe, aucun contexte projet.
        if _GREETING_RE.match(msg_raw.strip()) and len(msg_raw.split()) <= 5:
            return {
                "intent":        "chitchat",
                "agents":        ["chat_agent"],
                "context_level": "fast",
                "needs_rag":     False,
                "needs_git":     False,
                "needs_ci":      False,
                "confidence":    1.0,
                "_routing":      "regex",
                "reason":        "salutation / small-talk",
            }

        from config import config as _cfg

        # ── Phase 1.3 · Cache décision (1er tour uniquement) ─────────────────
        # On ne met en cache QUE lorsque history est vide : le routage d'un
        # follow-up dépend de l'historique (voir _decide_regex), donc le mettre
        # en cache par simple texte serait incorrect. Un 1er message est, lui,
        # auto-suffisant (routage fonction de message+fichier+intent seulement).
        cache_ok = (not history) and getattr(_cfg.chat, "decision_cache_enabled", False)
        ckey = self._decision_cache_key(msg_raw, target_file, base_intent) if cache_ok else ""
        if ckey:
            cached = self._decision_cache_get(ckey)
            if cached is not None:
                # Hit = AUCUN appel LLM ce tour → _routing="cache" pour que la
                # télémétrie (node_decision_agent) compte 0 (et non 1).
                cached["_routing"] = "cache"
                return self._enrich_cursor(cached, target_file, active_function, selected_text)

        # ── Phase 1.1 · Router regex-first (économie de tokens) ──────────────
        # Par défaut (config.chat.router_regex_first=True) : on classe d'abord par
        # règles déterministes (0 token). Si une règle FORTE matche (confiance ≥
        # seuil : git/ci/génération/explain/project_state), on s'en tient là et on
        # NE consulte PAS le LLM. Sinon (message ambigu), on escalade vers le LLM
        # pour préserver la précision de l'ancien routage.
        regex_first = getattr(_cfg.chat, "router_regex_first", False)
        min_conf    = getattr(_cfg.chat, "router_regex_min_confidence", 0.85)

        if regex_first:
            rplan = self._decide_regex(msg_raw, target_file, base_intent, params, history)
            if rplan.get("confidence", 0.0) >= min_conf:
                rplan["_routing"] = "regex"
                plan = rplan
            else:
                # Règle faible → on tente le LLM, avec repli sur le plan regex.
                llm_plan = self._decide_llm(
                    msg_raw, target_file, base_intent, params, history,
                    cursor_line, active_function, selected_text,
                )
                if llm_plan and llm_plan.get("confidence", 0.0) >= 0.5:
                    llm_plan["_routing"] = "llm"
                    plan = llm_plan
                else:
                    rplan["_routing"] = "regex"
                    plan = rplan
        else:
            # ── Legacy : LLM d'abord, regex en repli ─────────────────────────
            plan = self._decide_llm(
                msg_raw, target_file, base_intent, params, history,
                cursor_line, active_function, selected_text,
            )
            if plan is None or plan.get("confidence", 0.0) < 0.5:
                logger.debug("Decision LLM skipped — using regex fallback")
                plan = self._decide_regex(msg_raw, target_file, base_intent, params, history)
                plan["_routing"] = "regex"
            else:
                plan["_routing"] = "llm"

        if ckey:
            self._decision_cache_set(ckey, plan, getattr(_cfg.chat, "decision_cache_ttl", 900))

        return self._enrich_cursor(plan, target_file, active_function, selected_text)

    @staticmethod
    def _enrich_cursor(
        plan: Dict[str, Any],
        target_file: str,
        active_function: str,
        selected_text: str,
    ) -> Dict[str, Any]:
        """Complète le plan avec le contexte curseur (symbole actif, fichier sélectionné)."""
        if active_function and not plan.get("target_symbol"):
            plan["target_symbol"] = active_function
        if selected_text and not plan.get("target_file"):
            # Si l'utilisateur a sélectionné du code, on travaille sur le fichier courant.
            plan["target_file"] = target_file or plan.get("target_file", "")
        return plan

    # ── Phase 1.3 · Cache décision (Redis via MCP) ───────────────────────────

    @staticmethod
    def _decision_cache_key(message: str, target_file: str, base_intent: str) -> str:
        """Clé stable = hash(message normalisé | fichier | intent de base)."""
        import hashlib
        raw = f"{(message or '').strip().lower()}|{target_file or ''}|{base_intent or ''}"
        return "ca:chat:decision:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _decision_cache_get(key: str) -> Optional[Dict[str, Any]]:
        """Lit un plan mis en cache. Encodé en base64 pour éviter le gotcha du
        client MCP-Redis (les valeurs commençant par { ou [ sont ignorées)."""
        try:
            import base64
            from services.mcp_redis_service import get_mcp_redis
            raw = get_mcp_redis().get(key)
            if not raw:
                return None
            decoded = base64.b64decode(raw).decode("utf-8")
            plan = json.loads(decoded)
            return plan if isinstance(plan, dict) else None
        except Exception as e:
            # Erreur MCP-Redis renvoyée sous forme de chaîne → base64/json échoue → miss.
            logger.debug("decision cache get miss: %s", e)
            return None

    @staticmethod
    def _decision_cache_set(key: str, plan: Dict[str, Any], ttl: int) -> None:
        try:
            import base64
            from services.mcp_redis_service import get_mcp_redis
            payload = base64.b64encode(
                json.dumps(plan, default=str).encode("utf-8")
            ).decode("ascii")
            get_mcp_redis().set(key, payload, expire_seconds=ttl)
        except Exception as e:
            logger.debug("decision cache set failed: %s", e)

    # ── LLM routing ──────────────────────────────────────────────────────────

    def _decide_llm(
        self,
        message: str,
        target_file: str,
        base_intent: str,
        params: Dict[str, Any],
        history: List[Dict[str, Any]],
        cursor_line: int,
        active_function: str,
        selected_text: str,
    ) -> Optional[Dict[str, Any]]:
        # Skip LLM for pure Phase 2 intents — already classified upstream
        if base_intent in ("complete_fn", "new_class"):
            return None

        prompt = _DECISION_PROMPT.format(
            message         = message[:800],
            target_file     = target_file or "(none)",
            active_function = active_function or "(none)",
            selected_text   = (selected_text[:200] + "...") if len(selected_text) > 200 else selected_text or "(none)",
            cursor_line     = cursor_line or "(unknown)",
            history_snippet = _build_history_snippet(history),
        )

        raw_plan = _call_decision_llm(prompt)
        if not raw_plan or "intent" not in raw_plan:
            return None

        # Normalize to internal plan format
        intent = raw_plan.get("intent", "question")
        return {
            "intent":             intent,
            "target_file":        raw_plan.get("target_file", target_file) or target_file,
            "target_symbol":      raw_plan.get("target_symbol", "") or "",
            "generation_target":  raw_plan.get("generation_target", "") or "",
            "agents":             self._agents_for_intent(intent),
            "context_level":      raw_plan.get("context_level", "context"),
            "needs_file":         intent not in ("git_question", "ci_question"),
            "needs_project_summary": True,
            "needs_rag":          bool(raw_plan.get("needs_rag", True)),
            "needs_git":          bool(raw_plan.get("needs_git", False)),
            "needs_ci":           bool(raw_plan.get("needs_ci", False)),
            "needs_generation":   bool(raw_plan.get("needs_generation", False)),
            "needs_tests":        bool(raw_plan.get("needs_tests", False)),
            "needs_validation":   bool(raw_plan.get("needs_generation", False)),
            "safe_mode":          True,
            "confidence":         float(raw_plan.get("confidence", 0.8)),
            "reason":             raw_plan.get("reason", "LLM routing"),
        }

    @staticmethod
    def _agents_for_intent(intent: str) -> List[str]:
        return {
            "explain":          ["code_agent", "chat_agent"],
            "complete_fn":      ["code_generation_agent", "validator_agent"],
            "new_class":        ["code_generation_agent", "validator_agent"],
            "code_generation":  ["code_generation_agent", "validator_agent"],
            "git_question":     ["git_agent", "analysis_agent", "chat_agent"],
            "ci_question":      ["ci_agent", "retriever_agent", "chat_agent"],
            "test_generation":  ["test_agent", "retriever_agent", "validator_agent"],
            "project_state":    ["project_state_agent", "chat_agent"],
            "question":         ["retriever_agent", "chat_agent"],
        }.get(intent, ["retriever_agent", "chat_agent"])

    # ── Regex fallback (original v1 logic — kept as safety net) ─────────────

    def _decide_regex(
        self,
        msg_raw: str,
        target_file: str,
        base_intent: str,
        params: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        msg = msg_raw.lower()

        recent_files  = self._extract_recent_files(history)
        recent_intent = self._last_intent(history)

        resolved_target = target_file or (recent_files[-1] if recent_files else "")

        plan: Dict[str, Any] = {
            "intent":             base_intent or "question",
            "target_file":        resolved_target or params.get("file_hint", ""),
            "target_symbol":      params.get("method_hint", ""),
            "generation_target":  params.get("generation_target", ""),
            "agents":             ["retriever_agent", "chat_agent"],
            "context_level":      "context",
            "needs_file":         True,
            "needs_project_summary": True,
            # RAG off by default in regex fallback — enabled per-intent below
            "needs_rag":          False,
            "needs_git":          False,
            "needs_ci":           False,
            "needs_generation":   False,
            "needs_tests":        False,
            "needs_validation":   False,
            "safe_mode":          True,
            "confidence":         0.6,
            "reason":             "regex fallback",
        }

        if self._contains_word(msg, [
            "can i deploy", "can i merge safely", "is my project ready", "project ready",
            "next actions", "what should i fix", "what should i do next",
            "project state", "project health", "état du projet", "puis-je déployer",
            "quoi faire ensuite", "prochaines actions",
        ]):
            plan.update({"intent": "project_state", "agents": ["project_state_agent", "chat_agent"],
                         "context_level": "deep", "needs_file": False, "confidence": 0.9,
                         "reason": "holistic project-state keyword"})
            return plan

        if self._contains_word(msg, [
            "ci/cd", "pipeline", "github actions", "github action", "workflow",
            "build failed", "test failed", "sonar", "quality gate",
            "deploy", "deployment", "rollback", "release", "staging", "production",
        ]):
            plan.update({
                "intent": "ci_question", "agents": ["ci_agent", "chat_agent"],
                "context_level": "deep", "needs_file": False,
                "needs_ci": True, "needs_rag": False, "confidence": 0.9,
                "reason": "CI/CD keyword",
            })
            return plan

        if self._contains_word(msg, [
            "commit", "merge", "branch", "pull request", "pr", "conflict",
            "rebase", "stash", "diff", "safe to merge", "can i merge",
            "est-ce que je peux merge", "est-ce que je peux commit", "résume mes changements",
            "what changed", "qu'est-ce que j'ai changé",
        ]):
            plan.update({
                "intent": "git_question", "agents": ["git_agent", "chat_agent"],
                "context_level": "deep", "needs_file": False,
                "needs_git": True, "needs_rag": False, "confidence": 0.9,
                "reason": "Git keyword",
            })
            return plan

        if base_intent in ("complete_fn", "new_class"):
            target = (params.get("generation_target") or params.get("method_hint")
                      or self._extract_generation_target(msg_raw))
            plan.update({
                "intent": base_intent, "target_symbol": target, "generation_target": target,
                "agents": ["code_generation_agent", "validator_agent"],
                "needs_file": base_intent == "complete_fn",
                "needs_generation": True, "needs_validation": True,
                "needs_rag": True, "confidence": 0.9,
                "reason": f"Phase 2: {base_intent}",
            })
            return plan

        if self._contains_word(msg, [
            "complete", "complète", "implement", "implémente", "write the body",
            "fill in", "create class", "generate class", "crée une classe",
            "génère une classe", "new class",
        ]):
            target = (params.get("generation_target") or params.get("method_hint")
                      or self._extract_generation_target(msg_raw))
            plan.update({
                "intent": "code_generation", "target_symbol": target, "generation_target": target,
                "agents": ["code_generation_agent", "validator_agent"],
                "needs_generation": True, "needs_validation": True,
                "needs_rag": True, "confidence": 0.9,
                "reason": "code generation keyword",
            })
            return plan

        if (recent_intent in ("explain_code", "contextual_code_question", "explain")
                and self._contains_word(msg, ["dependencies", "dépendances", "risks", "risques",
                                              "impact", "details", "more", "plus", "pourquoi"])):
            plan.update({
                "intent": "question",
                "agents": ["code_agent", "retriever_agent", "chat_agent"],
                "needs_rag": bool(resolved_target),
                "reason": "follow-up from history",
            })
            return plan

        if base_intent == "explain" or self._contains_word(msg, [
            "explain", "explique", "what does", "que fait", "résume", "resume",
            "describe", "décrire", "comment fonctionne", "understand",
        ]):
            # Fast path: file code is enough, RAG not needed
            plan.update({
                "intent": "explain", "agents": ["code_agent", "chat_agent"],
                "context_level": "fast", "needs_project_summary": False,
                "needs_rag": False, "confidence": 0.9,
                "reason": "explain keyword — fast path",
            })
            return plan

        if self._contains_word(msg, [
            "risk", "risque", "impact", "impacted", "depend", "dépend",
            "used by", "where is used", "refactor", "architecture", "coupling",
        ]):
            plan.update({
                "intent": "question",
                "agents": ["code_agent", "retriever_agent", "chat_agent"],
                "needs_rag": bool(resolved_target),
                "reason": "risk/impact keyword",
            })
            return plan

        # Default general question — RAG only if a file is in context
        plan.update({"needs_rag": bool(resolved_target)})
        return plan

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _contains_word(text: str, keywords: list[str]) -> bool:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _extract_generation_target(text: str) -> str:
        raw = text or ""
        patterns = [
            r"(?:complete|complète|finish|implement|implémente|remplis|fill in|écris|develop|développe)\s+(?:the\s+)?(?:function\s+|method\s+|méthode\s+)?`?([\w]+)`?",
            r"(?:create|generate|crée|génère|générer|build|make)\s+(?:a\s+|une\s+)?(?:class\s+|classe\s+)?`?([A-Z][\w]*)`?",
            r"(?:create|generate|crée|génère|générer|build|make)\s+(?:a\s+|une\s+)?`?([A-Z][\w]*)`?\s+(?:class|classe)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _extract_recent_files(history: list[Dict[str, Any]]) -> list[str]:
        files = []
        for turn in history[-8:]:
            meta = turn.get("metadata", {}) or {}
            if meta.get("target_file"):
                files.append(meta["target_file"])
        return files

    @staticmethod
    def _last_intent(history: list[Dict[str, Any]]) -> str:
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                meta = turn.get("metadata", {}) or {}
                if meta.get("intent"):
                    return meta["intent"]
        return ""


chat_decision_agent = LCChatDecisionAgent()
