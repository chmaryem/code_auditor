"""
lc_learning_agent.py — LangChain LearningAgent.

4 Pillars:
  LLM     : For generalizing a fix → reusable KB rule
  Tools   : tool_write_kb_rule, tool_reload_chromadb, tool_check_rule_exists
  Memory  : PatternMemory (Redis Sorted Set) — tracks pattern frequency
  Planning: Conditional promotion (pattern seen 3+ times → promote to KB)

This agent handles self-improvement: it learns from developer feedback
and promotes recurring patterns into permanent Knowledge Base rules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_agents.memory.redis_memory import AgentRedisMemory, PatternMemory

logger = logging.getLogger(__name__)


class LCLearningAgent:
    """
    LangChain LearningAgent — self-improving knowledge base.

    Anatomy:
      - LLM:      For rule generalization (invoke_with_fallback)
      - Tools:    KB file operations
      - Memory:   PatternMemory (Redis) — frequency tracking
      - Planning: 3+ occurrences → auto-promote; CRITICAL → immediate promote
    """

    PROMOTION_THRESHOLD = 3
    CRITICAL_AUTO_PROMOTE = True

    def __init__(self):
        self.memory = AgentRedisMemory("learning_agent")
        self.pattern_memory = PatternMemory()

    # ── Memory: pattern tracking ─────────────────────────────────────────────

    def record_pattern(self, language: str, pattern: str, severity: str = "MEDIUM") -> int:
        """
        Record a pattern occurrence (Memory pillar).

        Returns the new count for this pattern.
        """
        count = self.pattern_memory.record_pattern(language, pattern)
        logger.debug("Pattern '%s' (%s) seen %d time(s)", pattern, language, count)
        return count

    # ── Planning: promotion decision ─────────────────────────────────────────

    def should_promote(self, language: str, pattern: str, severity: str = "MEDIUM") -> bool:
        """
        Planning pillar — decide if a pattern should be promoted to KB.

        Rules:
          - CRITICAL severity → always promote immediately
          - Other severity → promote if seen >= 3 times
        """
        if self.CRITICAL_AUTO_PROMOTE and severity.upper() == "CRITICAL":
            return True
        count = self.pattern_memory.get_frequency(language, pattern)
        return count >= self.PROMOTION_THRESHOLD

    # ── LLM: rule generalization ─────────────────────────────────────────────

    def generalize_to_rule(
        self,
        pattern: str,
        language: str,
        examples: List[str],
    ) -> Optional[str]:
        """
        LLM pillar — generalize a pattern into a reusable KB rule.

        Uses the cascade LLM to transform concrete fix examples into
        an abstract rule that can be stored in the Knowledge Base.

        Args:
            pattern: Pattern name (e.g. "sql_injection_concatenation").
            language: Programming language.
            examples: List of concrete code examples where this pattern was found.

        Returns:
            Generalized rule text in markdown format, or None if LLM fails.
        """
        from services.llm_factory import invoke_with_fallback

        examples_text = "\n---\n".join(examples[:5])
        prompt = (
            f"You are a code quality expert. Generalize this recurring pattern "
            f"into a reusable rule for {language}.\n\n"
            f"Pattern: {pattern}\n"
            f"Examples found in code:\n{examples_text}\n\n"
            f"Write a concise markdown rule with:\n"
            f"1. Rule name\n"
            f"2. Description (1-2 sentences)\n"
            f"3. Bad pattern (code example)\n"
            f"4. Good pattern (code example)\n"
            f"5. Severity: CRITICAL | HIGH | MEDIUM | LOW\n\n"
            f"Output ONLY the markdown rule. No explanations."
        )
        return invoke_with_fallback(prompt, label=f"learn:{pattern}")

    # ── Tool: KB promotion ───────────────────────────────────────────────────

    @staticmethod
    def _safe_name(pattern: str) -> str:
        return pattern.lower().replace(" ", "_").replace("/", "_")[:50]

    def promote_to_pending(
        self,
        pattern: str,
        language: str,
        rule_content: str,
    ) -> Optional[Path]:
        """
        Stage a learned rule for HUMAN APPROVAL (Tool pillar).

        Writes the rule to data/knowledge_base/{language}/pending/ instead of
        learned/. The developer reviews it in the plugin and explicitly accepts
        (→ moved to learned/, used by RAG) or rejects (→ deleted). Auto-writing to
        learned/ without consent was incorrect: it silently changed the KB the
        auditor relies on.

        Returns the pending file path, or None on failure.
        """
        from config import config

        pending_dir = config.KNOWLEDGE_BASE_DIR / language / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        rule_file = pending_dir / f"{self._safe_name(pattern)}.md"

        try:
            rule_file.write_text(rule_content, encoding="utf-8")
            logger.info("KB rule staged for approval: %s", rule_file)
            return rule_file
        except Exception as e:
            logger.error("Failed to stage KB rule: %s", e)
            return None

    def approve_pending(self, language: str, pattern: str) -> bool:
        """Promote a pending rule to learned/ (developer accepted)."""
        from config import config

        safe = self._safe_name(pattern)
        pending_file = config.KNOWLEDGE_BASE_DIR / language / "pending" / f"{safe}.md"
        if not pending_file.exists():
            logger.warning("approve_pending: no pending rule %s/%s", language, safe)
            return False

        learned_dir = config.KNOWLEDGE_BASE_DIR / language / "learned"
        learned_dir.mkdir(parents=True, exist_ok=True)
        try:
            content = pending_file.read_text(encoding="utf-8")
            (learned_dir / f"{safe}.md").write_text(content, encoding="utf-8")
            pending_file.unlink()
            logger.info("KB rule approved → learned: %s/%s", language, safe)
            return True
        except Exception as e:
            logger.error("approve_pending failed: %s", e)
            return False

    def reject_pending(self, language: str, pattern: str) -> bool:
        """Delete a pending rule (developer rejected)."""
        from config import config

        safe = self._safe_name(pattern)
        pending_file = config.KNOWLEDGE_BASE_DIR / language / "pending" / f"{safe}.md"
        try:
            if pending_file.exists():
                pending_file.unlink()
            logger.info("KB rule rejected: %s/%s", language, safe)
            return True
        except Exception as e:
            logger.error("reject_pending failed: %s", e)
            return False

    # ── Full pipeline: process feedback ──────────────────────────────────────

    def process_feedback(
        self,
        analysis_result: Dict[str, Any],
        language: str,
    ) -> Dict[str, Any]:
        """
        Full learning pipeline — called after each analysis.

        Steps:
          1. Extract patterns from analysis result (Tool)
          2. Record each pattern occurrence (Memory)
          3. Check promotion threshold (Planning)
          4. If threshold met → generalize to rule (LLM) → write to KB (Tool)

        Returns:
            Dict with 'patterns_recorded', 'rules_promoted'.
        """
        patterns_recorded = 0
        pending_rules: List[Dict[str, Any]] = []

        # Extract pattern-like strings from analysis
        analysis_text = analysis_result.get("analysis", "")
        patterns = self._extract_patterns(analysis_text)

        for pattern_name, severity in patterns:
            # Memory: record
            count = self.record_pattern(language, pattern_name, severity)
            patterns_recorded += 1

            # Planning: should we suggest promotion?
            if self.should_promote(language, pattern_name, severity):
                # Skip if already learned or already awaiting approval
                if self._rule_exists(language, pattern_name):
                    continue

                # LLM: generalize
                examples = [analysis_text[:500]]
                rule_content = self.generalize_to_rule(pattern_name, language, examples)

                if rule_content:
                    # Tool: STAGE for human approval (not auto-promote)
                    staged = self.promote_to_pending(pattern_name, language, rule_content)
                    if staged:
                        pending_rules.append({
                            "pattern":   pattern_name,
                            "language":  language,
                            "severity":  severity,
                            "count":     count,
                            "rule_md":   rule_content,
                            "file":      str(staged),
                        })

        return {
            "patterns_recorded": patterns_recorded,
            # Rules now require developer approval before entering the KB.
            "pending_rules":     pending_rules,
            # Back-compat: callers that log "rules_promoted" still get the names.
            "rules_promoted":    [r["pattern"] for r in pending_rules],
        }

    def _extract_patterns(self, analysis_text: str) -> List[tuple]:
        """Extract (pattern_name, severity) tuples from LLM analysis text."""
        import re
        patterns = []
        # Match **PROBLEM**: xxx **SEVERITY**: yyy in fix blocks
        blocks = re.findall(
            r'\*\*PROBLEM\*\*:\s*(.+?)(?:\n|\r).*?\*\*SEVERITY\*\*:\s*(\w+)',
            analysis_text,
            re.DOTALL,
        )
        for problem, severity in blocks:
            name = problem.strip()[:80].replace(" ", "_").lower()
            name = re.sub(r'[^a-z0-9_]', '', name)
            if name:
                patterns.append((name, severity.strip()))
        return patterns

    def _rule_exists(self, language: str, pattern: str) -> bool:
        """True if the rule is already learned OR already awaiting approval."""
        from config import config
        safe_name = self._safe_name(pattern)
        learned = config.KNOWLEDGE_BASE_DIR / language / "learned" / f"{safe_name}.md"
        pending = config.KNOWLEDGE_BASE_DIR / language / "pending" / f"{safe_name}.md"
        return learned.exists() or pending.exists()


# Singleton
lc_learning_agent = LCLearningAgent()
