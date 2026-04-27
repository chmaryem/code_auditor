"""
code_mode_agent.py — Agent MCP Code Mode.

FIX v6 — System prompts corrigés (noms de méthodes exacts) :

  CAUSE DU BUG "GitHubClient has no attribute 'get_pull_request'" :
    Les prompts listaient les tools MCP comme contexte.
    Gemini en déduisait les noms à appeler : get_pull_request, get_pull_request_files…
    Ces noms n'existent pas dans GitHubClient (qui utilise des noms wrappers).

  FIX DOUBLE :
    1. Les system prompts imposent maintenant les noms EXACTS des méthodes wrappers.
       Exemple : "USE github.get_pr_info() NOT github.get_pull_request()"
    2. GitHubClient a des alias pour les noms MCP bruts (code_mode_client.py v6).

  AUTRES FIXES (hérités v5) :
    - Cascade LLM : gemini-2.5-flash → 2.0-flash → 1.5-flash si quota 429
    - Backoff : 15s → 45s entre retries
    - Sandbox instruite d'appeler cache.read_analysis() AVANT rag.analyze()
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Cascade de modèles (OpenAI → Gemini → Groq) — géré via services.llm_factory
LLM_MODEL_CASCADE = [
    "gpt-4o",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "llama-3.3-70b-versatile",
]


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts v6 — Noms de méthodes imposés explicitement
# ─────────────────────────────────────────────────────────────────────────────

PR_REVIEW_SYSTEM_PROMPT = """\
You are a senior code reviewer. Write Python code that reviews a GitHub Pull Request.

## EXACT METHOD NAMES — USE THESE EXACTLY, NO SUBSTITUTION
github.get_pr_info(owner, repo, pr_number)             → dict with head.ref, base.ref
github.get_pr_files(owner, repo, pr_number)            → list of {filename, patch, status}
github.get_file_content(owner, repo, path, ref)        → str (file content)
github.post_review(owner, repo, pr_number, body, event) → posts review
  event must be one of: "APPROVE", "REQUEST_CHANGES", "COMMENT"
cache.read_analysis(filename)                          → str or None
rag.count_severity(text)                               → dict {critical, high, medium, score}
rag.analyze(code, filename, language, patch)           → dict {analysis, critical, high, score}
kg.detect_patterns(code, language)                     → list of strings

DO NOT USE: get_pull_request(), get_pull_request_files(), create_pull_request_review()
USE INSTEAD: get_pr_info(),    get_pr_files(),           post_review()

## WORKFLOW
1. pr_info = github.get_pr_info(owner, repo, pr_number)
   head_ref = pr_info['head']['ref']   ← feature branch (NOT main)
2. files = github.get_pr_files(owner, repo, pr_number)
3. For each file with extension .java .py .js .ts .jsx .tsx:
   content = github.get_file_content(owner, repo, filename, head_ref)
   cached = cache.read_analysis(filename)
   if cached:
       result = rag.count_severity(cached)   # zero LLM tokens
   else:
       result = rag.analyze(content, filename, language, patch)
   patterns = kg.detect_patterns(content, language)
4. total_score = sum of scores; total_critical = sum of criticals
5. event = "REQUEST_CHANGES" if critical>0 or score>=35 else "COMMENT" if score>=15 else "APPROVE"
6. Build body using list + "\\n".join(lines) — NEVER multiline f-strings
7. github.post_review(owner, repo, pr_number, body, event)
8. LAST LINE ONLY: print(json.dumps({"verdict": event, "score": total_score, "critical": total_critical, "high": total_high, "files_analyzed": n}))
"""

CONFLICT_RESOLUTION_SYSTEM_PROMPT = """\
You are a merge conflict resolver. Write Python code for a sandbox.

## EXACT METHOD NAMES — USE THESE EXACTLY
github.get_pr_mergeable_status(owner, repo, pr_number) → dict {has_conflicts, base_ref, head_ref}
github.get_pr_files(owner, repo, pr_number)            → list of {filename, status, patch}
github.get_file_content(owner, repo, path, ref)        → str
github.create_branch(owner, repo, branch_name, from_ref) → dict
github.push_file(owner, repo, path, content, message, branch) → dict
github.post_comment(owner, repo, pr_number, body)      → dict
github.create_pull_request(owner, repo, title, body, head, base) → dict with html_url
resolver.resolve(filename, conflicted, ours, theirs)   → str or None

DO NOT USE: get_pull_request_status(), push_files() — these do not exist

## WORKFLOW
1. status = github.get_pr_mergeable_status(owner, repo, pr_number)
   if not status['has_conflicts']:
     github.post_comment(owner, repo, pr_number, "No conflicts detected.")
     print(json.dumps({"resolved": [], "failed": [], "branch": "", "pr_url": ""}))
     exit()
2. base_ref = status['base_ref']; head_ref = status['head_ref']
3. branch_name = f"auto-resolve/pr-{pr_number}"
   github.create_branch(owner, repo, branch_name, base_ref)
4. files = github.get_pr_files(owner, repo, pr_number)
   resolved_files = []   ← list, NOT int
   failed_files = []
   diffs = []
5. For each code file (.java .py .js .ts .jsx .tsx) with status modified or renamed:
   base_content = github.get_file_content(owner, repo, filename, base_ref)
   head_content = github.get_file_content(owner, repo, filename, head_ref)
   if not base_content or not head_content: failed_files.append(filename); continue
   if base_content == head_content: continue
   conflicted = "<<<<<<< OURS\\n" + base_content + "\\n=======\\n" + head_content + "\\n>>>>>>> THEIRS"
   resolved = resolver.resolve(filename, conflicted, base_content, head_content)
   if resolved:
     github.push_file(owner, repo, filename, resolved, f"auto-resolve: {filename}", branch_name)
     resolved_files.append(filename)
     diffs.append(f"- {filename}: resolved")
   else:
     failed_files.append(filename)
6. Build report comment using list + join. Post it.
7. pr_url = ""
   if resolved_files:
     new_pr = github.create_pull_request(owner, repo, title, body, branch_name, base_ref)
     pr_url = new_pr.get("html_url", "")
8. LAST LINE: print(json.dumps({"resolved": resolved_files, "failed": failed_files, "branch": branch_name, "pr_url": pr_url}))
"""

MERGE_CHECK_SYSTEM_PROMPT = """\
You are a merge readiness checker. Write Python code for a sandbox.

## EXACT METHOD NAMES — USE THESE EXACTLY
github.get_pr_info(owner, repo, pr_number)             → dict {title, head:{sha}}
github.get_pr_mergeable_status(owner, repo, pr_number) → dict {has_conflicts, head_sha}
github.get_check_runs(owner, repo, ref)                → list of check run dicts
github.get_pr_reviews(owner, repo, pr_number)          → list of review dicts
github.post_comment(owner, repo, pr_number, body)      → dict

DO NOT USE: get_pull_request_status() — use get_pr_mergeable_status()

## WORKFLOW
1. pr_info = github.get_pr_info(owner, repo, pr_number)
   pr_title = pr_info.get('title', f'PR #{pr_number}')
   head_sha = pr_info.get('head', {}).get('sha', '')
2. status = github.get_pr_mergeable_status(owner, repo, pr_number)
   has_conflicts = status.get('has_conflicts', False)
   mergeable = not has_conflicts
3. checks = github.get_check_runs(owner, repo, head_sha) if head_sha else []
   failing = [c.get('name','?') for c in checks if c.get('conclusion','') not in ('success','neutral','skipped','')]
   ci_pass = len(failing) == 0
4. reviews = github.get_pr_reviews(owner, repo, pr_number)
   reviews_approved = any(r.get('state') == 'APPROVED' for r in reviews)
   changes_req = any(r.get('state') == 'REQUEST_CHANGES' for r in reviews)
   ready = mergeable and ci_pass and not changes_req
5. Build markdown report using list + join. Post it:
   github.post_comment(owner, repo, pr_number, report)
6. LAST LINE: print(json.dumps({"ready": ready, "mergeable": mergeable, "ci_pass": ci_pass, "reviews_approved": reviews_approved, "details": "..."}))
"""

TEST_GENERATION_SYSTEM_PROMPT = """\
You are a test generation AI. Write Python code for a sandbox.

## EXACT METHOD NAMES
github.post_comment(owner, repo, pr_number, body) → dict

## WORKFLOW
1. Parse analysis_text for [CRITICAL] and [HIGH] findings
2. For each finding, generate one test:
   Java: JUnit 5 @Test method testing that specific bug
   Python: pytest function testing that specific bug
3. Assemble complete test file (class + imports + all tests)
4. Post as comment: github.post_comment(owner, repo, pr_number, body)
5. LAST LINE: print(json.dumps({"test_code": "<full test file>", "test_filename": "<name>", "tests_count": n, "bugs_covered": [...]}))
"""

SELF_CORRECTION_SYSTEM_PROMPT = """\
You are a code fixer AI. A test failed. Analyze and fix it.

## WORKFLOW
1. Read the error_output to understand the failure
2. Decide: CODE is wrong or TEST is wrong?
3. Generate the fix for ONE target (code or test, not both)
4. LAST LINE: print(json.dumps({"fix_target": "code"|"test", "fixed_content": "<content>", "explanation": "<why>"}))
"""


# ─────────────────────────────────────────────────────────────────────────────
# CodeModeAgent
# ─────────────────────────────────────────────────────────────────────────────

class CodeModeAgent:
    """
    Agent MCP Code Mode.

    Pipeline :
      run(task)
        → _discover_mcp_tools()          : 1 appel sandbox léger
        → _invoke_with_cascade(prompt)   : Gemini génère le script Python
        → SandboxExecutor.execute(code)  : subprocess exécute le script
             (dans le script : github.*, rag.*, cache.*, kg.*, resolver.*)
        → Si erreur → Gemini corrige (max 3 tentatives)
    """

    def __init__(
        self,
        system_prompt: str,
        max_iterations: int = 3,
        temperature: float = 0.1,
    ):
        self.system_prompt    = system_prompt
        self.max_iterations   = max_iterations
        self.temperature      = temperature
        self._discovered_tools: Optional[List[str]] = None

    def _get_llm(self, model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv
        load_dotenv(Path(_project_root) / ".env")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=8192,
            temperature=self.temperature,
        )

    def _discover_mcp_tools(self) -> List[str]:
        if self._discovered_tools is not None:
            return self._discovered_tools
        try:
            from services.sandbox_executor import SandboxExecutor
            script = (
                "import json\n"
                "try:\n"
                "    tools = github.get_available_tools()\n"
                "    print(json.dumps({'tools': tools}))\n"
                "except Exception as e:\n"
                "    print(json.dumps({'tools': [], 'error': str(e)}))\n"
            )
            result = SandboxExecutor(timeout=60).execute(script)
            for line in reversed((result.stdout or "").strip().splitlines()):
                if line.strip().startswith("{"):
                    data = json.loads(line.strip())
                    self._discovered_tools = data.get("tools", [])
                    print(f"\n  🔌 Connexion MCP GitHub et discovery des tools...")
                    print(f"\n  🔍 MCP Tools découverts ({len(self._discovered_tools)}) :")
                    for t in sorted(self._discovered_tools):
                        print(f"     • {t}")
                    print()
                    return self._discovered_tools
        except Exception as e:
            logger.debug("Tool discovery: %s", e)
        self._discovered_tools = []
        return []

    def _build_prompt(self, task: str) -> str:
        """
        Construit le prompt complet.
        NE LISTE PLUS les tools MCP dans le prompt (évite que Gemini utilise leurs noms).
        Utilise uniquement les noms définis dans le system_prompt.
        """
        return (
            f"{self.system_prompt}\n\n"
            f"## Pre-imported objects in sandbox\n"
            f"  github, rag, kg, cache, resolver  — from services.code_mode_client\n"
            f"  json, re, Path, datetime — standard library\n\n"
            f"## Task\n{task}\n\n"
            f"Write COMPLETE solution in ONE ```python block.\n"
            f"LAST print() must be ONLY json.dumps({{...}})\n"
            f"Use list + '\\n'.join(lines) for multi-line strings.\n"
        )

    def _invoke_with_cascade(self, messages) -> Optional[str]:
        """
        Cascade OpenAI → Gemini → Groq avec backoff si quota 429.
        Délègue à services.llm_factory.build_llm_cascade_for_agent().
        """
        from services.llm_factory import build_llm_cascade_for_agent, _BACKOFF, _is_quota_error
        cascade = build_llm_cascade_for_agent(temperature=self.temperature, max_tokens=8192)

        if not cascade:
            logger.error("Aucun provider LLM disponible (vérifiez vos clés API)")
            return None

        for provider_name, llm in cascade:
            for attempt in range(2):
                try:
                    print(f"   ⏳ [{provider_name}] Appel LLM...")
                    response = llm.invoke(messages)
                    return response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    err = str(e)
                    if _is_quota_error(err):
                        wait = _BACKOFF[min(attempt, 1)]
                        print(f"   ⏳ [{provider_name}] Quota 429 — {wait}s...")
                        time.sleep(wait)
                        if attempt == 1:
                            print(f"   ✗ [{provider_name}] → provider suivant")
                            break
                    else:
                        logger.debug("[%s] LLM error: %s", provider_name, err[:80])
                        break
        return None

    def run(self, task: str) -> Dict[str, Any]:
        """Pipeline MCP Code Mode complet."""
        from services.sandbox_executor import SandboxExecutor

        self._discover_mcp_tools()
        sandbox  = SandboxExecutor()
        prompt   = self._build_prompt(task)
        messages = [prompt]
        last_code = ""

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n  {'─'*60}")
            print(f"   Agent Code Mode — itération {iteration}/{self.max_iterations}")

            response = self._invoke_with_cascade(messages)
            if response is None:
                return {
                    "success": False, "output": "",
                    "error": "Tous les modèles LLM indisponibles (quota)",
                    "iterations": iteration, "generated_code": last_code,
                }

            code = self._extract_code(response)
            if not code:
                messages.extend([response, "No ```python block. Rewrite with solution in ```python."])
                continue

            last_code = code
            print(f"  Code généré ({len(code)} chars)")

            code = self._autofix_multiline_strings(code)
            syntax_err = self._pre_validate_syntax(code)
            if syntax_err:
                if iteration < self.max_iterations:
                    messages.extend([response, f"SyntaxError: {syntax_err}\nRewrite full script."])
                    continue
                return {"success": False, "output": "", "error": syntax_err,
                        "iterations": iteration, "generated_code": code}

            print(f"   Exécution dans le sandbox...")
            result = sandbox.execute(code)

            if result.success:
                print(f"   ✓ Exécution réussie")
                return {
                    "success": True, "output": result.stdout,
                    "error": "", "iterations": iteration, "generated_code": code,
                }

            err_msg = self._filter_stderr(result.stderr) or result.error or f"Exit {result.return_code}"
            print(f"   Erreur: {err_msg[:200]}")

            if iteration < self.max_iterations:
                messages.extend([response, self._build_fix_prompt(err_msg)])

        return {
            "success": False, "output": "",
            "error": f"Échec après {self.max_iterations} tentatives",
            "iterations": self.max_iterations, "generated_code": last_code,
        }

    def _build_fix_prompt(self, error_msg: str) -> str:
        """Prompt de correction avec rappel des noms exacts de méthodes."""
        attr_err = "has no attribute" in error_msg
        hint = (
            "WRONG METHOD NAME. Use EXACTLY:\n"
            "  github.get_pr_info()            ← NOT get_pull_request()\n"
            "  github.get_pr_files()           ← NOT get_pull_request_files()\n"
            "  github.get_file_content()       ← NOT get_file_contents()\n"
            "  github.post_review()            ← NOT create_pull_request_review()\n"
            "  github.post_comment()           ← NOT add_issue_comment()\n"
            "  github.get_pr_mergeable_status() ← NOT get_pull_request_status()\n"
            "  github.push_file()              ← NOT push_files()\n"
        ) if attr_err else (
            "Fix the error. Remember:\n"
            "  cache.read_analysis(filename) BEFORE rag.analyze()\n"
            "  list + join for multi-line strings\n"
            "  LAST line = ONLY print(json.dumps({...}))\n"
        )
        return f"Error:\n```\n{error_msg[:400]}\n```\n\n{hint}\nRewrite COMPLETE script in ```python."

    def _autofix_multiline_strings(self, code: str) -> str:
        import ast
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass
        fixed = re.sub(r'(f?["\'])([^"\'\\]*)\n', lambda m: m.group(0).rstrip('\n') + '\\n', code)
        try:
            ast.parse(fixed)
            return fixed
        except SyntaxError:
            return code

    def _filter_stderr(self, stderr: str) -> str:
        if not stderr:
            return ""
        benign = ["npm notice", "npm warn", "GitHub MCP Server", "async_generator",
                  "stdio_client", "anyio._backends", "ClosedResourceError", "[MCP]"]
        lines = [l for l in stderr.splitlines() if l.strip() and not any(p in l for p in benign)]
        total = len(stderr.splitlines())
        if total and len(lines) / total < 0.3:
            return ""
        return "\n".join(lines).strip()

    def _pre_validate_syntax(self, code: str) -> Optional[str]:
        import ast
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"SyntaxError at line {e.lineno}: {e.msg}"

    def _extract_code(self, response: str) -> Optional[str]:
        for pattern in [r'```python\s*\n(.*?)```', r'```py\s*\n(.*?)```']:
            m = re.findall(pattern, response, re.DOTALL)
            if m:
                code = max(m, key=len).strip()
                if code:
                    return code
        generic = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
        if len(generic) == 1:
            code = generic[0].strip()
            if any(kw in code for kw in ["import", "print(", "def ", "for ", "if "]):
                return code
        return None