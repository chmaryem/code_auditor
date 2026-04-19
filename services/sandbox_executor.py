"""
sandbox_executor.py — Exécute le code généré par l'agent dans un sandbox.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
SANDBOX_TIMEOUT = 300

ALLOWED_IMPORTS = {
    "json", "re", "os", "os.path", "pathlib", "datetime",
    "collections", "itertools", "functools", "math",
    "code_mode_client", "services.code_mode_client",
}


@dataclass
class SandboxResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error: str = ""


class SandboxExecutor:

    def __init__(self, timeout: int = SANDBOX_TIMEOUT):
        self.timeout = timeout
        self._python_path = sys.executable

    def execute(self, code: str) -> SandboxResult:
        validation_error = self._validate_code(code)
        if validation_error:
            return SandboxResult(
                success=False,
                error=f"Code validation failed: {validation_error}",
                stderr=validation_error,
                return_code=1,
            )

        full_script = self._build_sandbox_script(code)

        sandbox_dir = _project_root / ".codeaudit" / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        script_file = sandbox_dir / "_sandbox_run.py"
        try:
            script_file.write_text(full_script, encoding="utf-8")
        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"Failed to write sandbox script: {e}",
                return_code=1,
            )

        result = None
        try:
            result = subprocess.run(
                [self._python_path, str(script_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(_project_root),
                env={
                    **dict(__import__("os").environ),
                    "PYTHONPATH": str(_project_root),
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                },
            )
            return SandboxResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"Execution timed out ({self.timeout}s)",
                stderr=f"TIMEOUT: Code execution exceeded {self.timeout} seconds",
                return_code=124,
            )
        except Exception as e:
            return SandboxResult(
                success=False, error=str(e), stderr=str(e), return_code=1,
            )
        finally:

            pass

    def _build_sandbox_script(self, agent_code: str) -> str:
        header = (
            "# -- Sandbox Runner -- Auto-generated --\n"
            "import sys\n"
            "import os\n"
            f'sys.path.insert(0, r"{_project_root}")\n'
            "\n"
            "try:\n"
            "    from dotenv import load_dotenv\n"
            f'    load_dotenv(os.path.join(r"{_project_root}", ".env"))\n'
            "except ImportError:\n"
            "    pass\n"
            "\n"
            "import json\n"
            "import re\n"
            "from pathlib import Path\n"
            "from datetime import datetime\n"
            "\n"
            "from services.code_mode_client import github, rag, kg, cache, resolver\n"
            "\n"
            "try:\n"
        )
        indented_code = textwrap.indent(agent_code, "    ")
        footer = (
            "\nexcept Exception as _sandbox_err:\n"
            "    import traceback\n"
            '    print(f"SANDBOX_ERROR: {_sandbox_err}", file=sys.stderr)\n'
            "    traceback.print_exc(file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "finally:\n"
            "    try:\n"
            "        github.disconnect()\n"
            "    except Exception:\n"
            "        pass\n"
        )
        return header + indented_code + footer

    def _validate_code(self, code: str) -> Optional[str]:
        blocked_patterns = [
            ("subprocess", "Direct subprocess access is blocked."),
            ("socket", "Direct socket access is blocked."),
            ("http.client", "Direct HTTP is blocked."),
            ("urllib", "Direct urllib is blocked."),
            ("os.system", "os.system is blocked."),
            ("os.popen", "os.popen is blocked."),
            ("shutil.rmtree", "Destructive filesystem operations are blocked."),
        ]
        for pattern, message in blocked_patterns:
            if pattern in code:
                return message
        return None


SANDBOX_INSTRUCTIONS = """
## Sandbox Environment

You write Python code that runs in a subprocess. The code MUST be syntactically valid Python.

### Pre-imported for you:
- json, re, Path (from pathlib), datetime
- github, rag, kg, cache, resolver (from services.code_mode_client)

### ══ RÈGLE ABSOLUE — TOOL DISCOVERY ══
The MCP server may NOT have the tool names you expect.
ALWAYS use the github.* wrapper methods — they handle tool name mapping:

```python
# CORRECT — wrapper handles everything:
files = github.get_pr_files(owner, repo, pr_number)   # multi-strategy fallback
pr    = github.get_pr_info(owner, repo, pr_number)
cont  = github.get_file_content(owner, repo, path, ref)

# WRONG — never call internal MCP methods directly:
# svc._call_tool("list_pull_request_files", ...)  ← may not exist!
# _loop_manager.run(svc.get_pr_files(...))         ← bypass protections!
```

### ══ RÈGLE STRING MULTI-LIGNES ══
FORBIDDEN:
```python
body = f"line1
line2"  # SyntaxError!
```
CORRECT:
```python
lines = []
lines.append("line1")
lines.append("line2")
body = "\\n".join(lines)
```

### Other Rules:
1. print(json.dumps(result)) for structured output — last line
2. Always check for None: if not pr_info: exit(0)
3. Maximum execution time: 180 seconds
"""