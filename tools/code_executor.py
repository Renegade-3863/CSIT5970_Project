"""
Code Executor Tool
==================
Safely executes Python or Bash code snippets in a restricted subprocess
and returns the output (stdout + stderr).

Security notes:
- Code runs in a separate subprocess with a configurable timeout.
- Only Python and Bash are supported.
- The execution environment does NOT have network access by design
  (enforced at the infrastructure level, e.g. Docker network policies).
- stdin is disabled.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
import os
from typing import Any, Dict

from tools.base import BaseTool

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 4000


class CodeExecutorTool(BaseTool):
    """
    Tool: code_executor
    Execute a Python or Bash code snippet and return stdout/stderr.
    Input format: "<language>\\n<code>"
    Supported languages: python, bash
    Example input:
        python
        print(sum(range(10)))
    """

    name = "code_executor"
    description = (
        "Execute a Python or Bash code snippet and return the output. "
        "Input format: first line is the language ('python' or 'bash'), "
        "followed by the code on subsequent lines. "
        "Example: 'python\\nprint(1+1)'"
    )

    def __init__(self, config: Dict[str, Any]):
        tool_cfg = config.get("tools", {}).get("code_executor", {})
        self._timeout: int = int(tool_cfg.get("timeout", 30))
        self._allowed: list = tool_cfg.get(
            "allowed_languages", ["python", "bash"]
        )

    def run(self, tool_input: str) -> str:
        lines = tool_input.strip().splitlines()
        if not lines:
            return "Error: empty input."

        language = lines[0].strip().lower()
        code = "\n".join(lines[1:]) if len(lines) > 1 else ""

        if language not in self._allowed:
            return (
                f"Error: language '{language}' is not allowed. "
                f"Allowed: {', '.join(self._allowed)}."
            )
        if not code.strip():
            return "Error: no code provided after the language identifier."

        if language == "python":
            return self._run_python(code)
        if language == "bash":
            return self._run_bash(code)

        return f"Error: unsupported language '{language}'."

    # ------------------------------------------------------------------
    # Language runners
    # ------------------------------------------------------------------

    def _run_python(self, code: str) -> str:
        """Write code to a temp file and execute with the current Python."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(textwrap.dedent(code))
            tmp_path = tmp.name
        try:
            return self._exec_subprocess(
                [sys.executable, tmp_path], timeout=self._timeout
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_bash(self, code: str) -> str:
        """Execute Bash code via /bin/bash -c."""
        return self._exec_subprocess(
            ["/bin/bash", "-c", code], timeout=self._timeout
        )

    @staticmethod
    def _exec_subprocess(cmd: list, timeout: int) -> str:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n[stderr]\n" + result.stderr) if output else result.stderr
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            output = output.strip()
            if len(output) > _MAX_OUTPUT_CHARS:
                output = output[:_MAX_OUTPUT_CHARS] + "\n... (output truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {timeout} seconds."
        except FileNotFoundError as exc:
            return f"Error: command not found — {exc}"
        except Exception as exc:  # pylint: disable=broad-except
            return f"Error: unexpected exception — {exc}"
