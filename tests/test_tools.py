"""
Test suite for the tools module.
Uses mocking so no real LLM / internet access is required.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.base import BaseTool, ToolRegistry
from tools.code_executor import CodeExecutorTool
from tools.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# BaseTool / ToolRegistry tests
# ---------------------------------------------------------------------------

class ConcreteTestTool(BaseTool):
    """Minimal concrete tool for testing the registry."""
    name = "test_tool"
    description = "A tool for unit testing."

    def run(self, tool_input: str) -> str:
        return f"echo:{tool_input}"


class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.config = {"tools": {"enabled": []}}
        self.registry = ToolRegistry(self.config)

    def test_register_and_execute(self):
        """Registered tools should be callable through execute()."""
        self.registry.register(ConcreteTestTool())
        result = self.registry.execute("test_tool", "hello")
        self.assertEqual(result, "echo:hello")

    def test_unknown_tool_returns_error(self):
        """Calling an unknown tool should return an error string."""
        result = self.registry.execute("nonexistent_tool", "x")
        self.assertIn("Error", result)
        self.assertIn("nonexistent_tool", result)

    def test_describe_tools(self):
        """describe_tools() should list all registered tools."""
        self.registry.register(ConcreteTestTool())
        desc = self.registry.describe_tools()
        self.assertIn("test_tool", desc)
        self.assertIn("A tool for unit testing.", desc)

    def test_list_tools(self):
        """list_tools() should return names of registered tools."""
        self.registry.register(ConcreteTestTool())
        names = self.registry.list_tools()
        self.assertIn("test_tool", names)

    def test_unregister_tool(self):
        """Unregistered tools should no longer be executable."""
        self.registry.register(ConcreteTestTool())
        self.registry.unregister("test_tool")
        result = self.registry.execute("test_tool", "x")
        self.assertIn("Error", result)

    def test_describe_no_tools(self):
        """describe_tools() on empty registry returns informative message."""
        desc = self.registry.describe_tools()
        self.assertEqual(desc, "No tools available.")

    def test_tool_exception_is_caught(self):
        """Exceptions inside tool.run() should be caught and returned as error string."""
        class BrokenTool(BaseTool):
            name = "broken"
            description = "Breaks."
            def run(self, tool_input: str) -> str:
                raise RuntimeError("intentional error")

        self.registry.register(BrokenTool())
        result = self.registry.execute("broken", "anything")
        self.assertIn("Error running tool", result)
        self.assertIn("broken", result)


# ---------------------------------------------------------------------------
# CodeExecutorTool tests
# ---------------------------------------------------------------------------

class TestCodeExecutorTool(unittest.TestCase):

    def setUp(self):
        self.config = {
            "tools": {
                "code_executor": {
                    "timeout": 10,
                    "allowed_languages": ["python", "bash"],
                }
            }
        }
        self.tool = CodeExecutorTool(self.config)

    def test_python_execution(self):
        """Should execute simple Python and return stdout."""
        result = self.tool.run("python\nprint('hello world')")
        self.assertIn("hello world", result)

    def test_python_arithmetic(self):
        """Should correctly evaluate arithmetic."""
        result = self.tool.run("python\nprint(2 + 2)")
        self.assertIn("4", result)

    def test_bash_execution(self):
        """Should execute Bash and return output."""
        result = self.tool.run("bash\necho 'bash test'")
        self.assertIn("bash test", result)

    def test_disallowed_language(self):
        """Should refuse unsupported languages."""
        result = self.tool.run("javascript\nconsole.log('hi')")
        self.assertIn("Error", result)
        self.assertIn("not allowed", result)

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        result = self.tool.run("")
        self.assertIn("Error", result)

    def test_empty_code(self):
        """Should handle missing code after language identifier."""
        result = self.tool.run("python\n")
        self.assertIn("Error", result)

    def test_timeout(self):
        """Long-running scripts should be killed after timeout."""
        tool = CodeExecutorTool({
            "tools": {
                "code_executor": {
                    "timeout": 1,
                    "allowed_languages": ["python"],
                }
            }
        })
        result = tool.run("python\nimport time; time.sleep(10)")
        self.assertIn("timed out", result.lower())

    def test_python_stderr_captured(self):
        """stderr output should be included in the result."""
        result = self.tool.run("python\nimport sys; sys.stderr.write('err output')")
        self.assertIn("err output", result)

    def test_exit_code_nonzero(self):
        """Non-zero exit should be indicated in output."""
        result = self.tool.run("bash\nexit 1")
        self.assertIn("exit code", result)

    def test_output_truncation(self):
        """Very large outputs should be truncated."""
        big_output_code = "python\nprint('x' * 10000)"
        result = self.tool.run(big_output_code)
        self.assertLessEqual(len(result), 5000)


# ---------------------------------------------------------------------------
# WebSearchTool tests (mocked network)
# ---------------------------------------------------------------------------

class TestWebSearchTool(unittest.TestCase):

    def setUp(self):
        self.config = {"tools": {"web_search": {"max_results": 3}}}
        self.tool = WebSearchTool(self.config)

    def test_empty_query(self):
        """Empty query should return an error."""
        result = self.tool.run("")
        self.assertIn("Error", result)

    def test_successful_search(self):
        """Mocked search should return formatted results."""
        mock_response = {
            "Abstract": "Python is a high-level programming language.",
            "Heading": "Python (programming language)",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python",
            "RelatedTopics": [
                {
                    "Text": "Python Software Foundation",
                    "FirstURL": "https://www.python.org",
                }
            ],
        }

        import json
        import io

        mock_resp_bytes = json.dumps(mock_response).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.read.return_value = mock_resp_bytes
            mock_urlopen.return_value = mock_ctx

            result = self.tool.run("Python programming language")

        self.assertIn("Python", result)
        self.assertIn("Search results", result)

    def test_network_failure_handled(self):
        """Network errors should return a graceful error message."""
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = self.tool.run("some query")
        self.assertIn("unavailable", result.lower())


if __name__ == "__main__":
    unittest.main()
