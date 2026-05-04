"""
Base Tool Definitions
=====================
Defines the abstract BaseTool interface and the ToolRegistry that
manages tool registration and dispatching.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    #: Unique identifier used by the agent (e.g. 'knowledge_base')
    name: str = ""
    #: Human-readable description shown to the LLM
    description: str = ""

    @abstractmethod
    def run(self, tool_input: str) -> str:
        """Execute the tool with the given input and return a string result."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool name={self.name!r}>"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Manages tool registration and execution.

    Tools are initialised lazily from the agent config on first call.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._tools: Dict[str, BaseTool] = {}
        self._load_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool must have a non-empty 'name' attribute.")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a registered tool by name."""
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, tool_input: Any) -> str:
        """Run the named tool with *tool_input*; return observation string."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return (
                f"Error: unknown tool '{name}'. Available tools: {available}. "
                "Use one of the listed tool names or 'finish'."
            )
        try:
            result = tool.run(str(tool_input))
            return result if result else "(no output)"
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Tool '%s' raised an exception: %s", name, exc)
            return f"Error running tool '{name}': {exc}"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe_tools(self) -> str:
        """Return a formatted description of all registered tools."""
        if not self._tools:
            return "No tools available."
        lines: List[str] = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load_tools(self) -> None:
        enabled = (
            self._config.get("tools", {}).get("enabled") or []
        )
        for tool_name in enabled:
            try:
                self._load_tool(tool_name)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not load tool '%s': %s", tool_name, exc)

    def _load_tool(self, name: str) -> None:
        """Instantiate and register a built-in tool by name."""
        if name == "knowledge_base":
            from tools.knowledge_base import KnowledgeBaseTool
            self.register(KnowledgeBaseTool(self._config))
        elif name == "code_executor":
            from tools.code_executor import CodeExecutorTool
            self.register(CodeExecutorTool(self._config))
        elif name == "web_search":
            from tools.web_search import WebSearchTool
            self.register(WebSearchTool(self._config))
        else:
            logger.warning("Unknown built-in tool '%s'.", name)
