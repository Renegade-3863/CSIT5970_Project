"""
Tools Package
=============
Provides a ToolRegistry and built-in tools for the agent:
  - knowledge_base: RAG-backed knowledge retrieval
  - code_executor:  Safe Python / Bash code execution
  - web_search:     Lightweight web search via DuckDuckGo
"""

from tools.base import BaseTool, ToolRegistry

__all__ = ["BaseTool", "ToolRegistry"]
