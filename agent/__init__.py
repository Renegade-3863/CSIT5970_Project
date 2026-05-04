"""
CSIT5970 Custom Agent Package
=============================
An end-to-end autonomous agent with persona injection, RAG memory, and custom tools.
"""

from agent.core import Agent
from agent.persona import PersonaManager
from agent.memory import MemoryManager

__all__ = ["Agent", "PersonaManager", "MemoryManager"]
__version__ = "1.0.0"
