"""
Test suite for the agent module (persona manager, memory manager, core agent).
These tests use mocking to avoid requiring real API keys.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure repo root is on sys.path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.persona import PersonaManager, _FALLBACK_PROMPT
from agent.memory import MemoryManager


# ---------------------------------------------------------------------------
# PersonaManager tests
# ---------------------------------------------------------------------------

class TestPersonaManager(unittest.TestCase):
    """Tests for PersonaManager loading and fallback behaviour."""

    def setUp(self):
        self.config = {"persona": {"default": "general"}}
        self.mgr = PersonaManager(self.config)

    def test_load_existing_persona(self):
        """Should load content from the general.txt prompt file."""
        prompt = self.mgr.get_prompt("general")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 10)

    def test_load_linux_expert_persona(self):
        """Should load the linux_expert persona from file."""
        prompt = self.mgr.get_prompt("linux_expert")
        self.assertIn("Linux", prompt)

    def test_load_security_expert_persona(self):
        """Should load the security_expert persona from file."""
        prompt = self.mgr.get_prompt("security_expert")
        self.assertIn("security", prompt.lower())

    def test_fallback_for_unknown_persona(self):
        """Unknown persona falls back to general or hardcoded default."""
        prompt = self.mgr.get_prompt("nonexistent_persona")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 5)

    def test_register_custom_persona(self):
        """Dynamically registered personas should be retrievable."""
        custom_text = "You are a data science wizard."
        self.mgr.register_persona("data_scientist", custom_text)
        self.assertEqual(self.mgr.get_prompt("data_scientist"), custom_text)

    def test_list_personas_returns_list(self):
        """list_personas() should return a list of available persona names."""
        personas = self.mgr.list_personas()
        self.assertIsInstance(personas, list)
        # At minimum the three bundled personas should be present
        self.assertIn("general", personas)

    def test_caching(self):
        """Second call should return cached value (file not re-read)."""
        p1 = self.mgr.get_prompt("general")
        p2 = self.mgr.get_prompt("general")
        self.assertIs(p1, p2)  # exact same object from cache


# ---------------------------------------------------------------------------
# MemoryManager tests
# ---------------------------------------------------------------------------

class TestMemoryManager(unittest.TestCase):
    """Tests for MemoryManager short-term buffer (RAG disabled)."""

    def setUp(self):
        self.config = {"rag": {"enabled": False}}
        self.mgr = MemoryManager(self.config)

    def test_add_and_get_history(self):
        """add_turn() and get_history() should work correctly."""
        self.mgr.add_turn("user", "Hello")
        self.mgr.add_turn("assistant", "Hi there!")
        history = self.mgr.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["content"], "Hi there!")

    def test_clear_history(self):
        """clear_history() should empty the buffer."""
        self.mgr.add_turn("user", "test")
        self.mgr.clear_history()
        self.assertEqual(self.mgr.get_history(), [])

    def test_retrieve_without_rag(self):
        """retrieve() should return [] when RAG is disabled."""
        result = self.mgr.retrieve("any query")
        self.assertEqual(result, [])

    def test_buffer_max_size(self):
        """Buffer should not exceed 20 entries (circular)."""
        for i in range(25):
            self.mgr.add_turn("user", f"message {i}")
        history = self.mgr.get_history()
        self.assertLessEqual(len(history), 20)


# ---------------------------------------------------------------------------
# Agent core – mocked LLM tests
# ---------------------------------------------------------------------------

class TestAgentCore(unittest.TestCase):
    """Tests for the Agent.run() loop using a mocked LLM client."""

    def _make_agent(self, mock_response: dict):
        """Create an Agent instance with the LLM client fully mocked."""
        from agent.core import Agent

        agent = Agent.__new__(Agent)
        agent.config = {
            "agent": {"max_iterations": 5, "verbose": False},
            "llm": {"provider": "openai", "model": "gpt-4o-mini",
                    "temperature": 0.1, "max_tokens": 512, "timeout": 30},
            "persona": {"default": "general"},
            "rag": {"enabled": False},
            "tools": {"enabled": []},
        }
        agent.provider = "openai"
        agent.model = "gpt-4o-mini"
        agent.temperature = 0.1
        agent.max_tokens = 512
        agent.timeout = 30

        # Mock LLM client
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(mock_response)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_usage = MagicMock()
        mock_usage.total_tokens = 42
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_completion
        agent._client = mock_client

        from agent.persona import PersonaManager
        from agent.memory import MemoryManager
        from tools import ToolRegistry

        agent.persona_mgr = PersonaManager(agent.config)
        agent.memory_mgr = MemoryManager(agent.config)
        agent.tool_registry = ToolRegistry(agent.config)
        return agent

    def test_finish_on_first_step(self):
        """Agent should return immediately when first response is 'finish'."""
        from agent.core import AgentResult

        mock_resp = {
            "thought": "I know the answer.",
            "action": "finish",
            "action_input": "The answer is 42.",
        }
        agent = self._make_agent(mock_resp)
        result = agent.run("What is the answer?")
        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.answer, "The answer is 42.")
        self.assertEqual(len(result.steps), 1)

    def test_result_includes_token_count(self):
        """AgentResult.total_tokens should reflect mocked usage."""
        mock_resp = {
            "thought": "Direct answer.",
            "action": "finish",
            "action_input": "Done.",
        }
        agent = self._make_agent(mock_resp)
        result = agent.run("Test question")
        self.assertEqual(result.total_tokens, 42)

    def test_result_to_dict(self):
        """AgentResult.to_dict() should serialise correctly."""
        mock_resp = {
            "thought": "Thinking...",
            "action": "finish",
            "action_input": "Final answer.",
        }
        agent = self._make_agent(mock_resp)
        result = agent.run("Serialisation test")
        d = result.to_dict()
        self.assertIn("answer", d)
        self.assertIn("steps", d)
        self.assertIn("total_tokens", d)
        self.assertIsInstance(d["steps"], list)

    def test_parse_response_valid_json(self):
        """_parse_response() should correctly parse valid JSON."""
        from agent.core import Agent
        raw = '{"thought": "ok", "action": "finish", "action_input": "done"}'
        parsed = Agent._parse_response(raw)
        self.assertEqual(parsed["action"], "finish")

    def test_parse_response_fallback(self):
        """_parse_response() should fall back gracefully for invalid JSON."""
        from agent.core import Agent
        parsed = Agent._parse_response("not json at all")
        self.assertIn("thought", parsed)
        self.assertEqual(parsed["action"], "finish")

    def test_parse_response_json_in_markdown(self):
        """_parse_response() should extract JSON from markdown code fences."""
        from agent.core import Agent
        raw = '```json\n{"thought": "test", "action": "finish", "action_input": "ok"}\n```'
        parsed = Agent._parse_response(raw)
        self.assertEqual(parsed["action"], "finish")


if __name__ == "__main__":
    unittest.main()
