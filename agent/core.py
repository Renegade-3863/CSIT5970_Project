"""
Agent Core Module
=================
Implements a ReAct (Reasoning + Acting) agent loop that coordinates:
  - LLM inference (OpenAI / Anthropic)
  - Tool execution
  - Persona management
  - Memory / RAG retrieval
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Any] = None
    observation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    """Final result produced by the agent after completing a task."""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    A ReAct-style autonomous agent that:
    1. Receives a task / question.
    2. Thinks (calls the LLM to reason).
    3. Acts (selects and executes a tool).
    4. Observes the result and repeats until a final answer is produced.
    """

    # System template injected before the persona block
    _REACT_SYSTEM_SUFFIX = """
You operate in a Thought → Action → Observation loop.

Available tools:
{tool_descriptions}

Response format (JSON):
{{
  "thought": "<your step-by-step reasoning>",
  "action": "<tool name or 'finish'>",
  "action_input": "<tool input string, or final answer when action=='finish'>"
}}

Rules:
- Always output valid JSON.
- Use 'finish' as the action when you have a definitive answer.
- Do NOT call a tool more than 3 times with the same input.
- Be concise in 'thought'; be precise in 'action_input'.
"""

    def __init__(self, config_path: str = "config/agent_config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_llm()

        # Lazy imports to avoid hard dependency if components are unused
        from agent.persona import PersonaManager
        from agent.memory import MemoryManager
        from tools import ToolRegistry

        self.persona_mgr = PersonaManager(self.config)
        self.memory_mgr = MemoryManager(self.config)
        self.tool_registry = ToolRegistry(self.config)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.warning("Config file %s not found; using defaults.", path)
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _setup_llm(self) -> None:
        llm_cfg = self.config.get("llm", {})
        self.provider = llm_cfg.get("provider", "openai")
        self.model = llm_cfg.get("model", "gpt-4o-mini")
        self.temperature = llm_cfg.get("temperature", 0.1)
        self.max_tokens = llm_cfg.get("max_tokens", 4096)
        self.timeout = llm_cfg.get("timeout", 60)

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=self.timeout,
                )
            except ImportError:
                logger.error("openai package not installed.")
                self._client = None
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
            except ImportError:
                logger.error("anthropic package not installed.")
                self._client = None
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # ------------------------------------------------------------------
    # Core ReAct loop
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        persona: Optional[str] = None,
        context_docs: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
    ) -> AgentResult:
        """
        Execute the ReAct loop for a given task.

        Args:
            task: The user's question or instruction.
            persona: Override the default persona (e.g. 'linux_expert').
            context_docs: Pre-fetched context documents to include.
            max_iterations: Override config max_iterations.

        Returns:
            AgentResult containing the final answer and reasoning trace.
        """
        max_iter = max_iterations or self.config.get("agent", {}).get("max_iterations", 10)
        persona_name = persona or self.config.get("persona", {}).get("default", "general")
        verbose = self.config.get("agent", {}).get("verbose", False)

        system_prompt = self._build_system_prompt(persona_name)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Optionally inject retrieved context at the start
        if context_docs:
            ctx_text = "\n\n---\n\n".join(context_docs)
            messages.append({
                "role": "user",
                "content": (
                    f"Relevant background knowledge:\n{ctx_text}\n\n"
                    f"Task: {task}"
                ),
            })
        else:
            messages.append({"role": "user", "content": task})

        steps: List[AgentStep] = []
        total_tokens = 0

        for i in range(max_iter):
            raw, tokens = self._call_llm(messages)
            total_tokens += tokens

            parsed = self._parse_response(raw)
            thought = parsed.get("thought", "")
            action = parsed.get("action", "finish")
            action_input = parsed.get("action_input", "")

            if verbose:
                logger.info("[Iter %d] Thought: %s", i + 1, thought)
                logger.info("[Iter %d] Action: %s | Input: %s", i + 1, action, action_input)

            if action == "finish":
                step = AgentStep(thought=thought, action="finish", action_input=action_input)
                steps.append(step)
                return AgentResult(
                    answer=str(action_input),
                    steps=steps,
                    total_tokens=total_tokens,
                    model=self.model,
                )

            # Execute tool
            observation = self.tool_registry.execute(action, action_input)
            if verbose:
                logger.info("[Iter %d] Observation: %s", i + 1, observation[:200])

            step = AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)

            # Append assistant turn + observation to message history
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue reasoning.",
            })

        # Exhausted iterations — return best effort
        last_thought = steps[-1].thought if steps else "No reasoning produced."
        return AgentResult(
            answer=last_thought,
            steps=steps,
            total_tokens=total_tokens,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, messages: List[Dict[str, str]]) -> tuple[str, int]:
        """Call the LLM and return (raw_text, tokens_used)."""
        if self._client is None:
            raise RuntimeError("LLM client not initialised.")

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return content, tokens

        if self.provider == "anthropic":
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_msgs = [m for m in messages if m["role"] != "system"]
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_msg,
                messages=user_msgs,
            )
            content = response.content[0].text if response.content else ""
            tokens = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else 0
            )
            return content, tokens

        raise ValueError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Parse JSON response from the LLM; fall back gracefully."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract JSON block from markdown fences
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse LLM response as JSON: %s", raw[:200])
            return {"thought": raw, "action": "finish", "action_input": raw}

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, persona_name: str) -> str:
        persona_text = self.persona_mgr.get_prompt(persona_name)
        tool_descriptions = self.tool_registry.describe_tools()
        react_suffix = self._REACT_SYSTEM_SUFFIX.format(
            tool_descriptions=tool_descriptions
        )
        return f"{persona_text}\n\n{react_suffix}"
