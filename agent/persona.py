"""
Persona Manager
===============
Loads and manages expert persona prompts that are injected into the
agent's system message to give it a specific identity and expertise.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

# Fallback minimal prompt when no file is found
_FALLBACK_PROMPT = (
    "You are a helpful AI assistant. Answer questions clearly and concisely."
)

KNOWN_PERSONAS = ("general", "linux_expert", "security_expert")


class PersonaManager:
    """
    Manages a registry of named persona prompts loaded from text files.

    Each persona corresponds to a .txt file in ``config/prompts/``.
    """

    def __init__(self, config: Dict):
        persona_cfg = config.get("persona", {})
        # Resolve prompt directory relative to the config's location or CWD
        self._prompt_dir = os.path.join("config", "prompts")
        self._cache: Dict[str, str] = {}
        self._default = persona_cfg.get("default", "general")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prompt(self, persona_name: str) -> str:
        """
        Return the system-prompt text for *persona_name*.
        Falls back to the 'general' persona, then to a hardcoded default.
        """
        if persona_name in self._cache:
            return self._cache[persona_name]

        prompt = self._load_from_file(persona_name)
        if prompt is None and persona_name != "general":
            logger.warning(
                "Persona '%s' not found; falling back to 'general'.", persona_name
            )
            prompt = self._load_from_file("general")
        if prompt is None:
            logger.warning("No persona files found; using built-in fallback.")
            prompt = _FALLBACK_PROMPT

        self._cache[persona_name] = prompt
        return prompt

    def list_personas(self) -> list[str]:
        """Return names of all available personas (based on .txt files present)."""
        if not os.path.isdir(self._prompt_dir):
            return []
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self._prompt_dir)
            if f.endswith(".txt")
        ]

    def register_persona(self, name: str, prompt_text: str) -> None:
        """Dynamically register a new persona at runtime (in-memory only)."""
        self._cache[name] = prompt_text
        logger.info("Registered new persona: '%s'", name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_file(self, persona_name: str) -> str | None:
        file_path = os.path.join(self._prompt_dir, f"{persona_name}.txt")
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        logger.debug("Loaded persona '%s' from %s", persona_name, file_path)
        return content if content else None
