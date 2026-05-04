"""
Web Search Tool
===============
Performs lightweight web searches using the DuckDuckGo Instant Answer
API (no API key required) and returns a concise summary of top results.

Falls back gracefully when network access is unavailable.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from tools.base import BaseTool

logger = logging.getLogger(__name__)

_DDG_URL = "https://api.duckduckgo.com/"
_TIMEOUT = 10  # seconds per HTTP request


class WebSearchTool(BaseTool):
    """
    Tool: web_search
    Search the web for up-to-date information.
    Input: a natural-language search query.
    Output: a brief summary of top search results.
    """

    name = "web_search"
    description = (
        "Search the web for current information. "
        "Input should be a concise search query. "
        "Returns titles, URLs, and short snippets from top results."
    )

    def __init__(self, config: Dict[str, Any]):
        tool_cfg = config.get("tools", {}).get("web_search", {})
        self._max_results: int = int(tool_cfg.get("max_results", 5))

    def run(self, tool_input: str) -> str:
        query = tool_input.strip()
        if not query:
            return "Error: empty search query."

        try:
            results = self._search_ddg(query)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Web search failed: %s", exc)
            return (
                f"Web search unavailable: {exc}. "
                "Consider using the knowledge_base tool instead."
            )

        if not results:
            return "No results found for the given query."

        lines: List[str] = [f"Search results for: '{query}'\n"]
        for i, item in enumerate(results[: self._max_results], start=1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_ddg(self, query: str) -> List[Dict[str, str]]:
        """
        Use the DuckDuckGo Instant Answer API.
        Returns a list of {title, url, snippet} dicts.
        """
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"{_DDG_URL}?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CSIT5970-Agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results: List[Dict[str, str]] = []

        # Abstract (top answer)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Abstract"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["Abstract"],
            })

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if len(results) >= self._max_results:
                break
            if "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })

        return results
