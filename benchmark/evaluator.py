"""
Benchmark Evaluator
===================
Runs the QA test set through three agent modes and scores each answer
using an LLM judge (GPT-4o by default).

Modes:
  llm_only        – Direct LLM call, no tools, no persona
  vanilla_agent   – ReAct agent with tools, default persona, RAG disabled
  custom_agent    – ReAct agent with persona + RAG enabled
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Result for a single question in a given mode."""
    question_id: str
    question: str
    reference_answer: str
    generated_answer: str
    mode: str
    score: float = 0.0          # 0.0 – 10.0
    judge_reasoning: str = ""
    latency_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""
    mode: str
    num_questions: int = 0
    avg_score: float = 0.0
    avg_latency_s: float = 0.0
    results: List[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "num_questions": self.num_questions,
            "avg_score": round(self.avg_score, 3),
            "avg_latency_s": round(self.avg_latency_s, 3),
            "results": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# LLM-as-a-Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an impartial expert evaluator. "
    "Score the ANSWER against the REFERENCE on a scale of 0–10 "
    "(0 = completely wrong, 10 = perfect). "
    "Respond with valid JSON: "
    '{\"score\": <number>, \"reasoning\": \"<one sentence>\"}'
)

_JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Reference Answer: {reference}\n\n"
    "Generated Answer: {answer}\n\n"
    "Score this answer from 0 to 10."
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    """
    Runs the benchmark test set and produces per-mode BenchmarkReports.
    """

    def __init__(self, config_path: str = "config/agent_config.yaml"):
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh) or {}

        bench_cfg = self._config.get("benchmark", {})
        self._judge_model: str = bench_cfg.get("judge_model", "gpt-4o")
        self._test_set_path: str = bench_cfg.get(
            "test_set_path", "benchmark/datasets/qa_test_set.json"
        )
        self._output_dir: str = bench_cfg.get("output_dir", "benchmark/results")
        self._modes: List[str] = bench_cfg.get(
            "modes", ["llm_only", "vanilla_agent", "custom_agent_with_rag"]
        )
        os.makedirs(self._output_dir, exist_ok=True)

        # Initialise OpenAI client for judging
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_test_set(self) -> List[Dict[str, Any]]:
        """Load the QA test set from the configured JSON file."""
        with open(self._test_set_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %d questions from %s", len(data), self._test_set_path)
        return data

    def run(self, modes: Optional[List[str]] = None) -> Dict[str, BenchmarkReport]:
        """
        Execute the benchmark for specified modes (default: all configured).

        Returns a dict mapping mode_name → BenchmarkReport.
        """
        modes_to_run = modes or self._modes
        questions = self.load_test_set()
        reports: Dict[str, BenchmarkReport] = {}

        for mode in modes_to_run:
            logger.info("=== Running mode: %s ===", mode)
            report = self._run_mode(mode, questions)
            reports[mode] = report
            self._save_report(report)
            logger.info(
                "Mode %s: avg_score=%.2f, avg_latency=%.2fs",
                mode,
                report.avg_score,
                report.avg_latency_s,
            )

        return reports

    def print_summary(self, reports: Dict[str, "BenchmarkReport"]) -> None:
        """Print a human-readable comparison table."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        header = f"{'Mode':<30} {'Avg Score':>10} {'Avg Latency':>12}"
        print(header)
        print("-" * 60)
        for mode, report in sorted(reports.items()):
            print(
                f"{mode:<30} {report.avg_score:>10.2f} {report.avg_latency_s:>11.2f}s"
            )
        print("=" * 60)

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    def _run_mode(
        self, mode: str, questions: List[Dict[str, Any]]
    ) -> BenchmarkReport:
        results: List[QuestionResult] = []
        for q in questions:
            result = self._answer_question(mode, q)
            results.append(result)

        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        avg_latency = sum(r.latency_s for r in results) / len(results) if results else 0.0

        return BenchmarkReport(
            mode=mode,
            num_questions=len(results),
            avg_score=avg_score,
            avg_latency_s=avg_latency,
            results=results,
        )

    def _answer_question(
        self, mode: str, question: Dict[str, Any]
    ) -> QuestionResult:
        qid = question.get("id", "unknown")
        q_text = question["question"]
        reference = question.get("answer", "")

        t0 = time.perf_counter()
        try:
            generated = self._dispatch(mode, q_text)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] Question %s failed: %s", mode, qid, exc)
            generated = f"Error: {exc}"
        latency = time.perf_counter() - t0

        score, reasoning = self._judge(q_text, reference, generated)

        return QuestionResult(
            question_id=qid,
            question=q_text,
            reference_answer=reference,
            generated_answer=generated,
            mode=mode,
            score=score,
            judge_reasoning=reasoning,
            latency_s=round(latency, 3),
        )

    def _dispatch(self, mode: str, question: str) -> str:
        if mode == "llm_only":
            return self._llm_only(question)
        if mode == "vanilla_agent":
            return self._vanilla_agent(question)
        if mode == "custom_agent_with_rag":
            return self._custom_agent(question)
        raise ValueError(f"Unknown benchmark mode: {mode}")

    # ------------------------------------------------------------------
    # Individual mode implementations
    # ------------------------------------------------------------------

    def _llm_only(self, question: str) -> str:
        """Direct call to the LLM with no tools."""
        llm_cfg = self._config.get("llm", {})
        response = self._client.chat.completions.create(
            model=llm_cfg.get("model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def _vanilla_agent(self, question: str) -> str:
        """ReAct agent with tools but no persona and no RAG."""
        vanilla_cfg = dict(self._config)
        vanilla_cfg["rag"] = {**self._config.get("rag", {}), "enabled": False}
        vanilla_cfg["persona"] = {"default": "general"}

        from agent.core import Agent
        agent = Agent.__new__(Agent)
        agent.config = vanilla_cfg
        agent._setup_llm()

        from agent.persona import PersonaManager
        from agent.memory import MemoryManager
        from tools import ToolRegistry
        agent.persona_mgr = PersonaManager(vanilla_cfg)
        agent.memory_mgr = MemoryManager(vanilla_cfg)
        agent.tool_registry = ToolRegistry(vanilla_cfg)

        result = agent.run(question)
        return result.answer

    def _custom_agent(self, question: str) -> str:
        """Full custom agent with persona and RAG."""
        from agent.core import Agent
        agent = Agent(config_path="config/agent_config.yaml")

        # Pre-retrieve context from RAG
        context_docs = agent.memory_mgr.retrieve(question)

        persona = self._config.get("persona", {}).get("default", "linux_expert")
        result = agent.run(question, persona=persona, context_docs=context_docs)
        return result.answer

    # ------------------------------------------------------------------
    # LLM Judge
    # ------------------------------------------------------------------

    def _judge(
        self, question: str, reference: str, answer: str
    ) -> tuple[float, str]:
        """Score *answer* against *reference* using the judge LLM."""
        user_msg = _JUDGE_USER_TEMPLATE.format(
            question=question,
            reference=reference,
            answer=answer,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
            return min(max(score, 0.0), 10.0), reasoning
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Judge call failed: %s", exc)
            return 0.0, f"Judge error: {exc}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, report: BenchmarkReport) -> None:
        path = os.path.join(self._output_dir, f"{report.mode}_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Saved report: %s", path)
