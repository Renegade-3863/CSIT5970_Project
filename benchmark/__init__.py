"""
Benchmark Package
=================
Provides tools for evaluating the agent across three modes:
  1. LLM Only          – direct LLM call, no tools
  2. Vanilla Agent     – ReAct agent with tools but no persona/RAG
  3. Custom Agent      – full agent with persona + RAG

Evaluation uses LLM-as-a-Judge (GPT-4o by default).
"""

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.metrics import EvaluationMetrics

__all__ = ["BenchmarkEvaluator", "EvaluationMetrics"]
