"""
Evaluation Metrics
==================
Helper functions for computing aggregate evaluation metrics over
benchmark results, independent of the judge LLM.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


class EvaluationMetrics:
    """
    Computes standard evaluation metrics from a list of scores.
    All scores are expected to be in the range [0, 10].
    """

    @staticmethod
    def mean(scores: List[float]) -> float:
        """Arithmetic mean of scores."""
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def std(scores: List[float]) -> float:
        """Sample standard deviation of scores."""
        n = len(scores)
        if n < 2:
            return 0.0
        mu = EvaluationMetrics.mean(scores)
        variance = sum((x - mu) ** 2 for x in scores) / (n - 1)
        return math.sqrt(variance)

    @staticmethod
    def median(scores: List[float]) -> float:
        """Median of scores."""
        if not scores:
            return 0.0
        s = sorted(scores)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    @staticmethod
    def pass_at_threshold(
        scores: List[float], threshold: float = 7.0
    ) -> float:
        """Fraction of scores at or above *threshold* (default 7/10)."""
        if not scores:
            return 0.0
        passing = sum(1 for s in scores if s >= threshold)
        return passing / len(scores)

    @staticmethod
    def summarise(
        scores: List[float], threshold: float = 7.0
    ) -> Dict[str, float]:
        """Return a dict with all key metrics."""
        return {
            "mean": EvaluationMetrics.mean(scores),
            "median": EvaluationMetrics.median(scores),
            "std": EvaluationMetrics.std(scores),
            "pass_rate": EvaluationMetrics.pass_at_threshold(scores, threshold),
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "count": len(scores),
        }

    @staticmethod
    def compare_modes(
        mode_scores: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple modes side-by-side.

        Args:
            mode_scores: Mapping of mode_name → list of scores.

        Returns:
            Mapping of mode_name → summary dict.
        """
        return {
            mode: EvaluationMetrics.summarise(scores)
            for mode, scores in mode_scores.items()
        }

    @staticmethod
    def relative_improvement(
        baseline_scores: List[float],
        improved_scores: List[float],
    ) -> float:
        """
        Compute relative improvement (%) of *improved* over *baseline*.
        Returns a positive value when improved > baseline.
        """
        baseline_mean = EvaluationMetrics.mean(baseline_scores)
        improved_mean = EvaluationMetrics.mean(improved_scores)
        if baseline_mean == 0:
            return float("inf") if improved_mean > 0 else 0.0
        return ((improved_mean - baseline_mean) / baseline_mean) * 100.0
