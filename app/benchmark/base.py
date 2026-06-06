"""Benchmark protocol — abstract interface for benchmark integration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BenchmarkReport:
    """Aggregate results from a benchmark run."""
    results: list[any] = field(default_factory=list)
    total_duration_ms: float = 0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if getattr(r, "passed", False))

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class Benchmark(ABC):
    """Protocol for benchmark integration.

    Subclasses implement load_tasks, build_semantic_model, parse_answer,
    and score to adapt a specific benchmark format.
    """

    @abstractmethod
    def load_tasks(self) -> list[dict]: ...

    @abstractmethod
    def build_semantic_model(self, task_config: dict): ...

    @abstractmethod
    def parse_answer(self, raw: str, guidelines: str) -> str: ...

    @abstractmethod
    def score(self, predicted: str, ground_truth: str, answer_type: str) -> bool: ...
