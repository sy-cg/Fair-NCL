"""Streaming diagnostics for controlled sequence replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AugmentationDiagnostics:
    attempted_positions: int = 0
    replaced_positions: int = 0
    lower_skew_replacements: int = 0
    skew_change_sum: float = 0.0
    similarity_sum: float = 0.0
    similarity_count: int = 0

    def record(self,
               replaced: bool,
               similarity: Optional[float] = None,
               base_bias: Optional[float] = None,
               candidate_bias: Optional[float] = None) -> None:
        self.attempted_positions += 1
        if not replaced:
            return
        self.replaced_positions += 1
        if base_bias is not None and candidate_bias is not None:
            change = float(base_bias) - float(candidate_bias)
            self.skew_change_sum += change
            if change > 0:
                self.lower_skew_replacements += 1
        if similarity is not None:
            self.similarity_sum += float(similarity)
            self.similarity_count += 1

    def summary(self):
        return {
            "attempted_positions": int(self.attempted_positions),
            "replaced_positions": int(self.replaced_positions),
            "replacement_rate": (
                float(self.replaced_positions / self.attempted_positions)
                if self.attempted_positions else 0.0
            ),
            "lower_skew_fraction": (
                float(self.lower_skew_replacements / self.replaced_positions)
                if self.replaced_positions else 0.0
            ),
            "mean_skew_change": (
                float(self.skew_change_sum / self.replaced_positions)
                if self.replaced_positions else 0.0
            ),
            "mean_similarity": (
                float(self.similarity_sum / self.similarity_count)
                if self.similarity_count else None
            ),
        }
