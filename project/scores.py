"""
Container for per-dialogue (or per-token) probe scores and labels.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from project.types import Label


class Scores:
    """
    Holds scores for a set of dialogues. Each element is a tensor of per-token scores (or length-1 for a single score per dialogue).
    """

    scores: list[Tensor]
    labels: list[Label]

    def __init__(
        self,
        scores: Sequence[list[float] | Tensor],
        labels: list[Label] | None = None,
    ):
        self.scores = [torch.as_tensor(s) for s in scores]
        if labels is None:
            labels = [Label.HONEST] * len(self.scores)
        if len(labels) != len(self.scores):
            raise ValueError(f"labels length {len(labels)} != scores length {len(self.scores)}")
        self.labels = labels

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> Tensor:
        return self.scores[index]

    def get_scores_by_label(self) -> dict[Label, "Scores"]:
        """Split into Scores per label."""
        out: dict[Label, "Scores"] = {}
        for label in Label:
            indices = [i for i, l in enumerate(self.labels) if l == label]
            if indices:
                out[label] = Scores(
                    [self.scores[i] for i in indices],
                    [self.labels[i] for i in indices],
                )
            else:
                out[label] = Scores([], [])
        return out