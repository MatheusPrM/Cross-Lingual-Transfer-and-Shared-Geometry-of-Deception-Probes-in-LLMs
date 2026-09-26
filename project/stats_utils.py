"""Torch-free statistical helpers shared by analysis scripts."""

from __future__ import annotations

import math


def spearman_brown(split_half_r: float) -> float:
    """
    Full-test reliability from a split-half correlation: 2r / (1 + r).
    """
    r = float(split_half_r)
    if r <= -1.0:
        return float("nan")
    return 2.0 * r / (1.0 + r)


def attenuation_corrected(raw_cos: float, rel_a: float, rel_b: float) -> float:
    """
    Spearman (1904) attenuation correction with full-test reliabilities.
    """
    if rel_a <= 0 or rel_b <= 0 or not (math.isfinite(rel_a) and math.isfinite(rel_b)):
        return float("nan")
    return float(raw_cos) / math.sqrt(rel_a * rel_b)
