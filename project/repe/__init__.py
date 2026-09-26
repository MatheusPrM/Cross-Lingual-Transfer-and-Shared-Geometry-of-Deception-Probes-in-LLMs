"""REPE dataset support (dialogue construction + labels, category-aware)."""

from project.repe.categories import (
    CATEGORY_DESCRIPTIONS,
    REPE_CATEGORIES,
    REPE_INSTRUCTIONS_BY_CATEGORY,
    RepeCategory,
    get_repe_instructions_for_category,
    is_valid_category,
)
from project.repe.loading import load_repe

__all__ = [
    "load_repe",
    "REPE_CATEGORIES",
    "REPE_INSTRUCTIONS_BY_CATEGORY",
    "RepeCategory",
    "CATEGORY_DESCRIPTIONS",
    "get_repe_instructions_for_category",
    "is_valid_category",
]
