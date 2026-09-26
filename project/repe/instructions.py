"""
RepE training-instruction strings, localized to match the training-data language.
"""

from __future__ import annotations

from project.repe.categories import (
    REPE_INSTRUCTIONS_BY_CATEGORY,
    RepeCategory,
    get_repe_instructions_for_category,
)

REPE_INSTRUCTIONS: dict[str, tuple[str, str, str]] = dict(
    REPE_INSTRUCTIONS_BY_CATEGORY["generic"]
)


def get_repe_instructions(
    lang: str, category: RepeCategory = "generic"
) -> tuple[str, str, str]:
    """Return (honest_system, deceptive_system, user_message) for ``(lang, category)``."""
    return get_repe_instructions_for_category(lang, category)
