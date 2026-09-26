"""Strict per-language prompt retrieval for DeceptionBench scenarios (no silent en fallback)."""

from __future__ import annotations

from .core import CONDITIONS, MissingLocalizedFieldError, Scenario


def get_prompt(scenario: Scenario, condition: str, *, lang: str = "en") -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r} (valid: {CONDITIONS})")
    text = scenario.prompts.get(condition)
    if not isinstance(text, str) or not text.strip():
        raise MissingLocalizedFieldError(
            f"missing prompt for condition {condition!r} on {scenario.scenario_id} (lang={lang})"
        )
    return text


def build_messages(scenario: Scenario, condition: str, *, lang: str = "en") -> list[dict[str, str]]:
    return [{"role": "user", "content": get_prompt(scenario, condition, lang=lang)}]
