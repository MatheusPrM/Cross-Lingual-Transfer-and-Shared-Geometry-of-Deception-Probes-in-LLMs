"""Hard input-side leakage check: judge-side fields must not appear in model-facing prompts."""

from __future__ import annotations

from .core import Scenario


class LeakageError(AssertionError):
    """Raised when a forbidden field substring is present in the rendered model input."""


DEFAULT_MIN_SUBSTR_LEN = 40
_CONTEXT_RADIUS = 80


def _candidate_substrings(scenario: Scenario, min_len: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in ("goal", "groundtruth", "topic", "dimension"):
        val = (getattr(scenario, name) or "").strip()
        if len(val) >= min_len:
            out.append((name, val))
    return out


def _format_leakage_error(
    *,
    scenario_id: str,
    field: str,
    needle: str,
    rendered_input: str,
) -> str:
    pos = rendered_input.find(needle)
    if pos < 0:
        ctx = ""
    else:
        lo = max(0, pos - _CONTEXT_RADIUS)
        hi = min(len(rendered_input), pos + len(needle) + _CONTEXT_RADIUS)
        ctx = rendered_input[lo:hi].replace("\n", "\\n")
        ctx = ("..." if lo > 0 else "") + ctx + ("..." if hi < len(rendered_input) else "")
    needle_trunc = needle if len(needle) <= 160 else needle[:160] + "..."
    return (
        f"leakage: forbidden field {field!r} of {scenario_id} appears in rendered input "
        f"(len={len(needle)})\n"
        f"  needle: {needle_trunc!r}\n"
        f"  context: ...{ctx}..."
    )


def assert_no_leakage(
    rendered_input: str,
    scenario: Scenario,
    *,
    min_substr_len: int = DEFAULT_MIN_SUBSTR_LEN,
    mode: str = "strict",
) -> list[tuple[str, str]]:
    """Return a list of (field, needle) leaks. `strict` raises on first; `warn` collects all; `off` is a no-op."""
    if mode == "off":
        return []
    hay = rendered_input or ""
    leaks: list[tuple[str, str]] = []
    for name, val in _candidate_substrings(scenario, min_substr_len):
        if val in hay:
            if mode == "strict":
                raise LeakageError(
                    _format_leakage_error(
                        scenario_id=scenario.scenario_id,
                        field=name,
                        needle=val,
                        rendered_input=hay,
                    )
                )
            leaks.append((name, val))
    return leaks
