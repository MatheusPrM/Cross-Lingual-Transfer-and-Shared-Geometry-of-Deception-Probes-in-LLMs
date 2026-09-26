"""DeceptionBench (NeurIPS 2025 D&B) cross-lingual response-only adaptation."""

from __future__ import annotations

from .core import (
    CONDITIONS,
    DOMAIN_FILES,
    MissingLocalizedFieldError,
    Scenario,
    load_deceptionbench_all,
    load_deceptionbench_domain,
    parse_condition,
)
from .dialogue import fill_assistant_response, messages_to_dialogue, tokenize_dialogues_length_masked
from .grading import is_wellformed, looks_like_refusal, normalize_label
from .judge import (
    JUDGE_SCHEMA_VERSION,
    JudgeLabel,
    VALID_LABELS,
    build_judge_prompt,
    judge_response,
    parse_judge_output,
)
from .leakage import LeakageError, assert_no_leakage
from .prompts import build_messages, get_prompt
from .rollouts import (
    DECEPTIONBENCH_ROLLOUT_FAMILY,
    DECEPTIONBENCH_SCHEMA_VERSION,
    load_deceptionbench_rollouts,
    save_deceptionbench_rollouts,
)

__all__ = [
    "CONDITIONS",
    "DECEPTIONBENCH_ROLLOUT_FAMILY",
    "DECEPTIONBENCH_SCHEMA_VERSION",
    "DOMAIN_FILES",
    "JudgeLabel",
    "LeakageError",
    "MissingLocalizedFieldError",
    "Scenario",
    "VALID_LABELS",
    "JUDGE_SCHEMA_VERSION",
    "assert_no_leakage",
    "build_judge_prompt",
    "build_messages",
    "fill_assistant_response",
    "get_prompt",
    "is_wellformed",
    "judge_response",
    "load_deceptionbench_all",
    "load_deceptionbench_domain",
    "load_deceptionbench_rollouts",
    "looks_like_refusal",
    "messages_to_dialogue",
    "tokenize_dialogues_length_masked",
    "normalize_label",
    "parse_condition",
    "parse_judge_output",
    "save_deceptionbench_rollouts",
]
