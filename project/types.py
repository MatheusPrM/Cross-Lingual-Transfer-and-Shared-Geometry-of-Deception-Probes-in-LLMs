"""
Shared types for the pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

Judgement = Literal["honest", "deceptive", "ambiguous", "skip"] | str | int
Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """Single message in a dialogue."""

    role: Role
    content: str
    detect: bool

Dialogue = list[Message]


class Label(Enum):
    """Dialogue label."""

    HONEST = "honest"
    DECEPTIVE = "deceptive"
    AMBIGUOUS = "ambiguous"
    SKIP = "skip"


DialogueDataType = tuple[list[Dialogue], list[Label], dict[str, list[Any]] | None]