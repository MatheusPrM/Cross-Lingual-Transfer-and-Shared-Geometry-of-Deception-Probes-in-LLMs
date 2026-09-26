"""
Model registry: the five paper models with pinned HF revisions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelInfo:
    """Single model entry."""

    id: str
    family: str
    size_b: int
    num_layers: int
    hf_id: str
    revision: str
    trust_remote_code: bool = True
    generation_batch_size: int = 4
    sweep_batch_size: int = 4
    notes: Optional[str] = None


DEFAULT_MODEL_ID = "gemma-2b"

MODEL_REGISTRY: list[ModelInfo] = [
    ModelInfo(
        id="gemma-2b",
        family="gemma",
        size_b=2,
        num_layers=26,
        hf_id="google/gemma-2-2b-it",
        generation_batch_size=4,
        revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        sweep_batch_size=8,
    ),
    ModelInfo(
        id="llama-8b",
        family="llama",
        size_b=8,
        num_layers=32,
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        generation_batch_size=4,
        revision="0e9e39f249a16976918f6564b8830bc894c89659",
        sweep_batch_size=4,
    ),
    ModelInfo(
        id="qwen3.5-9b",
        family="qwen",
        size_b=9,
        num_layers=32,
        hf_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        generation_batch_size=4,
        sweep_batch_size=2,
    ),
    ModelInfo(
        id="gemma-4-e4b",
        family="gemma",
        size_b=4,
        num_layers=42,
        hf_id="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        generation_batch_size=4,
        sweep_batch_size=2,
    ),
    ModelInfo(
        id="aya-expanse-8b",
        family="cohere",
        size_b=8,
        num_layers=32,
        hf_id="CohereForAI/aya-expanse-8b",
        generation_batch_size=4,
        revision="5062468bf9bc0c6035fd64e06274333ec127d980",
        sweep_batch_size=4,
        notes="Multilingual-by-design (Aya).",
    ),
]

_REGISTRY_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODEL_REGISTRY}


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Returns ModelInfo for an id."""
    return _REGISTRY_BY_ID.get(model_id)


def get_all_model_ids() -> list[str]:
    """Returns all model ids."""
    return [m.id for m in MODEL_REGISTRY]
