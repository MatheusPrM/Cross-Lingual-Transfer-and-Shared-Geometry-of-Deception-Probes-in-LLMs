"""Single source of truth for frozen-layer paths. Torch-free."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
FROZEN_LAYERS_ROOT = _REPO / "results" / "frozen_layers"


def chosen_layer_path(model: str) -> Path:
    return FROZEN_LAYERS_ROOT / model / "chosen_layer.json"


def load_frozen_layer(model: str) -> int:
    p = chosen_layer_path(model)
    if not p.exists():
        raise FileNotFoundError(
            f"No frozen-layer record for {model} at {p}; run "
            "scripts/sweep_layers.py then scripts/select_layer.py."
        )
    return int(json.loads(p.read_text(encoding="utf-8"))["chosen_layer"])
