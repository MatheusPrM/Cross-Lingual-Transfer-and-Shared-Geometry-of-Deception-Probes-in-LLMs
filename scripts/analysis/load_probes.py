"""Load saved REPE probe directions by (model, lang, layer, category)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBES_ROOT = REPO_ROOT / "probes"

DEFAULT_LANGS: tuple[str, ...] = ("en", "pt", "es", "de", "zh")
DEFAULT_CATEGORY: str = "generic"


def _new_probe_path(model: str, lang: str, layer: int, category: str) -> Path:
    return PROBES_ROOT / model / lang / category / f"repe_probe_layer{int(layer)}.pt"


def _legacy_probe_path(model: str, lang: str, layer: int) -> Path:
    return PROBES_ROOT / model / lang / f"repe_probe_layer{int(layer)}.pt"


def probe_path(
    model: str,
    lang: str,
    layer: int,
    *,
    category: str = DEFAULT_CATEGORY,
    require_category_path: bool = False,
) -> Path:
    new = _new_probe_path(model, lang, layer, category)
    if new.exists() or require_category_path:
        return new
    legacy = _legacy_probe_path(model, lang, layer)
    return legacy if legacy.exists() else new


def load_probe_direction(
    model: str,
    lang: str,
    layer: int,
    *,
    expected_train_data: Optional[str] = None,
    strict: bool = False,
    category: str = DEFAULT_CATEGORY,
    coords: str = "activation",
) -> np.ndarray:
    """Probe coefficient as 1D numpy ``[hidden]``; ``coords='activation'`` returns ``w_eff = w/sigma`` (default), ``'standardized'`` returns raw legacy directions."""
    import pickle
    import warnings

    if coords not in {"activation", "standardized"}:
        raise ValueError(f"coords must be 'activation' or 'standardized'; got {coords!r}")

    path = probe_path(model, lang, layer, category=category)
    if not path.exists():
        raise FileNotFoundError(f"Probe not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    if expected_train_data is not None:
        stored = data.get("train_data")
        if stored is None:
            warnings.warn(f"Legacy probe at {path} (no train_data stamp); skipping integrity check.", RuntimeWarning)
        elif str(stored).strip() != str(expected_train_data).strip():
            stored_str, expected_str = str(stored).strip(), str(expected_train_data).strip()
            synonyms = {f"repe_{lang}": f"repe_generic_{lang}", f"repe_generic_{lang}": f"repe_{lang}"}
            if synonyms.get(stored_str) != expected_str:
                msg = f"Probe train_data mismatch at {path}: file says {stored!r}, caller expected {expected_train_data!r}."
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)

    directions = data["directions"]
    if hasattr(directions, "cpu"):
        directions_np = directions.cpu().numpy()
    else:
        directions_np = np.asarray(directions)
    layers_meta = list(data.get("layers", []) or []) 
    if directions_np.ndim == 2 and len(layers_meta) == directions_np.shape[0] and layer in layers_meta:
        row_idx = layers_meta.index(layer)
        w_std = directions_np[row_idx]
    elif directions_np.ndim == 2 and directions_np.shape[0] == 1:
        w_std = directions_np[0]
    elif directions_np.ndim == 1:
        w_std = directions_np
    else:
        raise ValueError(
            f"Unexpected directions shape {directions_np.shape} at {path}; cannot locate layer {layer}."
        )

    if coords == "standardized":
        return w_std.astype(np.float32)

    scale_flag = bool(data.get("scale", True))
    if not scale_flag:
        return w_std.astype(np.float32)
    scaler_scale = data.get("scaler_scale")
    if scaler_scale is None:
        raise RuntimeError(
            f"Probe at {path} has scale=True but scaler_scale=None — refusing to return an "
            "activation-space direction for an un-unscalable legacy probe. Pass coords='standardized' "
            "to reproduce the legacy raw behavior, or retrain the probe."
        )
    scaler_np = scaler_scale.cpu().numpy() if hasattr(scaler_scale, "cpu") else np.asarray(scaler_scale)
    if scaler_np.ndim == 2 and len(layers_meta) == scaler_np.shape[0] and layer in layers_meta:
        sigma = scaler_np[layers_meta.index(layer)]
    elif scaler_np.ndim == 2 and scaler_np.shape[0] == 1:
        sigma = scaler_np[0]
    else:
        raise ValueError(
            f"Unexpected scaler_scale shape {scaler_np.shape} at {path}; cannot locate layer {layer}."
        )
    w_eff = (w_std / (sigma + 1e-12)).astype(np.float32)
    return w_eff


def load_language_stack(
    model: str,
    layer: int,
    *,
    langs: Iterable[str] = DEFAULT_LANGS,
    strict: bool = False,
    category: str = DEFAULT_CATEGORY,
) -> tuple[list[str], np.ndarray]:
    """Stack per-language probe directions at one layer. Returns ``(present_langs, V)``."""
    found: list[str] = []
    rows: list[np.ndarray] = []
    for lang in langs:
        try:
            v = load_probe_direction(
                model, lang, layer,
                expected_train_data=f"repe_{category}_{lang}",
                strict=strict,
                category=category,
            )
        except FileNotFoundError:
            continue
        found.append(lang)
        rows.append(v)
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return found, np.stack(rows, axis=0)


def available_layers(
    model: str,
    lang: str,
    *,
    category: str = DEFAULT_CATEGORY,
) -> list[int]:
    """Layers with a probe file for ``(model, lang, category)``; legacy fallback only for generic."""
    def _scan(base: Path) -> list[int]:
        if not base.exists():
            return []
        layers: list[int] = []
        for p in base.glob("repe_probe_layer*.pt"):
            try:
                layers.append(int(p.stem.replace("repe_probe_layer", "")))
            except ValueError:
                continue
        return layers

    new_layers = _scan(PROBES_ROOT / model / lang / category)
    if new_layers:
        return sorted(new_layers)
    if category == DEFAULT_CATEGORY:
        return sorted(_scan(PROBES_ROOT / model / lang))  
    return []


def common_layers(
    model: str,
    *,
    langs: Iterable[str] = DEFAULT_LANGS,
    category: str = DEFAULT_CATEGORY,
) -> list[int]:
    """Layers with probes in *all* requested languages, for one category."""
    sets = []
    for lang in langs:
        layers = set(available_layers(model, lang, category=category))
        if not layers:
            raise FileNotFoundError(
                f"{model}/{lang}/{category}: no probes found; the requested "
                f"language set is incomplete")
        sets.append(layers)
    if not sets:
        return []
    common: set[int] = set.intersection(*sets)
    return sorted(common)
