"""SVD-decompose per-language probes into shared direction + per-lang residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scripts.analysis.load_probes import (
    DEFAULT_CATEGORY,
    DEFAULT_LANGS,
    REPO_ROOT,
    common_layers,
    load_language_stack,
)

GEOMETRY_DIR = REPO_ROOT / "results" / "geometry"


def _model_out_dir(out_root: Path, model: str, category: str) -> Path:
    """Geometry output dir. ``generic`` keeps the flat layout (downstream analyses read v_shared there);
    control concepts are namespaced under the category so they never clobber the deception geometry."""
    base = out_root / model
    return base if category == DEFAULT_CATEGORY else base / category


def cosine_matrix(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``V``."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Vn = V / norms
    return Vn @ Vn.T


def component_spearman_matrix(V: np.ndarray) -> np.ndarray:
    """
    Pairwise Spearman rank correlation between the COORDINATES of row vectors.
    """
    ranks = np.argsort(np.argsort(V, axis=1), axis=1).astype(np.float64)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.maximum(np.linalg.norm(ranks, axis=1, keepdims=True), 1e-12)
    Rn = ranks / norms
    return Rn @ Rn.T


def svd_decomposition(V: np.ndarray, *, normalise_rows: bool = True) -> dict:
    """SVD of row-normalised ``V``; returns v_shared, singular_values, projections, residuals, variance_explained_top, residual_norms."""
    if V.shape[0] < 2:
        raise ValueError(f"Need at least 2 language probes; got {V.shape[0]}")
    Vw = V.copy()
    if normalise_rows:
        Vw = Vw / np.maximum(np.linalg.norm(Vw, axis=1, keepdims=True), 1e-12)
    _U, S, VT = np.linalg.svd(Vw, full_matrices=False)
    v_shared = VT[0]
    if (Vw @ v_shared).mean() < 0:
        v_shared = -v_shared
    projections = Vw @ v_shared
    residuals = Vw - np.outer(projections, v_shared)
    total = float((S ** 2).sum())
    return {
        "v_shared": v_shared,
        "singular_values": S,
        "projections": projections,
        "residuals": residuals,
        "variance_explained_top": float((S[0] ** 2) / total) if total > 0 else float("nan"),
        "residual_norms": np.linalg.norm(residuals, axis=1),
    }


def _unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` normalized to unit length (safe for near-zero vectors)."""
    return v / max(float(np.linalg.norm(v)), 1e-12)


def load_lang_identity(layer_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """Language-identity directions saved by scripts/analysis/language_identity.py, or None."""
    p = layer_dir / "lang_identity.npz"
    if not p.exists():
        return None
    out: dict[str, np.ndarray] = {}
    with np.load(p) as z:
        for k in z.files:
            if k.startswith("langid_"):
                out[k[len("langid_"):]] = np.asarray(z[k], dtype=np.float64)
    return out or None


def lang_identity_extras(
    langid: Optional[dict[str, np.ndarray]],
    *,
    v_shared: np.ndarray,
    present: list[str],
    V: np.ndarray,
    model: str,
    layer: int,
    category: str,
) -> dict:
    """Extra JSON fields comparing v_shared / per-language probes against language-identity directions."""
    if not langid:
        return {
            "langid_note": (
                "language-identity directions not found (no lang_identity.npz in this layer dir); "
                f"run: python scripts/analysis/language_identity.py --model {model} --layers {layer} "
                f"--category {category}"
            )
        }
    extras: dict = {}
    vs = _unit(np.asarray(v_shared, dtype=np.float64))
    Vn = np.asarray(V, dtype=np.float64)
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
    valid: list[str] = []
    for lg in sorted(langid):
        u = np.asarray(langid[lg], dtype=np.float64)
        if u.shape != vs.shape:
            continue  
        u = _unit(u)
        extras[f"cos_vshared_langid_{lg}"] = float(vs @ u)
        if lg in present:
            extras[f"cos_probe_langid_{lg}"] = float(Vn[present.index(lg)] @ u)
        valid.append(lg)
    extras["langid_langs"] = valid
    if len(valid) >= 2:
        extras["langid_cosine_matrix"] = cosine_matrix(
            np.stack([np.asarray(langid[lg], dtype=np.float64) for lg in valid])
        ).tolist()
    return extras


def write_depth_profile(rows: list[dict], *, out_dir: Path) -> None:
    """Write the shared-component depth profile to CSV (data only)."""
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("layer")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "shared_component_depth_profile.csv", index=False)
    print(f"  wrote {(out_dir / 'shared_component_depth_profile.csv').relative_to(REPO_ROOT)}")


def analyse_layer(
    model: str,
    layer: int,
    *,
    langs: Iterable[str] = DEFAULT_LANGS,
    category: str = DEFAULT_CATEGORY,
    out_root: Optional[Path] = None,
) -> Optional[dict]:
    """Decompose one layer; write artefacts; return summary row or None if <2 probes."""
    out_root = out_root or GEOMETRY_DIR
    present, V = load_language_stack(model, layer, langs=langs, category=category)
    if V.shape[0] < 2:
        print(f"[{model} layer {layer}] only {V.shape[0]} probe(s) available for category={category!r}; skipping")
        return None

    cos = cosine_matrix(V)
    decomp = svd_decomposition(V)

    layer_dir = _model_out_dir(out_root, model, category) / f"layer{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    np.save(layer_dir / "v_shared.npy", decomp["v_shared"])
    np.save(layer_dir / "v_residuals.npy", decomp["residuals"])
    payload = {
        "model": model,
        "layer": int(layer),
        "category": category,
        "langs": list(present),
        "cosine_matrix": cos.tolist(),
        "component_spearman_matrix": component_spearman_matrix(V).tolist(),
        "singular_values": decomp["singular_values"].tolist(),
        "projections": dict(zip(present, decomp["projections"].tolist())),
        "residual_norms": dict(zip(present, decomp["residual_norms"].tolist())),
        "variance_explained_top": decomp["variance_explained_top"],
    }
    langid = load_lang_identity(layer_dir)
    payload.update(lang_identity_extras(
        langid, v_shared=decomp["v_shared"], present=list(present), V=V,
        model=model, layer=int(layer), category=category,
    ))
    if langid is None:
        print(f"[{model} layer {layer}] note: no lang_identity.npz — langid cosines skipped "
              "(run scripts/analysis/language_identity.py to add them)")
    with open(layer_dir / "decomposition.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    row = {
        "layer": int(layer),
        "n_langs": len(present),
        "langs": "/".join(present),
        "variance_explained_top": decomp["variance_explained_top"],
    }
    _cs = np.asarray(payload["component_spearman_matrix"], dtype=float)
    _n = _cs.shape[0]
    if _n >= 2:
        row["mean_abs_component_spearman"] = float(
            np.mean([abs(_cs[i][j]) for i in range(_n) for j in range(_n) if i < j]))
    lid_cos = [abs(v) for k, v in payload.items() if k.startswith("cos_vshared_langid_")]
    if lid_cos:
        row["mean_abs_cos_vshared_langid"] = float(np.mean(lid_cos))
    return row


def analyse_model(
    model: str,
    *,
    layers: list[int],
    langs: Iterable[str] = DEFAULT_LANGS,
    category: str = DEFAULT_CATEGORY,
    out_root: Optional[Path] = None,
) -> None:
    out_root = out_root or GEOMETRY_DIR
    summary: list[dict] = []
    for L in layers:
        row = analyse_layer(model, L, langs=langs, category=category, out_root=out_root)
        if row is not None:
            summary.append(row)
    if len(summary) >= 2:
        write_depth_profile(summary, out_dir=_model_out_dir(out_root, model, category))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model id (e.g. gemma-2b, llama-8b).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices (e.g. 14,17,18).")
    g.add_argument(
        "--auto-layers", action="store_true",
        help="Auto-discover layers that have probes for every requested language.",
    )
    p.add_argument(
        "--langs", type=str, default=",".join(DEFAULT_LANGS),
        help="Comma-separated language codes (default: en,pt,es,de,zh).",
    )
    p.add_argument(
        "--category", type=str, default=DEFAULT_CATEGORY,
        help="Probe category (default: generic). Control concepts (formality, politeness) are "
             "namespaced under results/geometry/<model>/<category>/.",
    )
    args = p.parse_args()

    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip())
    if args.auto_layers:
        layers = common_layers(args.model, langs=langs, category=args.category)
        if not layers:
            raise SystemExit(f"No layers with probes for all of {langs} under {args.model} (category={args.category}).")
        print(f"Auto-discovered {len(layers)} common layer(s): {layers}")
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    analyse_model(args.model, layers=layers, langs=langs, category=args.category)


if __name__ == "__main__":
    main()
