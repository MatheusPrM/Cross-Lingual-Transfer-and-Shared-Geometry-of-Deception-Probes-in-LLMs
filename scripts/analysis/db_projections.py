#!/usr/bin/env python3
"""
Deception S5 projections + v_shared sufficiency + leave-one-language-out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _load_probe_dirs(model: str, layer: int, langs: list[str], category: str) -> dict[str, np.ndarray]:
    """Per-language probe directions (needs torch); {} when torch/probes unavailable."""
    try:
        from project.detectors import LogisticRegressionDetector
        from scripts.analysis.load_probes import probe_path
    except Exception as e:
        raise SystemExit(f"probe loading unavailable ({e}); probes are "
                         f"required for the transfer grid and LOLO")
    dirs = {}
    for lang in langs:
        pp = probe_path(model, lang, layer, category=category, require_category_path=True)  
        if pp.exists():
            det = LogisticRegressionDetector.load(pp)
            dirs[lang] = _unit(det.effective_direction(layer).float().numpy())
    return dirs


def _top_pc(stack: np.ndarray, align_to: np.ndarray | None = None) -> np.ndarray:
    _, _, vt = np.linalg.svd(stack, full_matrices=False)
    v = vt[0]
    ref = align_to if align_to is not None else stack.mean(axis=0)
    return _unit(v if float(v @ ref) >= 0 else -v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--family", default="deceptionbench")
    p.add_argument("--langs", nargs="+", default=["en", "pt", "es", "de", "zh"])
    p.add_argument("--category", default="generic", help="probe category for LOLO directions")
    p.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "results" / "activation_cache")
    p.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "db_projections")
    p.add_argument("--frozen-root", type=Path, default=_REPO_ROOT / "results" / "frozen_layers")
    p.add_argument("--geometry-root", type=Path, default=_REPO_ROOT / "results" / "geometry")
    args = p.parse_args()

    layer = int(json.loads((args.frozen_root / args.model / "chosen_layer.json").read_text())["chosen_layer"])
    geo = args.geometry_root / args.model / f"layer{layer}"
    vs_path = geo / "v_shared.npy"
    if not vs_path.exists():
        print(f"no v_shared at {vs_path}; run the geometry/build stage first", file=sys.stderr)
        return 1
    v_shared = _unit(np.load(vs_path).astype(np.float64))

    u_lang = None
    lid_path = geo / "lang_identity.npz"
    if lid_path.exists():
        with np.load(lid_path) as z:
            lid = np.stack([z[k] for k in z.files if k.startswith("langid_")]).astype(np.float64)
        u_lang = _top_pc(lid - lid.mean(axis=0, keepdims=True))

    probe_dirs = _load_probe_dirs(args.model, layer, args.langs, args.category)
    missing_dirs = [l for l in args.langs if l not in probe_dirs]
    if missing_dirs or len(args.langs) != 5:
        raise SystemExit(
            f"transfer/LOLO/geometry require all five generic probe "
            f"directions; missing: {missing_dirs or 'language list != 5'}")

    out_dir = args.out_root / args.model / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_rows, summary = [], []
    for lang in args.langs:
        npz = args.cache_root / args.model / args.family / f"{lang}_layer{layer}.npz"
        if not npz.exists():
            raise SystemExit(f"missing activation cache: {npz}")
        with np.load(npz) as z:
            acts = z["acts"].astype(np.float64)
            labels = z["labels"].astype(int)
        pv = acts @ v_shared
        pl = acts @ u_lang if u_lang is not None else np.full(len(acts), np.nan)
        for i in range(len(acts)):
            proj_rows.append({"eval_lang": lang, "item_idx": i, "label": int(labels[i]),
                              "proj_vshared": float(pv[i]), "proj_langaxis": float(pl[i])})
        a_vs = _auroc(pv[labels == 1], pv[labels == 0])
        a_lolo = float("nan")
        others = [l for l in probe_dirs if l != lang]
        if len(others) >= 2:
            v_lolo = _top_pc(np.stack([probe_dirs[l] for l in others]).astype(np.float64))
            plo = acts @ v_lolo
            a_lolo = _auroc(plo[labels == 1], plo[labels == 0])
        a_lid = _auroc(pl[labels == 1], pl[labels == 0]) if u_lang is not None else float("nan")
        summary.append({"model": args.model, "family": args.family, "lang": lang, "layer": layer,
                        "n": int(len(labels)), "n_pos": int(labels.sum()),
                        "auroc_vshared": a_vs, "auroc_vshared_lolo": a_lolo, "auroc_langaxis": a_lid})
        print(f"  {lang}: v_shared AUROC={a_vs:.3f}  LOLO={a_lolo:.3f}  langaxis(control)={a_lid:.3f}", flush=True)

    pd.DataFrame(proj_rows).to_csv(out_dir / "projections.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

    if probe_dirs:
        grid_rows = []
        for eval_lang in args.langs:
            npz = args.cache_root / args.model / args.family / f"{eval_lang}_layer{layer}.npz"
            if not npz.exists():
                continue
            with np.load(npz) as z:
                acts = z["acts"].astype(np.float64)
                labels = z["labels"].astype(int)
            for train_lang, w in probe_dirs.items():
                s = acts @ w.astype(np.float64)
                grid_rows.append({"model": args.model, "family": args.family, "layer": layer,
                                  "train_lang": train_lang, "eval_lang": eval_lang,
                                  "auroc": _auroc(s[labels == 1], s[labels == 0]), "n": int(len(labels))})
        if grid_rows:
            pd.DataFrame(grid_rows).to_csv(out_dir / "full_transfer_grid.csv", index=False)
            print(f"  full_transfer_grid.csv written ({len(grid_rows)} cells)", flush=True)

    print(f"Saved: {out_dir}/projections.csv + summary.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
