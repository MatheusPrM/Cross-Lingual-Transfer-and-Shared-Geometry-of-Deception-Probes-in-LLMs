#!/usr/bin/env python3
"""
Primary transfer statistics. Runs once over all models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

LANGS = ["en", "pt", "es", "de", "zh"]
B_REGISTERED = 2000
SEED_REGISTERED = 42
MARGIN_REGISTERED = 0.05


def auroc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUROC via average ranks. Exactly equals the pairwise
    gt + 0.5*eq definition used by db_projections._auroc (unit-tested). NaN if one class."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0.0], np.cumsum(cnt[:-1], dtype=np.float64)))
    ranks = (starts + (cnt + 1) / 2.0)[inv]
    r_pos = ranks[y == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def make_shared_resample(frame: list[str], b: int, seed: int) -> np.ndarray:
    """ONE (B, len(frame)) matrix of per-scenario draw COUNTS, shared across every cell of
    the model. frame must be sorted for determinism."""
    if sorted(frame) != list(frame):
        raise ValueError("scenario frame must be sorted for a deterministic resample")
    rng = np.random.default_rng(seed)
    n = len(frame)
    counts = np.zeros((b, n), dtype=np.int32)
    for bi in range(b):
        draw = rng.integers(0, n, size=n)
        np.add.at(counts[bi], draw, 1)
    return counts


def bootstrap_grid(
    cells: "dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]",
    train_langs: list[str],
    frame: list[str],
    counts: np.ndarray,
) -> np.ndarray:
    """Replicate AUROCs for every (train, eval) cell from the SHARED resample.
    """
    b = counts.shape[0]
    eval_langs = sorted(cells)
    out = np.full((b, len(train_langs), len(eval_langs)), np.nan)
    for ei, lang in enumerate(eval_langs):
        s_mat, y, scen_idx = cells[lang]
        for bi in range(b):
            w = counts[bi][scen_idx]                     
            keep = w > 0
            if not keep.any():
                continue
            rep = np.repeat(np.flatnonzero(keep), w[keep])  
            y_rep = y[rep]
            for ti in range(len(train_langs)):
                out[bi, ti, ei] = auroc_rank(s_mat[rep, ti], y_rep)
    return out


def native_vs_english(point: np.ndarray, reps: np.ndarray, train_langs: "list[str]",
                      eval_langs: "list[str]") -> "tuple[list[dict], dict]":
    en_ti = train_langs.index("en")
    lang_rows: list[dict] = []
    ds = []
    for el in eval_langs:
        if el == "en":
            continue
        ti, ei = train_langs.index(el), eval_langs.index(el)
        d = reps[:, ti, ei] - reps[:, en_ti, ei]
        ds.append(d)
        v = d[~np.isnan(d)]
        lang_rows.append({
            "eval_lang": el,
            "delta_native_minus_english": float(point[ti, ei] - point[en_ti, ei]),
            "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5)),
            "n_degenerate_reps": int(np.isnan(d).sum()),
        })
    mean_d = np.nanmean(np.stack(ds, axis=1), axis=1)
    v = mean_d[~np.isnan(mean_d)]
    point_mean = float(np.mean([r["delta_native_minus_english"] for r in lang_rows]))
    centered = v - point_mean
    p_two = (1 + np.sum(np.abs(centered) >= abs(point_mean))) / (len(v) + 1)
    model_row = {"mean_delta_native_minus_english": point_mean,
                 "ci_lo": float(np.percentile(v, 2.5)),
                 "ci_hi": float(np.percentile(v, 97.5)),
                 "p_two_sided": float(p_two)}
    return lang_rows, model_row


def holm(pvals: "list[float]") -> "list[float]":
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def noninferiority_pvalue(
    reps: np.ndarray,
    estimate: float,
    margin: float,
) -> float:
    """One-sided null-centered bootstrap test of H0: theta <= -margin."""
    valid = np.asarray(reps, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        raise ValueError("noninferiority test requires at least one finite bootstrap replicate")
    if not np.isfinite(estimate):
        raise ValueError("noninferiority point estimate must be finite")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("noninferiority margin must be a positive finite value")
    theta0 = -float(margin)
    observed_gap = float(estimate) - theta0
    centered = valid - float(estimate)
    return float((1 + np.sum(centered >= observed_gap)) / (valid.size + 1))


def summarize_model(
    model: str,
    point: np.ndarray,          
    reps: np.ndarray,           
    train_langs: list[str],
    eval_langs: list[str],
    margin: float,
) -> "tuple[list[dict], dict]":
    """Per-cell rows + the per-model primary statistic (mean Delta over cross cells)."""
    b = reps.shape[0]
    cell_rows: list[dict] = []
    cross: list[tuple[int, int, int]] = []  
    for ti, tl in enumerate(train_langs):
        for ei, el in enumerate(eval_langs):
            wi = train_langs.index(el)
            r = reps[:, ti, ei]
            row = {
                "model": model, "train_lang": tl, "eval_lang": el,
                "auroc": float(point[ti, ei]),
                "auroc_ci_lo": float(np.nanpercentile(r, 2.5)),
                "auroc_ci_hi": float(np.nanpercentile(r, 97.5)),
                "n_degenerate_reps": int(np.isnan(r).sum()),
            }
            if tl != el:
                d = reps[:, ti, ei] - reps[:, wi, ei]     
                point_delta = float(point[ti, ei] - point[wi, ei])
                row.update({
                    "delta": point_delta,
                    "delta_ci_lo": float(np.nanpercentile(d, 2.5)),
                    "delta_ci_hi": float(np.nanpercentile(d, 97.5)),
                    "p_noninf_cell": noninferiority_pvalue(d, point_delta, margin),
                })
                cross.append((ti, ei, wi))
            cell_rows.append(row)

    d_reps = np.stack([reps[:, ti, ei] - reps[:, wi, ei] for ti, ei, wi in cross], axis=1)
    degen_any = np.isnan(d_reps).any(axis=1)
    n_degen = int(degen_any.sum())
    if n_degen > 0.01 * b:
        raise RuntimeError(
            f"{model}: {n_degen}/{b} replicates have a one-class cell — this is a broken "
            "cell, not sampling noise; inspect the caches before quoting any transfer statistic.")
    mean_d = np.nanmean(d_reps, axis=1)                  
    point_mean_d = float(np.mean([point[ti, ei] - point[wi, ei] for ti, ei, wi in cross]))
    valid = mean_d[~np.isnan(mean_d)]
    model_row = {
        "model": model,
        "mean_delta": point_mean_d,
        "mean_delta_ci_lo": float(np.percentile(valid, 2.5)),
        "mean_delta_ci_hi": float(np.percentile(valid, 97.5)),
        "p_noninf": noninferiority_pvalue(valid, point_mean_d, margin),
        "n_cross_cells": len(cross),
        "n_degenerate_reps": n_degen,
        "noninferiority_margin": margin,
    }
    return cell_rows, model_row


def _load_probe_dirs(model: str, layer: int, category: str) -> "dict[str, np.ndarray]":
    """Unit-normalized effective directions — the EXACT convention of the descriptive grid
    (db_projections._load_probe_dirs). torch imported here only."""
    from project.detectors import LogisticRegressionDetector
    from scripts.analysis.load_probes import probe_path

    dirs = {}
    for lang in LANGS:
        pp = probe_path(model, lang, layer, category=category, require_category_path=True)
        if not pp.exists():
            raise FileNotFoundError(f"{model}: missing probe for {lang}: {pp}")
        det = LogisticRegressionDetector.load(pp)
        v = det.effective_direction(layer).float().numpy()
        dirs[lang] = v / np.linalg.norm(v)
    return dirs


def _load_cells(model: str, layer: int, cache_root: Path):
    cells = {}
    for lang in LANGS:
        npz = cache_root / model / "deceptionbench" / f"{lang}_layer{layer}.npz"
        if not npz.exists():
            raise FileNotFoundError(f"{model}: missing cache {npz}")
        with np.load(npz) as z:
            if "scenario_id" not in z:
                raise KeyError(f"{npz}: no per-row scenario_id — regenerate the cache "
                               "")
            cells[lang] = (z["acts"].astype(np.float64), z["labels"].astype(int),
                           np.array([str(s) for s in z["scenario_id"]]))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "results" / "activation_cache")
    ap.add_argument("--grid-root", type=Path, default=_REPO_ROOT / "analysis" / "db_projections")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "transfer_stats")
    ap.add_argument("--category", default="generic")
    ap.add_argument("--b", type=int, default=B_REGISTERED)
    ap.add_argument("--seed", type=int, default=SEED_REGISTERED)
    ap.add_argument("--margin", type=float, default=MARGIN_REGISTERED)
    ap.add_argument("--skip-grid-check", action="store_true",
                    help="ONLY for synthetic/debug data — the paper run must not skip it")
    args = ap.parse_args()

    if (args.b, args.seed, args.margin) != (B_REGISTERED, SEED_REGISTERED, MARGIN_REGISTERED):
        print(f"WARNING: non-default parameters (B={args.b} seed={args.seed} "
              f"margin={args.margin}) — the paper uses B={B_REGISTERED} "
              f"seed={SEED_REGISTERED} margin={MARGIN_REGISTERED}.", file=sys.stderr)

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "frozen_layers", _REPO_ROOT / "project" / "frozen_layers.py")
    _fl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fl)
    load_frozen_layer = _fl.load_frozen_layer 

    all_cells, all_models, all_natives, draws, meta_models = [], [], [], {}, {}
    for model in args.models:
        layer = load_frozen_layer(model)
        print(f"[{model}] layer {layer}: loading caches + probes...", flush=True)
        raw = _load_cells(model, layer, args.cache_root)
        dirs = _load_probe_dirs(model, layer, args.category)

        frame = sorted(set().union(*[set(scen) for _, _, scen in raw.values()]))
        f_idx = {s: i for i, s in enumerate(frame)}
        cells = {}
        point = np.full((len(LANGS), len(LANGS)), np.nan)
        for ei, lang in enumerate(sorted(raw)):
            acts, y, scen = raw[lang]
            s_mat = np.stack([acts @ dirs[t] for t in LANGS], axis=1)
            cells[lang] = (s_mat, y, np.array([f_idx[s] for s in scen]))
            for ti, t in enumerate(LANGS):
                point[ti, ei] = auroc_rank(s_mat[:, ti], y)
        eval_langs = sorted(raw)

        if not args.skip_grid_check:
            gp = args.grid_root / model / "deceptionbench" / "full_transfer_grid.csv"
            g = pd.read_csv(gp).set_index(["train_lang", "eval_lang"])["auroc"]
            worst = max(abs(point[LANGS.index(t), eval_langs.index(e)] - g[(t, e)])
                        for t in LANGS for e in eval_langs)
            if worst > 1e-8:
                raise RuntimeError(f"{model}: point AUROCs do not reproduce "
                                   f"full_transfer_grid.csv (max |diff| = {worst:.2e}) — "
                                   "probe/cache drift between grid and stats.")
            print(f"  grid cross-check OK (max |diff| = {worst:.2e})", flush=True)

        counts = make_shared_resample(frame, args.b, args.seed)
        reps = bootstrap_grid(cells, LANGS, frame, counts)
        cell_rows, model_row = summarize_model(model, point, reps, LANGS, eval_langs,
                                               args.margin)
        ne_lang_rows, ne_model = native_vs_english(point, reps, LANGS, eval_langs)
        for r in ne_lang_rows:
            all_natives.append({"model": model, "scope": "per_lang", **r})
        all_natives.append({"model": model, "scope": "model_mean", **ne_model})
        all_cells.extend(cell_rows)
        all_models.append(model_row)
        draws[f"{model}_reps"] = reps.astype(np.float32)
        draws[f"{model}_counts"] = counts.astype(np.int16)
        draws[f"{model}_frame"] = np.array(frame)
        meta_models[model] = {
            "layer": int(layer), "n_scenarios": len(frame),
            "frame_sha256": hashlib.sha256("\n".join(frame).encode()).hexdigest(),
            "n_items": {l: int(len(raw[l][1])) for l in eval_langs},
        }
        print(f"  mean Delta = {model_row['mean_delta']:+.4f} "
              f"[{model_row['mean_delta_ci_lo']:+.4f}, {model_row['mean_delta_ci_hi']:+.4f}] "
              f"p_noninf = {model_row['p_noninf']:.4g}", flush=True)

    padj = holm([m["p_noninf"] for m in all_models])
    for m, pa in zip(all_models, padj):
        m["p_noninf_holm"] = pa
        m["noninferior_at_05_holm"] = bool(pa < 0.05)
    ne_means = [r for r in all_natives if r["scope"] == "model_mean"]
    for r, pa in zip(ne_means, holm([r["p_two_sided"] for r in ne_means])):
        r["p_two_sided_holm"] = pa

    args.out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_cells).to_csv(args.out_root / "cells.csv", index=False)
    pd.DataFrame(all_models).to_csv(args.out_root / "models.csv", index=False)
    pd.DataFrame(all_natives).to_csv(args.out_root / "natives.csv", index=False)
    np.savez_compressed(args.out_root / "bootstrap_draws.npz", **draws)
    (args.out_root / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "B": args.b, "seed": args.seed, "margin": args.margin,
        "registered": (args.b, args.seed, args.margin) ==
                      (B_REGISTERED, SEED_REGISTERED, MARGIN_REGISTERED),
        "grid_check": "skipped" if args.skip_grid_check else "passed",
        "description": "Primary analysis of transfer learning across models and languages.",
        "ci": "percentile 95%, unadjusted", "multiplicity": "Holm over models",
        "noninferiority_note": "one-sided null-centered scenario-clustered bootstrap test "
                               "of H0: Delta <= -margin; with theta0=-margin, "
                               "p=(1+#{Delta*-Dhat >= Dhat-theta0})/(B+1); Holm adjustment "
                               "over the five model-level tests",
        "native_vs_english": "per-lang paired deltas + per-model mean, two-sided "
                             "approximate null-centered bootstrap p, Holm separately over "
                             "the 5 models; percentile CIs remain unadjusted",
        "models": meta_models,
    }, indent=2))
    print(f"Saved: {args.out_root}/cells.csv + models.csv + bootstrap_draws.npz + meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
