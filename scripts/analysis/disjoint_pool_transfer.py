#!/usr/bin/env python3
"""
Disjoint-pool robustness analysis.
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

from scripts.analysis.disjoint_pool_common import (DIRS_NPZ_NAME,
                                                   FIT_META_NAME, OUT_REL,
                                                   cosine_contrast_rows,
                                                   near_duplicate_scan,
                                                   summarize_cosine_contrast,
                                                   unit_rows)
from scripts.analysis.transfer_stats import (B_REGISTERED, LANGS,
                                             MARGIN_REGISTERED,
                                             SEED_REGISTERED, auroc_rank,
                                             bootstrap_grid,
                                             make_shared_resample,
                                             noninferiority_pvalue)

# --------------------------------------------------------------- pure core
def point_grid(cells: dict, dirs: "dict[str, np.ndarray]") -> np.ndarray:
    """(n_train, n_eval) tie-aware AUROCs; train order = LANGS, eval sorted."""
    eval_langs = sorted(cells)
    g = np.full((len(LANGS), len(eval_langs)), np.nan)
    for ei, el in enumerate(eval_langs):
        acts, y, _ = cells[el]
        for ti, tl in enumerate(LANGS):
            g[ti, ei] = auroc_rank(acts @ dirs[tl], y)
    return g


def verify_grid_parity(point: np.ndarray, grid_csv: Path, eval_langs: "list[str]",
                       tol: float = 1e-8) -> float:
    """The full-pool probes pushed through THIS script's scoring path must
    reproduce the descriptive grid before any half-probe number is trusted."""
    g = pd.read_csv(grid_csv).set_index(["train_lang", "eval_lang"])["auroc"]
    worst = max(abs(point[LANGS.index(t), eval_langs.index(e)] - g[(t, e)])
                for t in LANGS for e in eval_langs)
    if worst > tol:
        raise RuntimeError(f"parity FAILED vs {grid_csv} (max |diff| = {worst:.2e}) — "
                           "scoring-path drift; no disjoint-pool number is valid.")
    return worst


def half_delta_stats(point: np.ndarray, reps: np.ndarray, cols: "list[tuple[str, int]]",
                     eval_langs: "list[str]", margin: float) -> "dict[str, dict]":
    ci = {c: i for i, c in enumerate(cols)}
    plan = {}  
    for name, cross_h, base_h in (("matched_half_a", 0, 0),
                                  ("matched_half_b", 1, 1),
                                  ("disjoint", 1, 0)):
        plan[name] = [(ci[(t, cross_h)], ci[(el, base_h)], eval_langs.index(el))
                      for t in LANGS for el in eval_langs if t != el]
    out = {}
    kept = {}
    for name, idx in plan.items():
        d_point = np.array([point[a, e] - point[b, e] for a, b, e in idx])
        d_reps = np.stack([reps[:, a, e] - reps[:, b, e] for a, b, e in idx], axis=1)
        mean_reps = np.nanmean(d_reps, axis=1)
        valid = mean_reps[~np.isnan(mean_reps)]
        pm = float(d_point.mean())
        out[name] = {
            "mean_delta": pm,
            "mean_delta_ci_lo": float(np.percentile(valid, 2.5)),
            "mean_delta_ci_hi": float(np.percentile(valid, 97.5)),
            "p_noninf": noninferiority_pvalue(valid, pm, margin),
            "n_delta_terms": len(idx),
        }
        kept[name] = (pm, mean_reps)
    paired = kept["disjoint"][1] - kept["matched_half_a"][1]
    paired = paired[~np.isnan(paired)]
    out["difference"] = {
        "mean_delta": float(kept["disjoint"][0] - kept["matched_half_a"][0]),
        "mean_delta_ci_lo": float(np.percentile(paired, 2.5)),
        "mean_delta_ci_hi": float(np.percentile(paired, 97.5)),
    }
    return out


def half_swap_rows(model: str, seed: int, point: np.ndarray,
                   cols: "list[tuple[str, int]]", eval_langs: "list[str]") -> "list[dict]":
    """A(T_A->L) - A(T_B->L) per cross cell: the item-resampling noise floor."""
    ci = {c: i for i, c in enumerate(cols)}
    rows = []
    for t in LANGS:
        for el in eval_langs:
            if t == el:
                continue
            ei = eval_langs.index(el)
            rows.append({
                "model": model, "seed": seed, "train_lang": t, "eval_lang": el,
                "auroc_half_a": float(point[ci[(t, 0)], ei]),
                "auroc_half_b": float(point[ci[(t, 1)], ei]),
                "half_swap_abs_diff": float(abs(point[ci[(t, 0)], ei]
                                                - point[ci[(t, 1)], ei])),
            })
    return rows


def _load_cells(model: str, layer: int, cache_root: Path):
    from scripts.analysis.transfer_stats import _load_cells as tl
    return tl(model, layer, cache_root)


def _load_full_dirs(model: str, layer: int, category: str):
    from scripts.analysis.transfer_stats import _load_probe_dirs
    return _load_probe_dirs(model, layer, category)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path,
                    default=_REPO_ROOT / "results" / "activation_cache")
    ap.add_argument("--grid-root", type=Path,
                    default=_REPO_ROOT / "analysis" / "db_projections")
    ap.add_argument("--in-root", type=Path, default=_REPO_ROOT / OUT_REL,
                    help="root holding <model>/half_directions.npz from the GPU "
                         "stage (pass analysis/disjoint_pool to analyze the "
                         "shipped snapshot)")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / OUT_REL)
    ap.add_argument("--category", default="generic")
    ap.add_argument("--b", type=int, default=B_REGISTERED)
    ap.add_argument("--boot-seed", type=int, default=SEED_REGISTERED)
    ap.add_argument("--margin", type=float, default=MARGIN_REGISTERED)
    ap.add_argument("--skip-parity", action="store_true",
                    help="ONLY for synthetic/debug data")
    args = ap.parse_args()

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "frozen_layers", _REPO_ROOT / "project" / "frozen_layers.py")
    _fl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fl)

    cos_rows, transfer_rows, swap_rows = [], [], []
    meta_models, fact_universe = {}, {}
    for model in args.models:
        layer = _fl.load_frozen_layer(model)
        npz_path = args.in_root / model / DIRS_NPZ_NAME
        if not npz_path.exists():
            raise FileNotFoundError(f"{model}: run disjoint_pool_probes.py "
                                    f"first — missing {npz_path}")
        from scripts.analysis.disjoint_pool_common import validate_fit_artifact
        validate_fit_artifact(args.in_root / model)
        fit_meta = json.loads((args.in_root / model / FIT_META_NAME).read_text())
        if int(fit_meta["layer"]) != int(layer):
            raise RuntimeError(f"{model}: fit_meta layer {fit_meta['layer']} != frozen "
                               f"layer {layer} — stale half-probe artifact.")
        seeds = [int(s) for s in fit_meta["seeds"]]
        with np.load(npz_path) as z:
            dirs_by_lang = {l: z[l].astype(np.float64) for l in z.files}
        if sorted(dirs_by_lang) != sorted(LANGS):
            raise RuntimeError(f"{model}: half directions cover {sorted(dirs_by_lang)}, "
                               f"expected {sorted(LANGS)}")

        print(f"[{model}] layer {layer}: caches + parity gate...", flush=True)
        cells = _load_cells(model, layer, args.cache_root)
        eval_langs = sorted(cells)
        parity_val = None
        if not args.skip_parity:
            full = _load_full_dirs(model, layer, args.category)
            parity_val = verify_grid_parity(
                point_grid(cells, full),
                args.grid_root / model / "deceptionbench" / "full_transfer_grid.csv",
                eval_langs)
            print(f"  parity OK (max |diff| = {parity_val:.2e})", flush=True)
        cache_shas = {
            lang: hashlib.sha256(
                (args.cache_root / model / "deceptionbench"
                 / f"{lang}_layer{layer}.npz").read_bytes()).hexdigest()
            for lang in eval_langs}

        cos_rows.extend(cosine_contrast_rows(model, dirs_by_lang, seeds))

        frame = sorted(set().union(*[set(s) for _, _, s in cells.values()]))
        f_idx = {s: i for i, s in enumerate(frame)}
        counts = make_shared_resample(frame, args.b, args.boot_seed)
        cols = [(l, h) for l in LANGS for h in (0, 1)]
        for seed in seeds:
            si = seeds.index(seed)
            half_dirs = {(l, h): unit_rows(dirs_by_lang[l][si, h])
                         for l in LANGS for h in (0, 1)}
            grid_cells = {}
            pt = np.full((len(cols), len(eval_langs)), np.nan)
            for ei, el in enumerate(eval_langs):
                acts, y, scen = cells[el]
                s_mat = np.stack([acts @ half_dirs[c] for c in cols], axis=1)
                grid_cells[el] = (s_mat, y, np.array([f_idx[s] for s in scen]))
                for cix in range(len(cols)):
                    pt[cix, ei] = auroc_rank(s_mat[:, cix], y)
            reps = bootstrap_grid(grid_cells, cols, frame, counts)
            stats = half_delta_stats(pt, reps, cols, eval_langs, args.margin)
            for cond in ("matched_half_a", "matched_half_b", "disjoint", "difference"):
                transfer_rows.append({"model": model, "seed": seed,
                                      "condition": cond, **stats[cond]})
            swap_rows.extend(half_swap_rows(model, seed, pt, cols, eval_langs))
            d = stats["difference"]["mean_delta"]
            print(f"  seed {seed}: mean Delta A {stats['matched_half_a']['mean_delta']:+.4f}"
                  f" B {stats['matched_half_b']['mean_delta']:+.4f}"
                  f" disjoint {stats['disjoint']['mean_delta']:+.4f}"
                  f" (half-swap diff {d:+.4f})", flush=True)
        meta_models[model] = {"layer": int(layer), "seeds": seeds,
                              "n_scenarios": len(frame),
                              "fit_meta_created": fit_meta.get("created_utc"),
                              "groups_sha256": fit_meta.get("groups_sha256"),
                              "fitter_parity_cos": fit_meta.get("fitter_parity_cos"),
                              "grid_parity_max_abs_diff": parity_val,
                              "input_half_dirs_sha256": fit_meta.get("dirs_npz_sha256"),
                              "input_cache_sha256": cache_shas}
        fact_universe[model] = (
            fit_meta.get("used_statements")
            or {"en": fit_meta.get("used_statements_en")},
            fit_meta.get("duplicate_groups"))

    tdf = pd.DataFrame(transfer_rows)
    for (model, cond), g in tdf.groupby(["model", "condition"]):
        v = g["mean_delta"].to_numpy(dtype=float)
        transfer_rows.append({
            "model": model, "seed": -1, "condition": f"{cond}_across_seeds",
            "mean_delta": float(v.mean()),
            "mean_delta_seed_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "mean_delta_seed_min": float(v.min()),
            "mean_delta_seed_max": float(v.max()),
            "per_seed_mean_deltas": ";".join(f"{x:.6f}" for x in v),
        })

    args.out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cos_rows).to_csv(args.out_root / "cosine_contrast.csv", index=False)
    pd.DataFrame(summarize_cosine_contrast(cos_rows)).to_csv(
        args.out_root / "cosine_summary.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(args.out_root / "transfer_summary.csv", index=False)
    pd.DataFrame(swap_rows).to_csv(args.out_root / "half_swap_stability.csv", index=False)

    shas = {m["groups_sha256"] for m in meta_models.values()}
    if len(shas) != 1:
        raise RuntimeError(f"models disagree on the fact-universe grouping: {shas}")
    stmts_by_lang, groups = next(iter(fact_universe.values()))
    if stmts_by_lang and groups:
        any_seeds = next(iter(meta_models.values()))["seeds"]
        scan_rows = []
        for lang, stmts in sorted(stmts_by_lang.items()):
            if not stmts:
                continue
            for row in near_duplicate_scan(stmts, any_seeds, groups=groups):
                scan_rows.append({"lang": lang, **row})
        scan = pd.DataFrame(scan_rows)
        if int(scan["n_same_group_split"].sum()) != 0:
            raise RuntimeError("duplicate-equivalent facts span halves — the "
                               "'disjoint row IDs' label would be false; refit "
                               "with the group-aware split.")
        scan.to_csv(args.out_root / "near_duplicate_scan.csv", index=False)

    code_shas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (_SCRIPT_DIR / "disjoint_pool_transfer.py",
                           _SCRIPT_DIR / "disjoint_pool_common.py",
                           _SCRIPT_DIR / "transfer_stats.py")}
    (args.out_root / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": code_shas,
        "B": args.b, "boot_seed": args.boot_seed, "margin": args.margin,
        "registered_boot_params": (args.b, args.boot_seed, args.margin) ==
                                  (B_REGISTERED, SEED_REGISTERED, MARGIN_REGISTERED),
        "parity_gate": "skipped" if args.skip_parity else "passed",
        "design": {
            "fact_universe": "306 label==1 statements (the generic loader's exact "
                             "filter); duplicate-equivalent facts grouped so no "
                             "group spans halves (~153 facts per half)",
            "matched_half_a": "all probes trained on fact half A (same facts across "
                              "languages, half-size pool)",
            "matched_half_b": "all probes trained on fact half B",
            "disjoint": "cross probes from half B against half-A baselines — zero "
                        "shared training facts between any compared pair",
            "structural_note": "D_AB + D_BA = M_A + M_B is an algebraic identity for "
                               "any scorer, so a symmetrized matched-vs-disjoint "
                               "transfer contrast is tautologically zero. The eval "
                               "set's externality means each marginal AUROC cannot be "
                               "item-inflated. Train-item sharing CAN inflate "
                               "cross-language probe SIMILARITY — readout 1 (cosine "
                               "contrast) is the primary test; the 'difference' "
                               "condition is exactly a B-vs-A half-swap "
                               "stability/noise diagnostic, never overlap evidence.",
        },
        "interpretation_prespecified": {
            "primary": "DESCRIPTIVE ONLY: the cosine contrast reported as all "
                       "per-seed values plus mean, SD and range (the ten seeds "
                       "are overlapping partitions of one fact pool, not "
                       "independent datasets; no CI-like interval is computed "
                       "and no binary verdict is emitted)",
            "reference_scale": "the within-language split-half cosine decrement "
                               "already on record (analysis/probe_reliability: "
                               "1 - cos ~= 0.04-0.08) is the "
                               "natural yardstick for reading the contrast's "
                               "magnitude — context, not a pass/fail test",
            "note": "an earlier -0.05 cosine margin was removed: it 'mirrored' "
                    "the AUROC noninferiority margin, but the metrics have "
                    "unrelated scales. A zero-crossing seed interval is NOT "
                    "evidence of equivalence.",
            "disjointness_claim": "disjoint row IDs with duplicate-equivalent "
                                  "facts grouped (multilingual union + frozen "
                                  "manual list) — not absolute semantic "
                                  "independence",
            "fixed_before_results": True,
        },
        "models": meta_models,
    }, indent=2))
    print(f"Saved: {args.out_root}/cosine_contrast.csv + cosine_summary.csv + "
          f"transfer_summary.csv + half_swap_stability.csv + near_duplicate_scan.csv "
          f"+ meta.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
