#!/usr/bin/env python3
"""
Geometry battery complements: SB-corrected cosine uncertainty from the
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

LANGS = ["en", "pt", "es", "de", "zh"]
EXPECTED_RELIABILITY_SEEDS = 10
EXPECTED_CKA_ITEMS = 900
COSINE_POPULATION = (
    "full-data REPE instructed-deception probe directions; dispersion over 10 grouped "
    "split-half reliability seeds"
)
CKA_POPULATION = "full 900-item DeceptionBench corpus (admitted + nonadmitted)"

_spec = importlib.util.spec_from_file_location("stats_utils", _REPO_ROOT / "project" / "stats_utils.py")
_su = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_su)
spearman_brown = _su.spearman_brown


def corrected_cosine_dispersion(cos_raw: float, seeds_a: "list[float]",
                                seeds_b: "list[float]") -> "dict[str, float]":
    """Per-seed corrected cosines; mean ± sd = dispersion across split seeds."""
    if len(seeds_a) != EXPECTED_RELIABILITY_SEEDS \
            or len(seeds_b) != EXPECTED_RELIABILITY_SEEDS:
        raise ValueError(
            "corrected-cosine dispersion requires exactly "
            f"{EXPECTED_RELIABILITY_SEEDS} seeds per language; "
            f"got {len(seeds_a)} and {len(seeds_b)}"
        )
    vals = []
    for seed_idx, (ra, rb) in enumerate(zip(seeds_a, seeds_b)):
        if not (np.isfinite(ra) and np.isfinite(rb)):
            raise ValueError(f"non-finite reliability at seed index {seed_idx}: {ra}, {rb}")
        sa, sb = spearman_brown(ra), spearman_brown(rb)
        if not (np.isfinite(sa) and np.isfinite(sb) and sa > 0 and sb > 0):
            raise ValueError(
                f"invalid Spearman-Brown reliability at seed index {seed_idx}: "
                f"raw=({ra}, {rb}), corrected=({sa}, {sb})"
            )
        vals.append(cos_raw / np.sqrt(sa * sb))
    v = np.array(vals)
    return {"corrected_mean": float(v.mean()),
            "corrected_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "corrected_min": float(v.min()), "corrected_max": float(v.max()),
            "n_seeds": len(v)}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0] or x.shape[0] < 3:
        return float("nan")
    return linear_cka_from_grams(centered_linear_gram(x), centered_linear_gram(y))


def centered_linear_gram(x: np.ndarray) -> np.ndarray:
    """Return X_c X_c^T for reusable, feature-centered linear CKA."""
    if x.ndim != 2:
        raise ValueError(f"CKA activations must be a 2-D matrix, got shape {x.shape}")
    xc = x - x.mean(axis=0, keepdims=True)
    return xc @ xc.T


def linear_cka_from_grams(kx: np.ndarray, ky: np.ndarray) -> float:
    """Linear CKA from corresponding centered sample Gram matrices."""
    if kx.shape != ky.shape or kx.ndim != 2 or kx.shape[0] != kx.shape[1] \
            or kx.shape[0] < 3:
        return float("nan")
    num = float(np.sum(kx * ky))
    den = float(np.sqrt(np.sum(kx * kx)) * np.sqrt(np.sum(ky * ky)))
    return float(num / den) if den > 0 else float("nan")


def _assert_unique_identities(ids: "list[tuple]", label: str) -> None:
    """Fail closed when a purported parallel population contains duplicate stable IDs."""
    seen = set()
    duplicates = []
    for identity in ids:
        if identity in seen and identity not in duplicates:
            duplicates.append(identity)
        seen.add(identity)
    if duplicates:
        raise ValueError(
            f"{label} contains {len(ids) - len(seen)} duplicate "
            f"(scenario_id, condition) identities; examples: {duplicates[:3]}"
        )


def align_by_identity(ids_a: "list[tuple]", ids_b: "list[tuple]") -> "tuple[list[int], list[int]]":
    """Row indices aligning two caches on shared (scenario_id, condition) identities."""
    _assert_unique_identities(ids_a, "left cache")
    _assert_unique_identities(ids_b, "right cache")
    pos_b = {k: i for i, k in enumerate(ids_b)}
    ia, ib = [], []
    for i, k in enumerate(ids_a):
        j = pos_b.get(k)
        if j is not None:
            ia.append(i)
            ib.append(j)
    return ia, ib


def load_full_corpus_cache(path: Path) -> "tuple[np.ndarray, list[tuple], int, int]":
    """Load admitted + non-admitted activations and IDs, enforcing the 900-item contract."""
    required = {
        "acts", "scenario_id", "condition",
        "nonadmitted_acts", "nonadmitted_scenario_id", "nonadmitted_condition",
    }
    with np.load(path) as z:
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"{path}: missing full-corpus cache keys: {missing}")

        admitted_acts = z["acts"].astype(np.float64)
        admitted_ids = list(zip(
            [str(s) for s in z["scenario_id"]],
            [str(c) for c in z["condition"]],
        ))
        nonadmitted_acts = z["nonadmitted_acts"].astype(np.float64)
        nonadmitted_ids = list(zip(
            [str(s) for s in z["nonadmitted_scenario_id"]],
            [str(c) for c in z["nonadmitted_condition"]],
        ))

    if admitted_acts.shape[0] != len(admitted_ids):
        raise ValueError(
            f"{path}: admitted activation/identity rows differ "
            f"({admitted_acts.shape[0]} != {len(admitted_ids)})"
        )
    if nonadmitted_acts.shape[0] != len(nonadmitted_ids):
        raise ValueError(
            f"{path}: nonadmitted activation/identity rows differ "
            f"({nonadmitted_acts.shape[0]} != {len(nonadmitted_ids)})"
        )
    if admitted_acts.ndim != 2 or nonadmitted_acts.ndim != 2 \
            or admitted_acts.shape[1] != nonadmitted_acts.shape[1]:
        raise ValueError(
            f"{path}: incompatible admitted/nonadmitted activation shapes "
            f"{admitted_acts.shape} and {nonadmitted_acts.shape}"
        )
    if not (np.isfinite(admitted_acts).all() and np.isfinite(nonadmitted_acts).all()):
        raise ValueError(f"{path}: full-corpus activations contain non-finite values")

    acts = np.concatenate([admitted_acts, nonadmitted_acts], axis=0)
    ids = admitted_ids + nonadmitted_ids
    _assert_unique_identities(ids, str(path))
    if len(ids) != EXPECTED_CKA_ITEMS:
        raise ValueError(
            f"{path}: complete CKA population must contain exactly "
            f"{EXPECTED_CKA_ITEMS} items, found {len(ids)} "
            f"({len(admitted_ids)} admitted + {len(nonadmitted_ids)} nonadmitted)"
        )
    return acts, ids, len(admitted_ids), len(nonadmitted_ids)


def _parse_reliability_seeds(raw: object, lang: str) -> "list[float]":
    """Parse and validate the protocol-fixed ten archived split-half cosines."""
    parts = str(raw).split(";")
    if len(parts) != EXPECTED_RELIABILITY_SEEDS:
        raise ValueError(
            f"{lang}: expected exactly {EXPECTED_RELIABILITY_SEEDS} reliability seeds, "
            f"found {len(parts)}"
        )
    try:
        values = [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(f"{lang}: malformed per_seed_cosines") from exc
    if not all(np.isfinite(x) for x in values):
        raise ValueError(f"{lang}: non-finite value in per_seed_cosines")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reliability-root", type=Path, default=_REPO_ROOT / "analysis" / "probe_reliability")
    ap.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "results" / "activation_cache")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "geometry_battery")
    args = ap.parse_args()

    _fspec = importlib.util.spec_from_file_location("frozen_layers", _REPO_ROOT / "project" / "frozen_layers.py")
    _fl = importlib.util.module_from_spec(_fspec)
    _fspec.loader.exec_module(_fl)

    args.out_root.mkdir(parents=True, exist_ok=True)
    meta = {}
    for model in args.models:
        layer = _fl.load_frozen_layer(model)
        rel = pd.read_csv(args.reliability_root / model / "generic" / "reliability.csv")
        cos = pd.read_csv(args.reliability_root / model / "generic" / "corrected_cosines.csv")
        if rel["lang"].duplicated().any():
            duplicates = rel.loc[rel["lang"].duplicated(keep=False), "lang"].tolist()
            raise ValueError(f"{model}: duplicate reliability rows for languages {duplicates}")
        if "n_splits" in rel.columns \
                and not (rel["n_splits"] == EXPECTED_RELIABILITY_SEEDS).all():
            bad = rel.loc[rel["n_splits"] != EXPECTED_RELIABILITY_SEEDS,
                          ["lang", "n_splits"]].to_dict("records")
            raise ValueError(
                f"{model}: reliability.csv does not attest exactly "
                f"{EXPECTED_RELIABILITY_SEEDS} splits: {bad}"
            )
        seeds = {r.lang: _parse_reliability_seeds(r.per_seed_cosines, r.lang)
                 for r in rel.itertuples()}
        missing_seed_langs = sorted(set(LANGS) - set(seeds))
        if missing_seed_langs:
            raise ValueError(f"{model}: missing reliability seeds for {missing_seed_langs}")
        rows = []
        for r in cos.itertuples():
            d = corrected_cosine_dispersion(r.cos_raw, seeds[r.lang_a], seeds[r.lang_b])
            rows.append({"model": model, "lang_a": r.lang_a, "lang_b": r.lang_b,
                         "cos_raw": r.cos_raw, "cos_corrected_point": r.cos_corrected,
                         **d, "label": "dispersion across split seeds",
                         "population": COSINE_POPULATION})
        pd.DataFrame(rows).to_csv(args.out_root / f"{model}_corrected_cosines.csv", index=False)

        population_counts = {}
        grams = {}
        canonical_ids = None
        for lang in LANGS:
            cache_path = (
                args.cache_root / model / "deceptionbench" / f"{lang}_layer{layer}.npz"
            )
            acts, ids, n_admitted, n_nonadmitted = load_full_corpus_cache(cache_path)
            if canonical_ids is None:
                canonical_ids = ids
            canonical_idx, lang_idx = align_by_identity(canonical_ids, ids)
            aligned_canonical = [canonical_ids[i] for i in canonical_idx]
            aligned_lang = [ids[j] for j in lang_idx]
            if len(lang_idx) != EXPECTED_CKA_ITEMS or aligned_canonical != aligned_lang \
                    or len(set(aligned_lang)) != EXPECTED_CKA_ITEMS:
                raise ValueError(
                    f"{model}/{lang}: full alignment to the canonical language must contain "
                    f"{EXPECTED_CKA_ITEMS} unique, identical identities"
                )
            grams[lang] = centered_linear_gram(acts[lang_idx])
            population_counts[lang] = {
                "total": len(ids),
                "admitted": n_admitted,
                "nonadmitted": n_nonadmitted,
            }
        cka_rows = []
        for a, b in combinations(LANGS, 2):
            counts_a = population_counts[a]
            counts_b = population_counts[b]
            cka_rows.append({"model": model, "lang_a": a, "lang_b": b,
                             "cka_linear": linear_cka_from_grams(grams[a], grams[b]),
                             "n_matched": EXPECTED_CKA_ITEMS,
                             "n_a": counts_a["total"], "n_b": counts_b["total"],
                             "n_admitted_a": counts_a["admitted"],
                             "n_nonadmitted_a": counts_a["nonadmitted"],
                             "n_admitted_b": counts_b["admitted"],
                             "n_nonadmitted_b": counts_b["nonadmitted"],
                             "population": CKA_POPULATION})
        pd.DataFrame(cka_rows).to_csv(args.out_root / f"{model}_cka.csv", index=False)
        meta[model] = {
            "layer": int(layer),
            "cka_population_n_by_lang": population_counts,
        }
        print(f"[{model}] geometry battery: {len(rows)} cosine pairs, {len(cka_rows)} CKA pairs",
              flush=True)

    (args.out_root / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "corrected cosines = primary geometry; CKA = exploratory "
                "complement; descriptive only",
        "cka_alignment": "rows matched on (scenario_id, condition) across the parallel corpus",
        "cka_population": CKA_POPULATION,
        "cka_expected_items": EXPECTED_CKA_ITEMS,
        "corrected_cosine_population": COSINE_POPULATION,
        "reliability_seeds_required": EXPECTED_RELIABILITY_SEEDS,
        "models": meta}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
