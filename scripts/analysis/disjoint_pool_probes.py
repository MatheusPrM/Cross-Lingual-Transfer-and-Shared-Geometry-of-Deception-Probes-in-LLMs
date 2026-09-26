#!/usr/bin/env python3
"""Disjoint-pool half-probe fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from scripts.analysis.disjoint_pool_common import (DIRS_NPZ_NAME,
                                                   FIT_META_NAME, OUT_REL,
                                                   SPLIT_SEEDS, fact_halves,
                                                   load_equivalence_manifest,
                                                   used_fact_statements,
                                                   verify_row_parallel)

_CONSTRAINTS_PATH = _REPO_ROOT / "configs/environment/constraints.txt"
_PRODUCTION_LANGS = ["en", "pt", "es", "de", "zh"]
_ARTIFACT_SCHEMA = "disjoint-pool-half-probes/v1"

_TRAIN_TEMPLATE = {
    "generic": "datasets/training/repe/true_false_facts_{lang}.csv",
}


RUN_OF_RECORD_DTYPE = {
    "gemma-2b": "torch.bfloat16",
    "llama-8b": "torch.bfloat16",
    "gemma-4-e4b": "torch.bfloat16",
    "qwen3.5-9b": "torch.bfloat16",
    "aya-expanse-8b": "torch.float16",
}


class _ActsView:
    """Minimal duck-type for LogisticRegressionDetector.fit — mirrors
    probe_reliability._ActsView exactly."""

    def __init__(self, t, counts=None):
        self._t = t
        self.detect_token_count_per_dialogue = counts
        self.acts_are_masked_only = True

    def get_masked_activations(self):
        return self._t


def _atomic_save_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _checked_unit(v: np.ndarray, what: str) -> np.ndarray:
    if not np.isfinite(v).all():
        raise SystemExit(f"{what}: non-finite direction")
    nrm = float(np.linalg.norm(v))
    if nrm <= 0:
        raise SystemExit(f"{what}: zero-norm direction")
    return (v / nrm).astype(np.float32)


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"required provenance input missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _launch_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
        dirty_worktree = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=_REPO_ROOT,
            check=False).returncode != 0
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"cannot establish launch git provenance: {e}")
    if dirty_worktree:
        raise SystemExit("tracked worktree differs from HEAD — commit/push the "
                         "camera-ready implementation before spending GPU time")
    return {
        "launch_commit": commit,
        "tracked_tree_dirty": False,
        "protocol_sha256": None,
        "constraints_sha256": _file_sha256(_CONSTRAINTS_PATH),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--category", default="generic", choices=sorted(_TRAIN_TEMPLATE))
    p.add_argument("--langs", nargs="+", default=["en", "pt", "es", "de", "zh"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SPLIT_SEEDS),
                   help="split seeds (probe-reliability convention 1000..1009; "
                        "note: overlapping partitions, not independent datasets)")
    p.add_argument("--reg-coeff", type=float, default=10.0, help="match sweep_layers default")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parity-min", type=float, default=0.98,
                   help="min cos(full-pool refit, stored paper probe) per language")
    p.add_argument("--allow-dtype", action="store_true",
                   help="permit a dtype outside this model's run-of-record "
                        "regime; off-record only and rejected by production "
                        "artifact validation")
    p.add_argument("--skip-pooled-checkpoint", action="store_true")
    p.add_argument("--out-root", type=Path, default=_REPO_ROOT / OUT_REL)
    args = p.parse_args()

    if os.environ.get("PROJECT_USE_4BIT") != "0":
        raise SystemExit("disjoint-pool fitting requires an unquantized load: "
                         "export PROJECT_USE_4BIT=0 and rerun")
    if os.environ.get("PROJECT_4BIT_THRESHOLD"):
        raise SystemExit("disjoint-pool fitting requires PROJECT_4BIT_THRESHOLD "
                         "to be unset")

    production_root = (_REPO_ROOT / OUT_REL).resolve()
    if args.out_root.resolve() == production_root:
        expected = {
            "category": args.category == "generic",
            "languages/order": args.langs == _PRODUCTION_LANGS,
            "split seeds": args.seeds == list(SPLIT_SEEDS),
            "reg_coeff": args.reg_coeff == 10.0,
            "extraction seed": args.seed == 42,
            "parity_min": args.parity_min == 0.98,
            "dtype override disabled": not args.allow_dtype,
        }
        wrong = [k for k, ok in expected.items() if not ok]
        if wrong:
            raise SystemExit(f"production disjoint-pool run changes registered "
                             f"controls: {wrong}; use a separate --out-root for "
                             f"an explicitly off-record experiment")
    provenance = _launch_provenance()

    import torch

    from project.activations import extract_masked_activations
    from project.detectors import LogisticRegressionDetector
    from project.frozen_layers import load_frozen_layer  
    from project.loaders import load_dialogues_from_path
    from project.model_loader import (load_model_and_tokenizer,
                                      model_precision_info,
                                      resolved_model_revision)
    from project.reproducibility import set_seed
    from project.tokenized_data import tokenize_dialogues
    from project.types import Label
    from scripts.analysis.load_probes import probe_path

    set_seed(args.seed)
    layer = load_frozen_layer(args.model)
    model_revision = resolved_model_revision(args.model)
    if not model_revision:
        raise SystemExit(f"{args.model}: no locked model revision — refusing GPU run")
    out_dir = args.out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = {lang: _REPO_ROOT / _TRAIN_TEMPLATE[args.category].format(lang=lang)
            for lang in args.langs}
    for lang, cp in csvs.items():
        if not cp.exists():
            raise FileNotFoundError(cp)
    csv_contract = verify_row_parallel(csvs)
    n_used = {v["n_used_facts"] for v in csv_contract.values()}
    if len(n_used) != 1:
        raise SystemExit(f"used-fact counts differ across languages: {n_used}")

    ref_lang = "en" if "en" in args.langs else args.langs[0]
    statements_by_lang = {
        lang: used_fact_statements(
            csvs[lang],
            column="statement" if lang == "en" else f"statement_{lang}")
        for lang in args.langs}
    statements = statements_by_lang[ref_lang]
    groups, manifest_sha = load_equivalence_manifest(
        _REPO_ROOT, csvs, statements_by_lang)
    n_facts_expected = len(statements)
    halves_by_seed = {s: fact_halves(n_facts_expected, s, groups=groups)
                      for s in args.seeds}

    for s, (ha, hb) in halves_by_seed.items():
        if len(ha) != n_facts_expected // 2:
            raise SystemExit(f"seed {s}: halves are {len(ha)}/{len(hb)}, "
                             f"expected exact {n_facts_expected // 2} — "
                             f"balance repair failed.")
    n_dup_groups = sum(1 for g in set(groups)
                       if sum(1 for x in groups if x == g) > 1)
    print(f"[{args.model}/{args.category}] contract OK: {n_facts_expected} used "
          f"row-parallel facts; {n_dup_groups} duplicate-equivalent groups from "
          f"the frozen manifest ({manifest_sha[:12]}...) kept whole; every seed "
          f"splits exactly {n_facts_expected // 2}/{n_facts_expected - n_facts_expected // 2}; "
          f"layer {layer}; seeds {args.seeds}", flush=True)

    for lang in args.langs:
        dialogues, labels = load_dialogues_from_path(csvs[lang], repe_category=args.category)
        y = np.array([1 if l == Label.DECEPTIVE else 0 for l in labels])
        if int((y == 0).sum()) != n_facts_expected or int((y == 1).sum()) != n_facts_expected:
            raise SystemExit(f"{lang}: loader produced {int((y==0).sum())}h/"
                             f"{int((y==1).sum())}d dialogues; expected "
                             f"{n_facts_expected} of each")
        pattern_ok = all(int(y[2 * k]) == 0 and int(y[2 * k + 1]) == 1
                         for k in range(n_facts_expected))
        if not pattern_ok:
            raise SystemExit(f"{lang}: honest/deceptive pair alternation broken — "
                             f"the k-th honest and k-th deceptive dialogue would "
                             f"not share fact k")

    print(f"[{args.model}] loading model...", flush=True)
    hf_model, tok = load_model_and_tokenizer(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.bos_token_id is not None:
        tok.pad_token_id = tok.bos_token_id
    first_param = next(hf_model.parameters())
    if not torch.cuda.is_available() or first_param.device.type != "cuda":
        raise SystemExit(f"{args.model}: production fitting requires CUDA, but "
                         f"first parameter is on {first_param.device}")
    param_dtype = str(first_param.dtype)
    precision = model_precision_info(hf_model)
    if precision.get("quantized_4bit"):
        raise SystemExit(f"{args.model}: model loaded quantized despite the "
                         f"unquantized run-of-record contract: {precision}")

    expected_dtype = RUN_OF_RECORD_DTYPE.get(args.model)
    if expected_dtype is None and not args.allow_dtype:
        raise SystemExit(f"{args.model}: no run-of-record dtype registered — "
                         f"add it to RUN_OF_RECORD_DTYPE or pass --allow-dtype "
                         f"for a deliberate off-record experiment.")
    if expected_dtype is not None and param_dtype != expected_dtype \
            and not args.allow_dtype:
        raise SystemExit(f"{args.model}: param dtype {param_dtype} != "
                         f"run-of-record {expected_dtype}. Export "
                         f"PROJECT_USE_4BIT=0 and match the recorded regime; "
                         f"--allow-dtype only for off-record experiments.")

    def _fit_direction(tok_rows_h, tok_rows_d, counts_h, counts_d) -> np.ndarray:
        det = LogisticRegressionDetector(layers=[layer], reg_coeff=args.reg_coeff, scale=True)
        det.fit(_ActsView(tok_rows_h, counts_h), _ActsView(tok_rows_d, counts_d),
                unit="dialogue")
        return det.effective_direction(layer).float().numpy()

    arrays: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    for lang in args.langs:
        dialogues, labels = load_dialogues_from_path(csvs[lang], repe_category=args.category)
        print(f"  extracting {lang} ({len(dialogues)} dialogues)...", flush=True)
        acts = extract_masked_activations(
            hf_model, tokenize_dialogues(dialogues, tok, model_id=args.model),
            layers=[layer], batch_size=args.batch_size)
        counts = acts.detect_token_count_per_dialogue
        bounds = np.concatenate([[0], np.cumsum(counts)])
        y = np.array([1 if l == Label.DECEPTIVE else 0 for l in labels])[: len(counts)]

        def token_rows(dialogue_idx: np.ndarray) -> "torch.Tensor":
            rows = [acts.acts[bounds[i]:bounds[i + 1], :, :] for i in dialogue_idx]
            return torch.cat(rows, dim=0)

        idx_h_all = np.where(y == 0)[0]
        idx_d_all = np.where(y == 1)[0]
        if min(len(idx_h_all), len(idx_d_all)) < n_facts_expected:
            raise SystemExit(f"{lang}: extraction dropped dialogues below the "
                             f"preflight fact count ({len(idx_h_all)}h/"
                             f"{len(idx_d_all)}d < {n_facts_expected}) — refusing "
                             f"to fit on a silently shrunken pool.")

        w_full = _checked_unit(
            _fit_direction(token_rows(idx_h_all), token_rows(idx_d_all),
                           [counts[i] for i in idx_h_all],
                           [counts[i] for i in idx_d_all]),
            f"{lang} full-pool refit")
        pp = probe_path(args.model, lang, layer, category=args.category,
                        require_category_path=True)
        if not pp.exists():
            raise SystemExit(f"{lang}: stored paper probe missing at {pp} — "
                             f"cannot certify fitter parity.")
        det_stored = LogisticRegressionDetector.load(pp)
        w_stored = det_stored.effective_direction(layer).float().numpy()
        w_stored = w_stored / np.linalg.norm(w_stored)
        cos_parity = float(np.dot(w_full.astype(np.float64), w_stored.astype(np.float64)))
        parity[lang] = cos_parity
        print(f"    {lang}: fitter parity cos(refit, stored) = {cos_parity:.5f}", flush=True)
        if cos_parity < args.parity_min:
            raise SystemExit(f"{lang}: FITTER PARITY FAILED ({cos_parity:.5f} < "
                             f"{args.parity_min}) — extraction or fit drift vs the "
                             f"paper probes; half-probes from this extraction would "
                             f"not be comparable. Aborting before wasting GPU.")

        if not args.skip_pooled_checkpoint:
            pooled = np.stack([
                acts.acts[bounds[i]:bounds[i + 1], 0, :].float().mean(dim=0).numpy()
                for i in range(len(counts))]).astype(np.float16)
            _atomic_save_npz(out_dir / f"pooled_train_acts_{lang}.npz",
                             pooled=pooled, y=y.astype(np.int8),
                             counts=np.asarray(counts, dtype=np.int32))

        dirs = np.zeros((len(args.seeds), 2, acts.acts.shape[-1]), dtype=np.float32)
        for si, split_seed in enumerate(args.seeds):
            half_a, half_b = halves_by_seed[split_seed]
            for hi, sel in enumerate((half_a, half_b)):
                w = _fit_direction(token_rows(idx_h_all[sel]), token_rows(idx_d_all[sel]),
                                   [counts[i] for i in idx_h_all[sel]],
                                   [counts[i] for i in idx_d_all[sel]])
                dirs[si, hi] = _checked_unit(w, f"{lang} seed {split_seed} half {hi}")
            print(f"    {lang} seed {split_seed}: fitted A ({len(half_a)} facts) "
                  f"+ B ({len(half_b)} facts)", flush=True)
        arrays[lang] = dirs
        del acts

    _atomic_save_npz(out_dir / DIRS_NPZ_NAME, **arrays)
    npz_sha = hashlib.sha256((out_dir / DIRS_NPZ_NAME).read_bytes()).hexdigest()
    import transformers
    import sklearn
    _atomic_write_text(out_dir / FIT_META_NAME, json.dumps({
        "artifact_schema": _ARTIFACT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "category": args.category, "layer": int(layer),
        "model_revision": model_revision,
        "extraction_seed": args.seed,
        "param_dtype": param_dtype,
        "expected_dtype": expected_dtype,
        "off_record_dtype_override": bool(args.allow_dtype),
        "precision": precision,
        "dirs_npz_sha256": npz_sha,
        "n_langs": len(args.langs), "langs": list(args.langs),
        "seeds": list(args.seeds),
        "n_facts_used": int(n_facts_expected),
        "fact_universe": "label==1 rows of the training CSV, file order, "
                         "head(512) — the generic loader's exact filter",
        "halves_by_seed": {str(s): {"a": halves_by_seed[s][0].tolist(),
                                    "b": halves_by_seed[s][1].tolist()}
                           for s in args.seeds},
        "duplicate_groups": groups,
        "grouping": "FROZEN reviewed manifest (datasets/training/repe/"
                    "fact_equivalence_manifest.json): mechanical multilingual "
                    "union (exact + token Jaccard >0.8 segmented / char "
                    "3-gram >0.8 unsegmented) + adjudicated semantic pairs; "
                    "claim: disjoint row IDs with duplicate-equivalent facts "
                    "grouped, not absolute semantic independence",
        "groups_sha256": manifest_sha,
        "used_statements": statements_by_lang,
        "used_statements_en": statements,
        "detector": {"kind": "LogisticRegressionDetector", "unit": "dialogue",
                     "reg_coeff": args.reg_coeff, "scale": True},
        "direction_convention": "unit effective_direction(layer), float32; "
                                "axis 1: 0 = half A, 1 = half B",
        "fitter_parity_cos": parity,
        "parity_min": args.parity_min,
        "train_csvs": {l: csv_contract[l] for l in args.langs},
        **provenance,
        "env": {"torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "cpu")},
        "complete": True,  
    }, indent=2))
    print(f"Saved: {out_dir / DIRS_NPZ_NAME} + {FIT_META_NAME} (complete)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
