#!/usr/bin/env python3
"""
Cache per-example mean-pooled residual activations at the frozen layer per (model, family, lang).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.paths import REPO_ROOT  
from project.types import Dialogue, Label 

_DEFAULT_FAMILIES = ["deceptionbench"]
_DEFAULT_LANGS = ["en", "pt", "es", "de", "zh"]
_ROLLOUT_DIR = {}


def _frozen_layer(model: str, frozen_root: Path) -> int | None:
    p = frozen_root / model / "chosen_layer.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return int(data["chosen_layer"])


def _rollout_candidates(family: str, model: str, lang: str) -> list[str]:
    """On-disk rollout filenames per family (mirrors scripts/analysis/shared_direction_monitor.py)."""
    if family == "deceptionbench":
        return [f"deceptionbench_{lang}_{model}.json"]
    raise ValueError(f"Unsupported family: {family}")


def _rollout_paths(rollouts_root: Path, model: str, family: str, lang: str) -> list[Path]:
    """Existing rollout files for a cell (first hit per family)."""
    base = rollouts_root / model / _ROLLOUT_DIR.get(family, family)
    existing = [base / n for n in _rollout_candidates(family, model, lang) if (base / n).exists()]
    return existing[:1]


_DB_META_FIELDS = ("scenario_id", "row_index", "domain", "condition", "level", "role",
                   "inducement", "termination", "judge_label")

def _db_meta(it: dict, src_idx: int) -> dict:
    m = {"src_file": 0, "src_idx": src_idx}
    for f in _DB_META_FIELDS:
        m[f] = it.get(f)
    return m


def _deceptionbench_cell(paths: list[Path]) -> tuple[list[Dialogue], list[int], str, list[dict]]:
    from project.deceptionbench import fill_assistant_response, load_deceptionbench_rollouts, messages_to_dialogue
    from project.deceptionbench.grading import normalize_label

    dialogues: list[Dialogue] = []
    labels: list[int] = []
    meta: list[dict] = []
    items, _ = load_deceptionbench_rollouts(paths[0])
    for i, it in enumerate(items):
        lab_raw = normalize_label(it.get("judge_label"))
        if lab_raw == "deceptive":
            lab = 1
        elif lab_raw == "honest_answer":
            lab = 0
        else:
            continue
        msgs = it.get("prompt_messages") or []
        base = messages_to_dialogue([dict(x) for x in msgs if isinstance(x, dict)])
        dialogues.append(fill_assistant_response(base, str(it.get("response") or "")))
        labels.append(lab)
        meta.append(_db_meta(it, i))
    return dialogues, labels, "judge_label", meta


def _deceptionbench_nonadmitted(paths: list[Path]) -> tuple[list[Dialogue], list[dict]]:
    from project.deceptionbench import fill_assistant_response, load_deceptionbench_rollouts, messages_to_dialogue
    from project.deceptionbench.grading import normalize_label

    dialogues: list[Dialogue] = []
    meta: list[dict] = []
    items, _ = load_deceptionbench_rollouts(paths[0])
    for i, it in enumerate(items):
        lab_raw = normalize_label(it.get("judge_label"))
        if lab_raw in ("deceptive", "honest_answer"):
            continue
        if not lab_raw or lab_raw == "malformed" or not it.get("response_wellformed", True):
            continue
        msgs = it.get("prompt_messages") or []
        base = messages_to_dialogue([dict(x) for x in msgs if isinstance(x, dict)])
        dialogues.append(fill_assistant_response(base, str(it.get("response") or "")))
        meta.append(_db_meta(it, i))
    return dialogues, meta


_CELL_LOADERS = {
    "deceptionbench": _deceptionbench_cell,
}


def _mean_pooled_acts(model, tokenizer, model_id: str, family: str, dialogues: list[Dialogue],
                      layer: int, batch_size: int):
    """Extract masked activations at ``layer`` and mean-pool per dialogue.

    Returns (acts, kept, prompt_acts, resp_tok_counts, prompt_tok_counts): kept = input
    dialogue indices with >0 detect tokens; the two count lists are per KEPT row (prompt side
    is None outside deceptionbench)."""
    import torch

    from project.activations import extract_masked_activations

    if family == "deceptionbench":
        from project.deceptionbench import tokenize_dialogues_length_masked
        tokenized = tokenize_dialogues_length_masked(dialogues, tokenizer, model_id=model_id)
    else:
        from project.tokenized_data import tokenize_dialogues
        tokenized = tokenize_dialogues(dialogues, tokenizer, model_id=model_id)
    acts = extract_masked_activations(model, tokenized, layers=[layer], batch_size=batch_size)
    counts = acts.detect_token_count_per_dialogue
    per_dialogue = torch.split(acts.acts[:, 0, :].float(), counts)
    pooled: list[torch.Tensor] = []
    kept: list[int] = []
    for i, chunk in enumerate(per_dialogue):
        if chunk.shape[0] == 0:
            continue
        pooled.append(chunk.mean(dim=0))
        kept.append(i)
    if not pooled:
        return np.zeros((0, 0), dtype=np.float32), [], None, [], None
    resp = torch.stack(pooled).cpu().numpy().astype(np.float32)
    resp_tok_counts = [int(counts[i]) for i in kept]

    from project.deceptionbench.dialogue import tokenize_prompt_only_length_masked
    ptok = tokenize_prompt_only_length_masked([dialogues[i] for i in kept], tokenizer, model_id=model_id)
    pa = extract_masked_activations(model, ptok, layers=[layer], batch_size=batch_size)
    pchunks = torch.split(pa.acts[:, 0, :].float(), pa.detect_token_count_per_dialogue)
    prows = [c.mean(dim=0) for c in pchunks if c.shape[0] > 0]
    if len(prows) != len(kept):
        raise RuntimeError(
            f"prompt/response row mismatch ({len(prows)} vs {len(kept)})")
    prompt_acts = torch.stack(prows).cpu().numpy().astype(np.float32)
    prompt_tok_counts = [int(c.shape[0]) for c in pchunks if c.shape[0] > 0]
    if prompt_acts.shape != resp.shape:
        raise RuntimeError(
            f"prompt/response shape mismatch ({prompt_acts.shape} vs {resp.shape})")
    if not np.isfinite(prompt_acts).all() or min(prompt_tok_counts) <= 0:
        raise RuntimeError("nonfinite prompt activations or empty prompt mask")
    return resp, kept, prompt_acts, resp_tok_counts, prompt_tok_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache admitted evaluation activations at the frozen layer.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--families", nargs="+", default=_DEFAULT_FAMILIES)
    parser.add_argument("--langs", nargs="+", default=_DEFAULT_LANGS)
    parser.add_argument("--rollouts-root", type=Path, default=REPO_ROOT / "rollouts")
    parser.add_argument("--frozen-root", type=Path, default=REPO_ROOT / "results" / "frozen_layers")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "activation_cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Re-extract cells whose .npz already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Report per-cell rollout files and label counts without loading the model.")
    args = parser.parse_args()

    layer = _frozen_layer(args.model, args.frozen_root)
    if layer is None:
        print(f"No frozen-layer JSON for {args.model}; run scripts/select_layer.py first.", file=sys.stderr)
        return 1

    hf_model = None
    tokenizer = None
    n_written = 0
    n_failed = 0
    for fam in args.families:
        if fam not in _CELL_LOADERS:
            print(f"  [fail] unknown family {fam!r}", file=sys.stderr)
            n_failed += len(args.langs)
            continue
        for lang in args.langs:
            out_path = args.out_root / args.model / fam / f"{lang}_layer{layer}.npz"
            if out_path.exists() and not args.overwrite:
                print(f"  [skip] {fam}/{lang}: already cached at {out_path}")
                continue
            if args.overwrite and not args.dry_run:
                out_path.unlink(missing_ok=True)
            paths = _rollout_paths(args.rollouts_root, args.model, fam, lang)
            if not paths:
                print(f"  [fail] {fam}/{lang}: no rollout file", file=sys.stderr)
                n_failed += 1
                continue
            try:
                dialogues, labels, label_source, meta = _CELL_LOADERS[fam](paths)
            except Exception as e:
                print(f"  [fail] {fam}/{lang}: {e}", file=sys.stderr)
                n_failed += 1
                continue
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            if n_pos == 0 or n_neg == 0:
                print(f"  [fail] {fam}/{lang}: degenerate cell (pos={n_pos}, neg={n_neg}, source={label_source})",
                      file=sys.stderr)
                n_failed += 1
                continue
            if args.dry_run:
                print(f"  [dry] {fam}/{lang}: n={len(labels)} (pos={n_pos}, neg={n_neg}, source={label_source}) from {[p.name for p in paths]}")
                continue
            if hf_model is None:
                from project.model_loader import load_model_and_tokenizer
                print(f"Loading model {args.model} ...", flush=True)
                hf_model, tokenizer = load_model_and_tokenizer(args.model)
                tokenizer.padding_side = "left"
                if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.bos_token_id
            try:
                acts, kept, prompt_acts, resp_counts, prompt_counts = _mean_pooled_acts(
                    hf_model, tokenizer, args.model, fam, dialogues, layer, args.batch_size)
                kept_labels = np.array([labels[i] for i in kept], dtype=np.int8)
                if acts.shape[0] == 0:
                    raise RuntimeError("no detect tokens after tokenization")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                _extra = {"prompt_acts": prompt_acts}
                kept_meta = [meta[i] for i in kept]
                _extra["src_file"] = np.array([m["src_file"] for m in kept_meta], dtype=np.int32)
                _extra["src_idx"] = np.array([m["src_idx"] for m in kept_meta], dtype=np.int32)
                _extra["n_detect_tokens"] = np.array(resp_counts, dtype=np.int32)
                if fam == "deceptionbench":
                    for f in _DB_META_FIELDS:
                        _extra[f] = np.array([str(m.get(f)) for m in kept_meta])
                    if prompt_counts is not None:
                        _extra["prompt_n_detect_tokens"] = np.array(prompt_counts, dtype=np.int32)
                    na_dialogues, na_meta = _deceptionbench_nonadmitted(paths)
                    if na_dialogues:
                        na_acts, na_kept, na_prompt, na_counts, na_pcounts = _mean_pooled_acts(
                            hf_model, tokenizer, args.model, fam, na_dialogues, layer, args.batch_size)
                        _extra["nonadmitted_n_dropped_no_detect"] = np.int64(len(na_dialogues) - len(na_kept))
                        if na_acts.shape[0] > 0:
                            na_km = [na_meta[i] for i in na_kept]
                            _extra["nonadmitted_acts"] = na_acts
                            if na_prompt is not None:
                                _extra["nonadmitted_prompt_acts"] = na_prompt
                            _extra["nonadmitted_src_idx"] = np.array([m["src_idx"] for m in na_km], dtype=np.int32)
                            _extra["nonadmitted_n_detect_tokens"] = np.array(na_counts, dtype=np.int32)
                            for f in _DB_META_FIELDS:
                                _extra[f"nonadmitted_{f}"] = np.array([str(m.get(f)) for m in na_km])
                            print(f"  [+] {fam}/{lang}: archived {na_acts.shape[0]} non-admitted rows", flush=True)
                tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
                tmp_path.unlink(missing_ok=True)
                try:
                    np.savez_compressed(
                        tmp_path,
                        acts=acts,
                        labels=kept_labels,
                        **_extra,
                        layer=np.int64(layer),
                        model=np.str_(args.model),
                        family=np.str_(fam),
                        lang=np.str_(lang),
                        label_source=np.str_(label_source),
                        rollout_files=np.array(
                            [str(p.relative_to(args.rollouts_root)) for p in paths]),
                        n_dropped_no_detect=np.int64(len(labels) - len(kept)),
                    )
                    with np.load(tmp_path) as chk:
                        req = ("acts", "labels", "prompt_acts",
                               "n_detect_tokens", "prompt_n_detect_tokens",
                               "scenario_id", "condition")
                        missing_keys = [k for k in req if k not in chk.files]
                        if missing_keys:
                            raise RuntimeError(f"missing arrays: {missing_keys}")
                        a, pl = chk["acts"], chk["prompt_acts"]
                        yv = chk["labels"].astype(int)
                        nt = chk["n_detect_tokens"].astype(int)
                        pt = chk["prompt_n_detect_tokens"].astype(int)
                        if a.shape[0] == 0 or a.shape != pl.shape:
                            raise RuntimeError("empty or misaligned acts/prompt_acts")
                        if not (np.isfinite(a).all() and np.isfinite(pl).all()):
                            raise RuntimeError("nonfinite activations")
                        if (nt <= 0).any() or (pt <= 0).any():
                            raise RuntimeError("non-positive detect-token counts")
                        if int((yv == 1).sum()) == 0 or int((yv == 0).sum()) == 0:
                            raise RuntimeError(
                                f"admitted pool missing a class "
                                f"(pos={int((yv == 1).sum())}, neg={int((yv == 0).sum())})")
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
                os.replace(tmp_path, out_path)
                print(f"  [ok] {fam}/{lang}: wrote {out_path} (n={acts.shape[0]}, pos={int(kept_labels.sum())}, hidden={acts.shape[1]})")
                n_written += 1
            except Exception as e:
                print(f"  [fail] {fam}/{lang}: {e}", file=sys.stderr)
                n_failed += 1

    print(f"Done: {n_written} cells written, {n_failed} failed.")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
