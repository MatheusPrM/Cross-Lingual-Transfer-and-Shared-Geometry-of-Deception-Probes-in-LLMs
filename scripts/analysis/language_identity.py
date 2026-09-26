#!/usr/bin/env python3
"""
Compute language-identity directions from the row-parallel REPE fact CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis.load_probes import DEFAULT_CATEGORY, DEFAULT_LANGS, REPO_ROOT, common_layers
from scripts.analysis.probe_geometry import GEOMETRY_DIR, _model_out_dir

REPE_DIR = REPO_ROOT / "datasets" / "training" / "repe"


def repe_csv_path(lang: str) -> Path:
    return REPE_DIR / f"true_false_facts_{lang}.csv"


def compute_mean_activations(
    model_id: str,
    langs: list[str],
    layers: list[int],
    *,
    category: str = DEFAULT_CATEGORY,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Per language: mean detect-token activation, shape [n_layers, hidden] float32."""
    import torch

    from project.activations import extract_masked_activations
    from project.loaders import load_dialogues_from_path
    from project.model_loader import load_model_and_tokenizer
    from project.tokenized_data import tokenize_dialogues

    print(f"Loading model {model_id}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    means: dict[str, np.ndarray] = {}
    for lang in langs:
        csv = repe_csv_path(lang)
        if not csv.exists():
            print(f"  [{lang}] missing {csv}; skipping", flush=True)
            continue
        dialogues, _labels = load_dialogues_from_path(csv, repe_category=category)
        if not dialogues:
            print(f"  [{lang}] no dialogues from {csv}; skipping", flush=True)
            continue
        tok = tokenize_dialogues(dialogues, tokenizer, model_id=model_id, exclude_last_n_detect_tokens=0)
        acts = extract_masked_activations(model, tok, layers=layers, batch_size=batch_size)
        if acts.acts.shape[0] == 0:
            print(f"  [{lang}] zero detect tokens; skipping", flush=True)
            continue
        _chunks = torch.split(acts.acts.to(torch.float32), acts.detect_token_count_per_dialogue)
        means[lang] = torch.stack([c.mean(dim=0) for c in _chunks if c.shape[0] > 0]).mean(dim=0).cpu().numpy() 
        print(f"  [{lang}] {acts.acts.shape[0]} detect tokens over {len(dialogues)} dialogues", flush=True)
    return means


def save_lang_identity(
    means: dict[str, np.ndarray],
    layers: list[int],
    *,
    model_id: str,
    category: str,
    ref_lang: str,
    out_root: Path | None = None,
) -> None:
    """Write per-layer lang_identity.npz next to (or ahead of) decomposition.json."""
    out_root = out_root or GEOMETRY_DIR
    if ref_lang not in means:
        raise SystemExit(f"Reference language {ref_lang!r} has no activations; cannot form differences.")
    base = _model_out_dir(out_root, model_id, category)
    for i, layer in enumerate(layers):
        arrays: dict[str, np.ndarray] = {"ref_lang": np.array(ref_lang)}
        for lang, m in means.items():
            if lang == ref_lang:
                continue
            d = m[i] - means[ref_lang][i]
            n = float(np.linalg.norm(d))
            if n < 1e-8:
                print(f"  layer {layer} [{lang}] near-zero mean difference; skipping", flush=True)
                continue
            arrays[f"langid_{lang}"] = (d / n).astype(np.float32)
            arrays[f"diff_norm_{lang}"] = np.array(n, dtype=np.float32)
        layer_dir = base / f"layer{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        out = layer_dir / "lang_identity.npz"
        np.savez(out, **arrays)
        n_dirs = sum(1 for k in arrays if k.startswith("langid_"))
        print(f"  wrote {out.relative_to(REPO_ROOT)} ({n_dirs} direction(s))", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model id (e.g. gemma-2b, llama-8b).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices (e.g. 14,17,18).")
    g.add_argument("--auto-layers", action="store_true",
                   help="Use the layers that have probes for every requested language (matches probe_geometry).")
    p.add_argument("--langs", type=str, default=",".join(DEFAULT_LANGS),
                   help="Comma-separated language codes incl. the reference (default: en,pt,es,de,zh).")
    p.add_argument("--ref-lang", type=str, default="en", help="Reference language for the differences (default: en).")
    p.add_argument("--category", type=str, default=DEFAULT_CATEGORY,
                   help="REPE instruction category; matches the geometry output dir layout (default: generic).")
    p.add_argument("--batch-size", type=int, default=8, help="Forward batch size.")
    args = p.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if args.ref_lang not in langs:
        langs = [args.ref_lang] + langs
    if args.auto_layers:
        layers = common_layers(args.model, langs=tuple(langs), category=args.category)
        if not layers:
            raise SystemExit(f"No layers with probes for all of {langs} under {args.model} (category={args.category}).")
        print(f"Auto-discovered {len(layers)} common layer(s): {layers}")
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    means = compute_mean_activations(
        args.model, langs, layers, category=args.category, batch_size=args.batch_size
    )
    save_lang_identity(
        means, layers, model_id=args.model, category=args.category, ref_lang=args.ref_lang
    )
    print("Done. Re-run scripts/analysis/probe_geometry.py to fold langid cosines into decomposition.json.")


if __name__ == "__main__":
    main()
