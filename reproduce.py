#!/usr/bin/env python3
"""Reproduction driver for the paper pipeline.

    python reproduce.py --dry-run          # print the full-grid command sequence
    python reproduce.py --smoke            # end-to-end smoke test (one full cell:
                                           #   gemma-2b, English; expensive: GPU + judge API)
    python reproduce.py                    # full paper pipeline (5 models x 5 languages)
    python reproduce.py --stages analysis  # selected stages only

Stage order and hardware needs (full mode always runs the fixed paper grid):
  rollouts   generation + judging               GPU + OPENAI_API_KEY -> rollouts/
  sweep      English generic layer sweep +
             frozen-layer selection             GPU  -> results/sweeps/, results/frozen_layers/
  probes     final probes at the frozen layer
             (5 languages x 5 categories)       GPU  -> probes/
  geometry   shared-direction + language-offset
             artifacts                          GPU  -> results/geometry/
  caches     evaluation activations             GPU  -> results/activation_cache/
  analysis   transfer grid + paper analyses     GPU for probe_reliability and
             db_control_scores; CPU otherwise   -> analysis/

Outputs overwrite the conventional repository paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODELS = ["llama-8b", "aya-expanse-8b", "gemma-2b", "gemma-4-e4b", "qwen3.5-9b"]
LANGS = ["en", "pt", "es", "de", "zh"]
CATEGORIES = ["generic", "formality", "politeness", "truthfulness", "sentiment"]
SWEEP_LAYERS = {
    "llama-8b": "14,16,18,20,22",
    "aya-expanse-8b": "8,10,12,14,16,18,20,22",
    "gemma-2b": "10,12,14,16,18",
    "gemma-4-e4b": "18,22,26,30,34",
    "qwen3.5-9b": "14,16,18,20,22",
}

DRY = False


def batch_size(model: str) -> int:
    from project.models import get_model_info
    return int(get_model_info(model).sweep_batch_size)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    if not DRY:
        subprocess.run(cmd, cwd=ROOT, check=True)


def frozen_layer(model: str) -> int:
    p = ROOT / "results" / "frozen_layers" / model / "chosen_layer.json"
    if not p.exists():
        raise SystemExit(f"missing frozen-layer record: {p} (run the sweep stage)")
    return int(json.loads(p.read_text())["chosen_layer"])


def require(path: Path, hint: str) -> None:
    if not DRY and not path.exists():
        raise SystemExit(f"missing input: {path} ({hint})")


def remove_targets(*paths: Path) -> None:
    if DRY:
        return
    for p in paths:
        if p.exists():
            p.unlink()


def probe_out(model: str, lang: str, cat: str, layer: str) -> str:
    return f"probes/{model}/{lang}/{cat}/repe_probe_layer{layer}.pt"


JUDGE = "gpt-4.1-mini-2025-04-14"
GEN_BATCH = "4"


def stage_rollouts():
    for m in MODELS:
        for lang in LANGS:
            run([sys.executable, "scripts/generate_deceptionbench_rollouts.py",
                 "--model", m, "--lang", lang,
                 "--batch-size", GEN_BATCH, "--judge-model", JUDGE])


def stage_sweep():
    for m in MODELS:
        run([sys.executable, "scripts/sweep_layers.py", "--model", m,
             "--train-data", "repe_generic_en",
             "--layers", SWEEP_LAYERS[m],
             "--batch-size", "BS" if DRY else str(batch_size(m))])
        run([sys.executable, "scripts/select_layer.py", "--model", m])


def stage_probes():
    for m in MODELS:
        layer = "L" if DRY else str(frozen_layer(m))
        for lang in LANGS:
            for cat in CATEGORIES:
                run([sys.executable, "scripts/train_probe.py", "--model", m,
                     "--train-data", f"repe_{cat}_{lang}", "--layers", layer,
                     "--batch-size", "BS" if DRY else str(batch_size(m)),
                     "--out", probe_out(m, lang, cat, layer)])


def stage_geometry():
    for m in MODELS:
        layer = "L" if DRY else str(frozen_layer(m))
        run([sys.executable, "scripts/analysis/language_identity.py",
             "--model", m, "--layers", layer,
             "--batch-size", "BS" if DRY else str(batch_size(m))])
        run([sys.executable, "scripts/analysis/probe_geometry.py",
             "--model", m, "--layers", layer])


def stage_caches():
    for m in MODELS:
        run([sys.executable, "scripts/analysis/cache_eval_activations.py",
             "--model", m, "--langs", *LANGS, "--overwrite",
             "--batch-size", "BS" if DRY else str(batch_size(m))])


def stage_analysis():
    for m in MODELS:
        if not DRY:
            L = frozen_layer(m)
            for lang in LANGS:
                require(ROOT / "results" / "activation_cache" / m /
                        "deceptionbench" / f"{lang}_layer{L}.npz",
                        "run the caches stage")
        run([sys.executable, "scripts/analysis/db_projections.py", "--model", m])
        run([sys.executable, "scripts/analysis/probe_reliability.py",
             "--model", m, "--category", "generic",
             "--batch-size", "BS" if DRY else str(batch_size(m))])
        run([sys.executable, "scripts/analysis/db_control_scores.py",
             "--model", m,
             "--batch-size", "BS" if DRY else str(batch_size(m))])
    run([sys.executable, "scripts/analysis/transfer_stats.py", "--models", *MODELS])
    run([sys.executable, "scripts/analysis/transfer_decomposition.py"])
    run([sys.executable, "scripts/analysis/geometry_battery.py", "--models", *MODELS])
    run([sys.executable, "scripts/analysis/rank1_null.py"])
    run([sys.executable, "scripts/analysis/metric_battery.py", "--models", *MODELS])
    run([sys.executable, "scripts/analysis/operating_points.py", "--models", *MODELS])
    run([sys.executable, "scripts/analysis/residualization.py", "--models", *MODELS])
    run([sys.executable, "scripts/analysis/residualization_oof.py"])


def stage_disjoint():
    os.environ["PROJECT_USE_4BIT"] = "0"
    os.environ.pop("PROJECT_4BIT_THRESHOLD", None)
    for m in MODELS:
        if not DRY:
            L = frozen_layer(m)
            for lang in LANGS:
                require(ROOT / "results" / "activation_cache" / m /
                        "deceptionbench" / f"{lang}_layer{L}.npz",
                        "run the caches stage")
    for m in MODELS:
        run([sys.executable, "scripts/analysis/disjoint_pool_probes.py",
             "--model", m,
             "--batch-size", "BS" if DRY else str(batch_size(m))])
    run([sys.executable, "scripts/analysis/disjoint_pool_transfer.py",
         "--models", *MODELS])
    run([sys.executable, "scripts/analysis/direction_specificity.py",
         "--models", ",".join(MODELS)])


STAGES = {
    "rollouts": stage_rollouts,
    "sweep": stage_sweep,
    "probes": stage_probes,
    "geometry": stage_geometry,
    "caches": stage_caches,
    "analysis": stage_analysis,
    "disjoint": stage_disjoint,
}
ORDER = ["rollouts", "sweep", "probes", "geometry", "caches", "analysis",
         "disjoint"]

SMOKE_REF = {"layer": 16, "n": 749, "deceptive": 220, "honest": 529,
             "auroc": 0.856874}


def tie_aware_auroc(scores, y):
    import numpy as np
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    unsorted = np.empty(len(s))
    unsorted[order] = ranks
    return float((unsorted[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def smoke(model: str = "gemma-2b", lang: str = "en") -> None:
    """End-to-end smoke test: regenerates one full cell from raw included inputs
    (rollouts -> judging -> sweep -> layer selection -> probe -> cache -> AUROC)."""
    import numpy as np

    require(ROOT / "datasets" / "training" / "repe" / "true_false_facts_en.csv",
            "training data")
    require(ROOT / "datasets" / "evaluation" / "deceptionbench" / "healthcare.json",
            "evaluation scenarios")

    rollout = (ROOT / "rollouts" / model / "deceptionbench" /
               f"deceptionbench_{lang}_{model}.json")
    sweep_csv = ROOT / "results" / "sweeps" / f"{model}_repe_generic_en.csv"
    chosen = ROOT / "results" / "frozen_layers" / model / "chosen_layer.json"
    remove_targets(rollout, sweep_csv, chosen)

    run([sys.executable, "scripts/generate_deceptionbench_rollouts.py",
         "--model", model, "--lang", lang,
         "--batch-size", GEN_BATCH, "--judge-model", JUDGE])
    require(rollout, "rollout generation did not produce its output")
    run([sys.executable, "scripts/sweep_layers.py", "--model", model,
         "--train-data", "repe_generic_en", "--layers", SWEEP_LAYERS[model],
         "--batch-size", str(batch_size(model))])
    require(sweep_csv, "layer sweep did not produce its output")
    run([sys.executable, "scripts/select_layer.py", "--model", model])
    require(chosen, "layer selection did not produce its output")
    layer = frozen_layer(model)

    probe_file = ROOT / probe_out(model, lang, "generic", str(layer))
    remove_targets(probe_file)
    run([sys.executable, "scripts/train_probe.py", "--model", model,
         "--train-data", f"repe_generic_{lang}", "--layers", str(layer),
         "--out", str(probe_file.relative_to(ROOT))])
    require(probe_file, "probe training did not produce its output")

    cache = (ROOT / "results" / "activation_cache" / model / "deceptionbench" /
             f"{lang}_layer{layer}.npz")
    remove_targets(cache)
    run([sys.executable, "scripts/analysis/cache_eval_activations.py",
         "--model", model, "--langs", lang, "--overwrite",
         "--batch-size", str(batch_size(model))])
    require(cache, "cache extraction did not produce its output")

    from scripts.analysis._probe_loading import load_detector_stack
    LogisticRegressionDetector, probe_path = load_detector_stack()
    det = LogisticRegressionDetector.load(probe_file)
    v = det.effective_direction(layer).float().numpy()
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0.0:
        raise SystemExit("degenerate probe direction")
    v = v / np.linalg.norm(v)
    with np.load(cache) as npz:
        acts = npz["acts"].astype(np.float64)
        y = npz["labels"].astype(int)
        n_tok = npz["n_detect_tokens"].astype(int)
        if "prompt_acts" not in npz or "prompt_n_detect_tokens" not in npz:
            raise SystemExit("cache lacks prompt activations")
        pacts = npz["prompt_acts"].astype(np.float64)
        p_tok = npz["prompt_n_detect_tokens"].astype(int)
    if len(acts) == 0:
        raise SystemExit("empty activation cache")
    if (n_tok <= 0).any():
        raise SystemExit("zero-token detection masks in cache")
    if pacts.shape != acts.shape or not np.isfinite(pacts).all():
        raise SystemExit("misaligned or nonfinite prompt activations")
    if (p_tok <= 0).any():
        raise SystemExit("non-positive prompt token counts")
    scores = acts @ v
    if not np.isfinite(scores).all():
        raise SystemExit("nonfinite probe scores")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(f"admitted pool missing a class: pos={n_pos} neg={n_neg}")
    auroc = tie_aware_auroc(scores, y)
    print(f"smoke result:    layer={layer} n={len(y)} deceptive={n_pos} "
          f"honest={n_neg} auroc={auroc:.6f}")
    print(f"shipped reference: layer={SMOKE_REF['layer']} n={SMOKE_REF['n']} "
          f"deceptive={SMOKE_REF['deceptive']} honest={SMOKE_REF['honest']} "
          f"auroc={SMOKE_REF['auroc']:.6f} (diagnostic comparison only; "
          f"regenerated rollouts and judgments may differ slightly)")


def main() -> int:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", nargs="+", choices=ORDER,
                    default=[s for s in ORDER if s != "disjoint"])
    ap.add_argument("--smoke", action="store_true",
                    help="end-to-end smoke test: regenerate one full cell "
                         "(gemma-2b, English); needs GPU + OPENAI_API_KEY")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full command sequence without executing")
    args = ap.parse_args()
    DRY = args.dry_run
    if args.smoke:
        smoke()
        return 0
    for name in ORDER:
        if name in args.stages:
            STAGES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
