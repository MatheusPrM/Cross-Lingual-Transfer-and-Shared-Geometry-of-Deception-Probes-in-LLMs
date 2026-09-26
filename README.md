# Cross-Lingual Transfer and Shared Geometry of Deception Probes in LLMs

This is the repository for the paper Cross-Lingual Transfer and Shared
Geometry of Deception Probes in LLMs, to be published at the 6th Workshop
on Multilingual Representation Learning (MRL) at EMNLP 2026.

## Abstract

Linear probes can recover deception-related signals from large language model activations, but existing evidence is overwhelmingly English-centric. We investigate whether this discrimination transfers across languages and what representational geometry accompanies such transfer. We train instructed-deception probes on controlled honest/deceptive instruction contrasts in English, Portuguese, Spanish, German, and Chinese, and evaluate them without target-language refitting on a multilingual adaptation of DeceptionBench across five models spanning four different families. Cross-language performance remains close to matched-language baselines, with all models satisfying our prespecified noninferiority criterion, although absolute discrimination varies substantially across models. We also find that language-specific probe directions are strongly aligned and concentrate around a shared consensus direction, while response representations retain clear language-specific organization. Additional construct-validity analyses show that measurable response-level cues, including style, sentiment, factual truthfulness, and length, explain part but not all of the strongest probe discrimination. These findings together suggest that cross-lingual transfer and shared probe geometry can coexist with model-dependent detection quality and persistent language structure, indicating that representational alignment alone is insufficient evidence of reliable multilingual deception monitoring.

## Reproduction

The recorded runs used Python 3.12 and CUDA 12.8 wheels.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c configs/environment/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

`analysis/` contains the outputs behind the paper's tables and figures. To
re-run the pipeline:

```
python reproduce.py                           # generation through analysis, 5 models x 5 languages
python reproduce.py --stages caches analysis  # analysis stages from the shipped rollouts
python reproduce.py --stages caches disjoint  # disjoint-pool fits, transfer, direction specificity
```

Model downloads require `HF_TOKEN` and the judging stage requires `OPENAI_API_KEY`. The
shipped rollouts are already judged. `reproduce.py` regenerates rollouts,
probes, and caches in place, so run it in a separate clone to keep the
shipped artifacts intact. Sentiment reliability is not reproducible via this repository.

## Data

Redistributed and derived datasets keep their upstream terms, documented in
`DATA_SOURCES.md`.
