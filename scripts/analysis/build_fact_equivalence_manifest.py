#!/usr/bin/env python3
"""
Build the FROZEN fact-equivalence manifest for the disjoint-pool analysis.
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

from scripts.analysis.disjoint_pool_common import (_UnionFind,
                                                   duplicate_groups_multilingual,
                                                   normalize_statement,
                                                   used_fact_statements)

MANIFEST_REL = "datasets/training/repe/fact_equivalence_manifest.json"
LANGS = ("en", "pt", "es", "de", "zh")

SEMANTIC_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("the sky is blue", "the color of the sky is blue", "same proposition"),
    ("mercury is the smallest planet",
     "the smallest planet in our solar system is mercury", "same proposition"),
    ("the largest mammal on earth is the blue whale",
     "the worlds largest mammal is the blue whale", "same proposition"),
    ("the currency of japan is the yen",
     "the currency used in japan is the yen", "same proposition"),
    ("the earths largest ocean is the pacific ocean",
     "the largest ocean in the world is the pacific ocean", "same proposition"),
    ("sound travels through the air as vibrations",
     "sound travels through the air as waves", "same proposition"),
    ("sound travels as a wave through various mediums",
     "sound travels through the air as waves",
     "same proposition (medium generalized)"),
    ("tornadoes are rapidly rotating columns of air",
     "tornadoes are rapidly rotating columns of air that can cause extensive damage",
     "literal containment"),
    ("water freezes at 0 degrees celsius 32 degrees fahrenheit",
     "water freezes at 0 degrees celsius 32 degrees fahrenheit and boils at "
     "100 degrees celsius 212 degrees fahrenheit", "literal containment"),
    ("rainbows form when light refracts through water droplets",
     "rainbows are formed when light is refracted through water droplets in the air",
     "same proposition"),
    ("diamonds are made of carbon", "diamonds are formed from carbon",
     "same proposition"),
    ("gravity is the force that attracts objects with mass towards each other",
     "gravity pulls objects towards each other", "same proposition"),
    ("gravity makes things fall down", "gravity pulls objects towards each other",
     "same proposition (everyday formulation)"),
    ("uranus orbits the sun on its side",
     "the planet uranus is tilted on its side", "same proposition"),
    ("plants use photosynthesis to create energy from sunlight",
     "photosynthesis is the process by which plants convert sunlight into "
     "chemical energy", "same proposition"),
    ("the first law of thermodynamics states that energy cannot be created or "
     "destroyed only converted from one form to another",
     "the conservation of energy principle states that energy cannot be "
     "created or destroyed", "same proposition (naming differs)"),
    ("the planet mars has the largest volcano in our solar system",
     "mars has the largest volcano olympus mons", "same proposition"),
    ("the scientific method is a systematic process for investigating "
     "phenomena and acquiring new knowledge",
     "the scientific method is a process for testing hypotheses and acquiring "
     "knowledge", "same proposition (definition wording)"),
    ("the human brain controls the bodys functions",
     "the human brain is the control center for the bodys functions and emotions",
     "same proposition + containment"),
    ("the first successful powered flight was made by the wright brothers in 1903",
     "the wright brothers made the first successful airplane flight",
     "same proposition (year attribution)"),
    ("the human eye can detect millions of different colors",
     "the human eye can distinguish about 10 million different colors",
     "same proposition (precision differs)"),
    ("the human respiratory system includes the trachea bronchi and lungs",
     "the human respiratory system consists of lungs and airways",
     "same proposition (composition granularity)"),
    ("the process by which plants release oxygen and absorb carbon dioxide is "
     "called respiration",
     "trees absorb carbon dioxide and release oxygen",
     "same gas-exchange proposition (plants/trees)"),
    ("the earths tides are primarily caused by the gravitational pull of the moon",
     "tides are caused by the gravitational interactions between the earth "
     "moon and sun", "same proposition (cause granularity)"),
    ("oxygen is essential for human life", "oxygen is necessary for humans to breathe",
     "same proposition"),
    ("oxygen is essential for human life", "oxygen is essential for respiration",
     "same proposition (life/breathing)"),
    ("humans have five senses sight hearing touch taste and smell",
     "humans have five basic senses", "same proposition + containment"),
    ("the immune system defends the body against pathogens",
     "the human immune system helps protect the body from infections",
     "same proposition"),
    ("the endocrine system regulates body functions through hormones",
     "the human endocrine system produces hormones that regulate various "
     "bodily functions", "same proposition"),
    ("the theory of evolution proposed by charles darwin states that species "
     "evolve over time through natural selection",
     "evolution occurs through the process of natural selection",
     "same proposition (attribution differs)"),
    ("cells are the basic units of life", "the smallest unit of life is the cell",
     "same proposition"),
    ("fossils are the preserved remains or traces of organisms that lived in "
     "the past",
     "fossils provide evidence of past life on earth",
     "same proposition (definition/evidential wording)"),
)

EXCLUDED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("earths atmosphere protects from radiation", "ozone layer protects from UV",
     "different subject (atmosphere vs ozone layer)"),
    ("auroras are caused by solar particles", "auroras occur near polar regions",
     "cause vs location"),
    ("earth has a magnetic field", "magnetic field makes compasses point north",
     "existence vs causal effect"),
    ("atmosphere mostly nitrogen and oxygen", "most abundant gas is nitrogen",
     "composition pair vs single most-abundant"),
    ("diamonds made of carbon", "diamonds are the hardest substance",
     "composition vs hardness"),
    ("blue whale largest mammal", "elephants largest land animals",
     "overall vs land scope"),
    ("moon orbits the earth", "earth has one moon", "orbit vs count"),
    ("sublimation definition", "evaporation definition", "different processes"),
    ("oxygen essential for human life", "water is essential for life",
     "different substance"),
    ("photosynthesis converts sunlight to energy",
     "photosynthesis produces oxygen as a byproduct",
     "energy conversion vs byproduct"),
    ("respiratory system composition", "respiratory system exchanges gases",
     "composition vs function"),
    ("planet name origins (neptune/saturn/mars)", "-", "different planets"),
    ("group-of-animals terms (school/pack/pride)", "-", "different species"),
    ("jupiter largest planet", "jupiter most moons", "different superlatives"),
    ("saturn largest rings", "saturn thousands of rings",
     "largest vs count"),
    ("digestion begins in mouth", "digestive system breaks down food",
     "anatomy-path vs function"),
    ("metamorphosis definition", "butterflies undergo metamorphosis",
     "definition vs instance"),
    ("camouflage helps animals", "polar bears white fur camouflage",
     "general vs instance"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="verify the frozen file is identical in scientific "
                           "content; never write")
    mode.add_argument("--write", action="store_true",
                      help="write/update the frozen file (preserving its original "
                           "created_utc when content is regenerated)")
    args = ap.parse_args()
    csvs = {l: _REPO_ROOT / "datasets" / "training" / "repe"
            / f"true_false_facts_{l}.csv" for l in LANGS}
    stmts = {l: used_fact_statements(
        csvs[l], column="statement" if l == "en" else f"statement_{l}")
        for l in LANGS}
    n = len(stmts["en"])
    mech = duplicate_groups_multilingual(stmts, manual_pairs=())
    uf = _UnionFind(n)
    for i, g in enumerate(mech):
        uf.union(g, i)
    by_norm = {}
    for i, s in enumerate(stmts["en"]):
        by_norm.setdefault(normalize_statement(s), i)
    semantic_resolved = []
    for a, b, reason in SEMANTIC_PAIRS:
        if a not in by_norm or b not in by_norm:
            raise SystemExit(f"semantic pair text not found: "
                             f"{[t for t in (a, b) if t not in by_norm]}")
        uf.union(by_norm[a], by_norm[b])
        semantic_resolved.append({"a": by_norm[a], "b": by_norm[b], "reason": reason})
    groups = [uf.find(i) for i in range(n)]
    members: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    multi = {g: m for g, m in members.items() if len(m) > 1}

    payload = {
        "method": "mechanical union over all five languages (exact normalized "
                  "text + token Jaccard>0.8 for segmented scripts + char "
                  "3-gram Jaccard>0.8 for unsegmented scripts) + frozen "
                  "semantic pair list from a full review of all 306 facts "
                  "(multi-channel candidates + topic-sorted read-through), "
                  "adjudicated before any half-probe fit existed",
        "claim": "disjoint row IDs with duplicate-equivalent facts grouped — "
                 "not absolute semantic independence",
        "fact_universe": "label==1 rows of the training CSVs, file order, "
                         "head(512)",
        "n_facts": n,
        "n_components": len(members),
        "n_multi_member_groups": len(multi),
        "corpus_sha256": {l: hashlib.sha256(csvs[l].read_bytes()).hexdigest()
                          for l in LANGS},
        "groups": {str(g): {"members": m,
                            "statements_en": [stmts["en"][i] for i in m]}
                   for g, m in sorted(multi.items())},
        "semantic_pairs": semantic_resolved,
        "excluded_pairs_reviewed": [
            {"a": a, "b": b, "reason": r} for a, b, r in EXCLUDED_PAIRS],
        "group_assignment": groups,
    }
    canon = json.dumps(payload["group_assignment"]).encode()
    payload["groups_sha256"] = hashlib.sha256(canon).hexdigest()
    out = _REPO_ROOT / MANIFEST_REL
    current = json.loads(out.read_text()) if out.exists() else None
    if args.check:
        if current is None:
            raise SystemExit(f"missing frozen manifest: {out}")
        current_science = {k: v for k, v in current.items() if k != "created_utc"}
        if current_science != payload:
            raise SystemExit(f"{out}: STALE — builder output differs from frozen "
                             f"scientific content; review before --write")
        print(f"{out}: CHECK OK; {n} facts -> {len(members)} components "
              f"({len(multi)} multi-member groups); groups_sha256 "
              f"{payload['groups_sha256'][:16]}...")
        return 0

    payload = {"created_utc": ((current or {}).get("created_utc")
                                or datetime.now(timezone.utc).isoformat()),
               **payload}
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"{out}: {n} facts -> {len(members)} components "
          f"({len(multi)} multi-member groups); groups_sha256 "
          f"{payload['groups_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
