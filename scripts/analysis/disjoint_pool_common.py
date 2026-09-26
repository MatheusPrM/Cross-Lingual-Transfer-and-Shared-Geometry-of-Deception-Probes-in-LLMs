"""
Shared, torch-free primitives for the disjoint-pool robustness analysis.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

SPLIT_SEEDS: tuple[int, ...] = tuple(range(1000, 1010))

OUT_REL = "results/disjoint_pool"
DIRS_NPZ_NAME = "half_directions.npz"
FIT_META_NAME = "fit_meta.json"

NEAR_DUP_JACCARD = 0.8

MANUAL_EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("the sky is blue",
     "the color of the sky is blue"),
    ("mercury is the smallest planet",
     "the smallest planet in our solar system is mercury"),
    ("the largest mammal on earth is the blue whale",
     "the worlds largest mammal is the blue whale"),
    ("the currency of japan is the yen",
     "the currency used in japan is the yen"),
    ("the earths largest ocean is the pacific ocean",
     "the largest ocean in the world is the pacific ocean"),
    ("sound travels through the air as vibrations",
     "sound travels through the air as waves"),
    ("tornadoes are rapidly rotating columns of air",
     "tornadoes are rapidly rotating columns of air that can cause extensive damage"),

    ("water freezes at 0 degrees celsius 32 degrees fahrenheit",
     "water freezes at 0 degrees celsius 32 degrees fahrenheit and boils at "
     "100 degrees celsius 212 degrees fahrenheit"),
)


def used_fact_statements(csv_path, max_rows: int | None = 512,
                         column: str = "statement") -> "list[str]":
    """The EXACT ordered statements the generic probe trains on: label==1
    rows in file order, then head(max_rows) — replicating
    project.repe.loading.load_repe's filtering (pinned by test against the
    loader's own dialogue count). ``column`` selects the localized text
    (statement_<lang>) for non-English corpora."""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[df["label"] == 1].copy()
    if max_rows is not None:
        df = df.head(max_rows)
    return [str(s).strip() for s in df[column] if str(s).strip()]


def verify_row_parallel(csv_paths: "dict[str, str | object]") -> "dict":
    """Preflight contract across the five training CSVs (run BEFORE any GPU
    work). Verifies, row by row: identical ordered English `statement`
    column, identical `label` column, and non-empty localized text for every
    USED (label==1) row. Returns per-language sha256s + used-fact count;
    raises SystemExit with a precise complaint otherwise."""
    import pandas as pd
    frames = {}
    for lang, p in csv_paths.items():
        frames[lang] = pd.read_csv(p, encoding="utf-8")
    ref_lang = "en" if "en" in frames else sorted(frames)[0]
    ref = frames[ref_lang]
    out = {}
    for lang, df in frames.items():
        if len(df) != len(ref):
            raise SystemExit(f"ROW-PARALLEL CONTRACT: {lang} has {len(df)} rows, "
                             f"{ref_lang} has {len(ref)}")
        if not (df["label"].astype(int) == ref["label"].astype(int)).all():
            bad = int((df["label"].astype(int) != ref["label"].astype(int)).idxmax())
            raise SystemExit(f"ROW-PARALLEL CONTRACT: label mismatch {lang} vs "
                             f"{ref_lang} at row {bad}")
        if not (df["statement"].astype(str) == ref["statement"].astype(str)).all():
            bad = int((df["statement"].astype(str) != ref["statement"].astype(str)).idxmax())
            raise SystemExit(f"ROW-PARALLEL CONTRACT: source statement differs in "
                             f"{lang} at row {bad} — fact indices would misalign")
        col = "statement" if lang == "en" else f"statement_{lang}"
        if col not in df.columns:
            raise SystemExit(f"ROW-PARALLEL CONTRACT: {lang} missing column {col}")
        used = df[df["label"].astype(int) == 1]
        empty = used[used[col].fillna("").astype(str).str.strip() == ""]
        if len(empty):
            raise SystemExit(f"ROW-PARALLEL CONTRACT: {lang}: {len(empty)} used "
                             f"(label==1) rows have empty {col}: rows "
                             f"{empty.index.tolist()[:5]}")
        with open(csv_paths[lang], "rb") as fh:
            out[lang] = {"sha256": hashlib.sha256(fh.read()).hexdigest(),
                         "rows": int(len(df)),
                         "n_used_facts": int((df["label"].astype(int) == 1).sum())}
    return out


def normalize_statement(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def _char_ngrams(s: str, n: int = 3) -> frozenset:
    s = normalize_statement(s).replace(" ", "")
    if len(s) < n:
        return frozenset([s]) if s else frozenset()
    return frozenset(s[i:i + n] for i in range(len(s) - n + 1))


def similarity_features(statements: "list[str]") -> "tuple[list[frozenset], str]":
    """Per-statement comparison sets + the channel used: word tokens for
    segmented scripts, character 3-grams for unsegmented scripts (zh) —
    whitespace tokens there are single blobs, which made the earlier zh
    lexical audit vacuous."""
    toks = [frozenset(normalize_statement(s).split()) for s in statements]
    if float(np.median([len(t) for t in toks])) >= 2:
        return toks, "token"
    return [_char_ngrams(s) for s in statements], "char_3gram"


def _merge_one_language(uf: "_UnionFind", statements: "list[str]",
                        jaccard: float) -> None:
    normed = [normalize_statement(s) for s in statements]
    first_seen: dict[str, int] = {}
    for i, s in enumerate(normed):
        if s in first_seen:
            uf.union(first_seen[s], i)
        else:
            first_seen[s] = i
    feats, _channel = similarity_features(statements)
    n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            inter = len(feats[i] & feats[j])
            if inter and inter / len(feats[i] | feats[j]) > jaccard:
                uf.union(i, j)


def duplicate_groups(statements: "list[str]",
                     jaccard: float = NEAR_DUP_JACCARD) -> "list[int]":
    """Single-language grouping (exact normalized text + token Jaccard).
    Production code uses duplicate_groups_multilingual; this remains for
    single-corpus scans and tests."""
    uf = _UnionFind(len(statements))
    _merge_one_language(uf, statements, jaccard)
    return [uf.find(i) for i in range(len(statements))]


def duplicate_groups_multilingual(
    statements_by_lang: "dict[str, list[str]]",
    jaccard: float = NEAR_DUP_JACCARD,
    manual_pairs: "tuple[tuple[str, str], ...]" = MANUAL_EQUIVALENT_PAIRS,
    reference_lang: str = "en",
) -> "list[int]":
    """Group id per fact index: the UNION of every language's exact +
    lexical merges (translations can collapse distinct English phrasings
    into identical localized text) plus the frozen manual semantic-
    equivalence list, matched on normalized reference-language text. A
    manual statement that is absent from the corpus fails loudly — silent
    no-ops would rot the list."""
    langs = sorted(statements_by_lang)
    n_set = {len(v) for v in statements_by_lang.values()}
    if len(n_set) != 1:
        raise ValueError(f"statement lists differ in length: {n_set}")
    n = n_set.pop()
    uf = _UnionFind(n)
    for lang in langs:
        _merge_one_language(uf, statements_by_lang[lang], jaccard)
    if manual_pairs:
        ref = statements_by_lang[reference_lang]
        by_norm: dict[str, int] = {}
        for i, s in enumerate(ref):
            by_norm.setdefault(normalize_statement(s), i)
        for a, b in manual_pairs:
            if a not in by_norm or b not in by_norm:
                missing = [t for t in (a, b) if t not in by_norm]
                raise ValueError(f"manual equivalence text not found in the "
                                 f"{reference_lang} corpus: {missing}")
            uf.union(by_norm[a], by_norm[b])
    return [uf.find(i) for i in range(n)]


MANIFEST_REL = "datasets/training/repe/fact_equivalence_manifest.json"


def load_equivalence_manifest(repo_root, csv_paths: "dict[str, object]",
                              statements_by_lang: "dict[str, list[str]]") -> "tuple[list[int], str]":
    """Load the FROZEN fact-equivalence manifest and certify it against the
    corpus on disk. Three fatal gates:
      1. the manifest's per-language corpus sha256s must match the CSVs;
      2. the fact count must match;
      3. every merge the MECHANICAL multilingual pass finds on today's data
         must already be contained inside a manifest group — otherwise the
         manifest is stale (corpus changed after the review) and fitting on
         it would resurrect duplicate leakage.
    Returns (group assignment, manifest groups_sha256)."""
    import hashlib as _hashlib
    import json as _json
    from pathlib import Path as _Path
    mp = _Path(repo_root) / MANIFEST_REL
    if not mp.exists():
        raise SystemExit(f"missing frozen equivalence manifest {mp} — run "
                         f"scripts/analysis/build_fact_equivalence_manifest.py "
                         f"and commit it before fitting.")
    m = _json.loads(mp.read_text())
    for lang, p in csv_paths.items():
        sha = _hashlib.sha256(_Path(p).read_bytes()).hexdigest()
        if m["corpus_sha256"].get(lang) != sha:
            raise SystemExit(f"equivalence manifest is STALE: {lang} corpus "
                             f"sha mismatch — re-run the review/builder.")
    groups = list(m["group_assignment"])
    n = len(statements_by_lang[sorted(statements_by_lang)[0]])
    if len(groups) != n or int(m["n_facts"]) != n:
        raise SystemExit(f"equivalence manifest covers {m['n_facts']} facts, "
                         f"corpus has {n}")
    recomputed = _hashlib.sha256(_json.dumps(groups).encode()).hexdigest()
    if recomputed != m.get("groups_sha256"):
        raise SystemExit("equivalence manifest is STALE: declared groups_sha256 "
                         "does not match the recomputed hash of "
                         "group_assignment — the manifest was hand-edited or "
                         "corrupted; re-run the builder.")
    mech = duplicate_groups_multilingual(statements_by_lang, manual_pairs=())
    mg: dict[int, set[int]] = {}
    for i, g in enumerate(mech):
        mg.setdefault(g, set()).add(i)
    for g, mem in mg.items():
        manifest_sides = {groups[i] for i in mem}
        if len(manifest_sides) > 1:
            raise SystemExit(f"equivalence manifest is STALE: mechanical pass "
                             f"merges facts {sorted(mem)} but the manifest "
                             f"separates them — re-run the review/builder.")
    return groups, m["groups_sha256"]


def groups_sha256(statements: "list[str]", groups: "list[int]") -> str:
    payload = "\n".join(f"{g}\t{normalize_statement(s)}"
                        for s, g in zip(statements, groups))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fact_halves(n_facts: int, seed: int,
                groups: "list[int] | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministic half split over fact indices 0..n_facts-1.

    With ``groups`` (production path): duplicate-equivalent facts share a
    group and the split permutes GROUPS, assigning whole groups to half A,
    then deterministically repairs any imbalance by moving singleton groups
    (in reverse assignment order) until half A holds EXACTLY n_facts // 2
    facts — equal sample size is part of the design contract ("only the
    overlap differs"), so an unbalanced split raises instead of shipping.

    Without ``groups`` (compatibility/reference path): identical permutation
    logic to probe_reliability.py's split-half loop (rng.permutation over
    facts, first half = A), so seed s reproduces the reliability convention.
    """
    if n_facts < 4:
        raise ValueError(f"n_facts={n_facts}: too few facts to split")
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n_facts)
        return np.sort(perm[: n_facts // 2]), np.sort(perm[n_facts // 2:])
    if len(groups) != n_facts:
        raise ValueError(f"groups has {len(groups)} entries for {n_facts} facts")
    members: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    gids = sorted(members)
    gperm = rng.permutation(len(gids))
    target = n_facts // 2
    a_groups: list[int] = []   # gids assigned to A, in assignment order
    n_a = 0
    for gi in gperm:
        if n_a >= target:
            break
        a_groups.append(gids[gi])
        n_a += len(members[gids[gi]])
    b_groups = [g for g in (gids[gi] for gi in gperm) if g not in set(a_groups)]

    while n_a > target:
        sing = next((g for g in reversed(a_groups) if len(members[g]) == 1), None)
        if sing is None:
            raise ValueError(f"seed {seed}: cannot balance halves exactly — "
                             f"no singleton group left to move (n_a={n_a}, "
                             f"target={target})")
        a_groups.remove(sing)
        b_groups.insert(0, sing)
        n_a -= 1
    while n_a < target:
        sing = next((g for g in b_groups if len(members[g]) == 1), None)
        if sing is None:
            raise ValueError(f"seed {seed}: cannot balance halves exactly — "
                             f"no singleton group left to move (n_a={n_a}, "
                             f"target={target})")
        b_groups.remove(sing)
        a_groups.append(sing)
        n_a += 1
    a = sorted(i for g in a_groups for i in members[g])
    a_set = set(a)
    b = [i for i in range(n_facts) if i not in a_set]
    if len(a) != target:
        raise AssertionError(f"seed {seed}: balance repair failed ({len(a)} != {target})")
    return np.array(a, dtype=int), np.array(b, dtype=int)


def unit_rows(v: np.ndarray) -> np.ndarray:
    """Row-normalize a stack of directions (no-op on already-unit rows)."""
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def cosine_contrast_rows(
    model: str,
    dirs_by_lang: "dict[str, np.ndarray]",
    seeds: "tuple[int, ...] | list[int]",
) -> "list[dict]":
    """The primary readout for the item-matching concern.

    dirs_by_lang[lang] has shape (n_seeds, 2, d): unit effective directions of
    probes trained on half A ([:,0,:]) and half B ([:,1,:]) of the same seed's
    split. Per unordered language pair and seed:
      matched  = mean(cos(A_T,A_L), cos(B_T,B_L))   same facts, half n
      disjoint = mean(cos(A_T,B_L), cos(B_T,A_L))   zero shared facts, half n
      contrast = disjoint - matched
    Symmetrizing over A/A + B/B vs A/B + B/A uses all four probes equally,
    controlling half size AND half quality. Only the train-pool overlap
    differs between the two conditions.
    """
    langs = sorted(dirs_by_lang)
    rows: list[dict] = []
    for i, t in enumerate(langs):
        for l in langs[i + 1:]:
            dt = unit_rows(dirs_by_lang[t])
            dl = unit_rows(dirs_by_lang[l])
            for si, seed in enumerate(seeds):
                caa = float(np.dot(dt[si, 0], dl[si, 0]))
                cbb = float(np.dot(dt[si, 1], dl[si, 1]))
                cab = float(np.dot(dt[si, 0], dl[si, 1]))
                cba = float(np.dot(dt[si, 1], dl[si, 0]))
                matched = 0.5 * (caa + cbb)
                disjoint = 0.5 * (cab + cba)
                rows.append({
                    "model": model, "lang_a": t, "lang_b": l, "seed": int(seed),
                    "cos_matched_aa": caa, "cos_matched_bb": cbb,
                    "cos_disjoint_ab": cab, "cos_disjoint_ba": cba,
                    "cos_matched_mean": matched, "cos_disjoint_mean": disjoint,
                    "contrast_disjoint_minus_matched": disjoint - matched,
                })
    return rows


def summarize_cosine_contrast(rows: "list[dict]") -> "list[dict]":
    """Descriptive summaries for each pair and model.

    Per-model aggregation is built from seed-level averages (per seed, mean
    over the ten language pairs). Because the ten partitions overlap, return
    every seed value plus its mean, SD, and range; do not construct a CI or
    treat seeds as independent replicates.
    """

    out: list[dict] = []
    keyed: dict[tuple, list[dict]] = {}
    for r in rows:
        keyed.setdefault((r["model"], r["lang_a"], r["lang_b"]), []).append(r)
    per_model_seed: dict[str, dict[int, list[float]]] = {}
    for (model, a, b), rs in sorted(keyed.items()):
        rs = sorted(rs, key=lambda r: int(r["seed"]))
        c = np.array([r["contrast_disjoint_minus_matched"] for r in rs])
        m = np.array([r["cos_matched_mean"] for r in rs])
        d = np.array([r["cos_disjoint_mean"] for r in rs])
        n = len(c)
        out.append({
            "model": model, "lang_a": a, "lang_b": b, "scope": "pair",
            "n_seeds": n,
            "cos_matched_mean": float(m.mean()),
            "cos_disjoint_mean": float(d.mean()),
            "contrast_mean": float(c.mean()),
            "contrast_sd": float(np.std(c, ddof=1)) if n > 1 else 0.0,
            "contrast_seed_min": float(c.min()),
            "contrast_seed_max": float(c.max()),
            "per_seed_contrasts": ";".join(f"{v:.6f}" for v in c),
        })
        for r in rs:
            per_model_seed.setdefault(model, {}).setdefault(
                int(r["seed"]), []).append(r["contrast_disjoint_minus_matched"])
    for model, by_seed in sorted(per_model_seed.items()):
        seed_means = np.array([float(np.mean(v)) for _, v in sorted(by_seed.items())])
        n = len(seed_means)
        out.append({
            "model": model, "lang_a": "ALL", "lang_b": "ALL",
            "scope": "model_mean_of_seed_means", "n_seeds": n,
            "cos_matched_mean": float("nan"), "cos_disjoint_mean": float("nan"),
            "contrast_mean": float(seed_means.mean()),
            "contrast_sd": float(np.std(seed_means, ddof=1)) if n > 1 else 0.0,
            "contrast_seed_min": float(seed_means.min()),
            "contrast_seed_max": float(seed_means.max()),
            "per_seed_contrasts": ";".join(f"{v:.6f}" for v in seed_means),
        })
    return out


def validate_fit_artifact(
    model_dir,
    expected_langs=None,
    *,
    expected_model=None,
    expected_seeds=None,
    expected_n_facts=None,
    expected_layer=None,
    expected_model_revision=None,
    expected_dtype=None,
    expected_groups=None,
    expected_groups_sha256=None,
    expected_train_csvs=None,
    expected_launch_commit=None,
    expected_protocol_sha256=None,
    expected_constraints_sha256=None,
    expected_hidden=None,
) -> "dict":
    """Strict validation of a half-probe artifact (resume must never trust
    file existence alone). All checks are
    fatal: complete meta; npz present with sha256 MATCHING the meta binding;
    per-language arrays shaped (n_seeds, 2, d) float32, finite, unit-norm;
    halves exactly balanced, disjoint, covering; groups length == n_facts;
    the registered detector/precision/parity contract; and immutable
    provenance fields. Optional expected_* arguments bind a resume decision
    to the CURRENT production checkout rather than merely accepting a
    structurally valid artifact. Returns a summary dict."""
    import hashlib as _hashlib
    import json as _json
    from pathlib import Path as _Path
    d = _Path(model_dir)
    mp, np_path = d / FIT_META_NAME, d / DIRS_NPZ_NAME
    if not mp.exists():
        raise SystemExit(f"{d}: missing {FIT_META_NAME}")
    try:
        meta = _json.loads(mp.read_text())
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"{mp}: unreadable meta ({e})")
    if not meta.get("complete"):
        raise SystemExit(f"{mp}: not marked complete — interrupted run; refit.")

    required = {
        "artifact_schema", "model", "category", "layer", "model_revision",
        "extraction_seed", "param_dtype", "off_record_dtype_override",
        "expected_dtype", "precision", "seeds", "n_facts_used", "langs",
        "halves_by_seed", "duplicate_groups", "groups_sha256", "detector",
        "fitter_parity_cos", "parity_min", "train_csvs", "launch_commit",
        "tracked_tree_dirty", "protocol_sha256", "constraints_sha256",
        "dirs_npz_sha256",
    }
    missing = sorted(required - set(meta))
    if missing:
        raise SystemExit(f"{mp}: missing required production metadata: {missing}")

    if meta["artifact_schema"] != "disjoint-pool-half-probes/v1":
        raise SystemExit(f"{d}: unsupported artifact schema "
                         f"{meta['artifact_schema']!r}")
    if int(meta["extraction_seed"]) != 42:
        raise SystemExit(f"{d}: extraction seed {meta['extraction_seed']} != 42")

    model = str(meta["model"])
    if expected_model is not None and model != str(expected_model):
        raise SystemExit(f"{d}: artifact model {model!r} != expected "
                         f"{str(expected_model)!r}")
    if meta["category"] != "generic":
        raise SystemExit(f"{d}: category {meta['category']!r} is not registered 'generic'")
    if expected_layer is not None and int(meta["layer"]) != int(expected_layer):
        raise SystemExit(f"{d}: layer {meta['layer']} != frozen layer {expected_layer}")
    if not meta["model_revision"]:
        raise SystemExit(f"{d}: missing resolved model revision")
    if (expected_model_revision is not None
            and meta["model_revision"] != expected_model_revision):
        raise SystemExit(f"{d}: model revision {meta['model_revision']} != locked "
                         f"revision {expected_model_revision}")

    detector = meta["detector"]
    detector_ok = (
        detector.get("kind") == "LogisticRegressionDetector"
        and detector.get("unit") == "dialogue"
        and float(detector.get("reg_coeff", float("nan"))) == 10.0
        and detector.get("scale") is True
    )
    if not detector_ok:
        raise SystemExit(f"{d}: detector settings are not the registered "
                         f"LogisticRegressionDetector/dialogue/reg=10/scale=True: "
                         f"{detector}")

    if meta["param_dtype"] != meta["expected_dtype"]:
        raise SystemExit(f"{d}: parameter dtype {meta['param_dtype']} != artifact's "
                         f"registered dtype {meta['expected_dtype']}")
    if expected_dtype is not None and meta["param_dtype"] != expected_dtype:
        raise SystemExit(f"{d}: parameter dtype {meta['param_dtype']} != current "
                         f"run-of-record dtype {expected_dtype}")
    if meta["off_record_dtype_override"] is not False:
        raise SystemExit(f"{d}: artifact used --allow-dtype/off-record precision")
    precision = meta["precision"]
    if precision.get("quantized_4bit") is not False:
        raise SystemExit(f"{d}: precision stamp is quantized or ambiguous: {precision}")
    if precision.get("param_dtype") != meta["param_dtype"]:
        raise SystemExit(f"{d}: precision dtype disagrees with top-level dtype")
    if (str(precision.get("project_use_4bit_env")) != "0"
            or str(precision.get("project_4bit_threshold_env", "")) != ""):
        raise SystemExit(f"{d}: precision environment is not the registered "
                         f"PROJECT_USE_4BIT=0/empty-threshold regime: {precision}")
    if precision.get("cuda_available") is not True \
            or str(precision.get("device_name", "")).lower() == "cpu":
        raise SystemExit(f"{d}: artifact was not produced on a CUDA device: {precision}")

    for key in ("groups_sha256", "protocol_sha256", "constraints_sha256"):
        val = meta[key]
        if key == "protocol_sha256" and val is None:
            continue
        if not isinstance(val, str) or len(val) != 64:
            raise SystemExit(f"{d}: invalid {key}: {val!r}")
    launch_commit = meta["launch_commit"]
    if (not isinstance(launch_commit, str) or len(launch_commit) != 40
            or any(c not in "0123456789abcdef" for c in launch_commit.lower())):
        raise SystemExit(f"{d}: invalid launch_commit: {launch_commit!r}")
    if expected_groups_sha256 is not None \
            and meta["groups_sha256"] != expected_groups_sha256:
        raise SystemExit(f"{d}: equivalence-manifest hash differs from current freeze")
    if expected_launch_commit is not None and launch_commit != expected_launch_commit:
        raise SystemExit(f"{d}: launch commit {launch_commit} != current commit "
                         f"{expected_launch_commit}; do not resume across code revisions")
    if expected_protocol_sha256 is not None \
            and meta["protocol_sha256"] != expected_protocol_sha256:
        raise SystemExit(f"{d}: protocol hash differs from current checkout")
    if expected_constraints_sha256 is not None \
            and meta["constraints_sha256"] != expected_constraints_sha256:
        raise SystemExit(f"{d}: probe-environment constraints hash differs from "
                         f"current checkout")
    if meta.get("tracked_tree_dirty") is not False:
        raise SystemExit(f"{d}: fit was launched from a dirty tracked worktree")

    if not np_path.exists():
        raise SystemExit(f"{d}: missing {DIRS_NPZ_NAME}")
    sha = _hashlib.sha256(np_path.read_bytes()).hexdigest()
    if meta.get("dirs_npz_sha256") != sha:
        raise SystemExit(f"{d}: npz sha256 {sha[:12]}... does not match the "
                         f"meta binding {str(meta.get('dirs_npz_sha256'))[:12]}"
                         f"... — corrupted or mixed artifact; refit.")
    langs = meta["langs"]
    if expected_langs is not None and sorted(langs) != sorted(expected_langs):
        raise SystemExit(f"{d}: langs {sorted(langs)} != expected "
                         f"{sorted(expected_langs)}")
    seeds = [int(s) for s in meta["seeds"]]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise SystemExit(f"{d}: split seeds are empty or duplicated: {seeds}")
    if expected_seeds is not None and seeds != [int(s) for s in expected_seeds]:
        raise SystemExit(f"{d}: seeds {seeds} != registered {list(expected_seeds)}")
    n_facts = int(meta["n_facts_used"])
    if expected_n_facts is not None and n_facts != int(expected_n_facts):
        raise SystemExit(f"{d}: n_facts {n_facts} != registered {expected_n_facts}")
    if n_facts <= 0 or n_facts % 2:
        raise SystemExit(f"{d}: n_facts must be positive and even, got {n_facts}")
    groups = meta.get("duplicate_groups")
    if groups is None or len(groups) != n_facts:
        raise SystemExit(f"{d}: duplicate_groups missing or wrong length")
    groups_hash = _hashlib.sha256(_json.dumps(groups).encode()).hexdigest()
    if groups_hash != meta["groups_sha256"]:
        raise SystemExit(f"{d}: duplicate_groups do not match groups_sha256")
    if expected_groups is not None and list(groups) != list(expected_groups):
        raise SystemExit(f"{d}: duplicate_groups differ from current frozen manifest")
    if set(meta["halves_by_seed"]) != {str(s) for s in seeds}:
        raise SystemExit(f"{d}: halves_by_seed keys do not exactly match seeds")
    target = n_facts // 2
    for s in seeds:
        h = meta["halves_by_seed"][str(s)]
        a_list, b_list = [int(i) for i in h["a"]], [int(i) for i in h["b"]]
        a, b = set(a_list), set(b_list)
        if (len(a_list) != target or len(b_list) != target
                or len(a) != target or len(b) != target
                or (a & b) or (a | b) != set(range(n_facts))):
            raise SystemExit(f"{d}: seed {s} halves unbalanced/overlapping")
        expected_a, expected_b = fact_halves(n_facts, s, groups=groups)
        if a_list != expected_a.tolist() or b_list != expected_b.tolist():
            raise SystemExit(f"{d}: seed {s} halves differ from deterministic "
                             f"registered split")
        split = {groups[i] for i in a_list} & {groups[i] for i in b_list}
        if split:
            raise SystemExit(f"{d}: seed {s} splits duplicate groups {split}")

    parity = meta["fitter_parity_cos"]
    if set(parity) != set(langs):
        raise SystemExit(f"{d}: parity languages {sorted(parity)} != {sorted(langs)}")
    try:
        parity_min = float(meta["parity_min"])
        parity_vals = {l: float(parity[l]) for l in langs}
    except (TypeError, ValueError) as e:
        raise SystemExit(f"{d}: invalid parity metadata ({e})")
    if not np.isfinite(parity_min) or not all(np.isfinite(v) for v in parity_vals.values()):
        raise SystemExit(f"{d}: non-finite parity metadata")
    if parity_min != 0.98:
        raise SystemExit(f"{d}: parity_min {parity_min} != registered 0.98")
    if any(v < -1.0 or v > 1.0 for v in parity_vals.values()):
        raise SystemExit(f"{d}: fitter parity cosine outside [-1,1]: {parity_vals}")
    if any(v < parity_min for v in parity_vals.values()):
        raise SystemExit(f"{d}: fitter parity below {parity_min}: {parity_vals}")

    train_csvs = meta["train_csvs"]
    if set(train_csvs) != set(langs):
        raise SystemExit(f"{d}: train_csvs languages do not match artifact languages")
    for lang, stamp in train_csvs.items():
        if len(str(stamp.get("sha256", ""))) != 64 \
                or int(stamp.get("n_used_facts", -1)) != n_facts:
            raise SystemExit(f"{d}: invalid training CSV stamp for {lang}: {stamp}")
    if expected_train_csvs is not None and train_csvs != expected_train_csvs:
        raise SystemExit(f"{d}: training CSV stamps differ from the current corpus")
    dims = set()
    with np.load(np_path) as z:
        if sorted(z.files) != sorted(langs):
            raise SystemExit(f"{d}: npz keys {sorted(z.files)} != langs {sorted(langs)}")
        for lang in langs:
            arr = z[lang]
            if arr.dtype != np.float32 or arr.ndim != 3 \
                    or arr.shape[0] != len(seeds) or arr.shape[1] != 2:
                raise SystemExit(f"{d}: {lang} array shape/dtype wrong: "
                                 f"{arr.shape} {arr.dtype}")
            if not np.isfinite(arr).all():
                raise SystemExit(f"{d}: {lang} has non-finite directions")
            norms = np.linalg.norm(arr, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise SystemExit(f"{d}: {lang} directions not unit-norm "
                                 f"(|n-1| up to {abs(norms - 1).max():.2e})")
            dims.add(arr.shape[-1])
    if len(dims) != 1:
        raise SystemExit(f"{d}: inconsistent hidden dims {dims}")
    hidden = dims.pop()
    if expected_hidden is not None and hidden != int(expected_hidden):
        raise SystemExit(f"{d}: hidden dimension {hidden} != stored paper-probe "
                         f"dimension {expected_hidden}")
    return {"model": model, "langs": sorted(langs),
            "n_seeds": len(seeds), "hidden": hidden,
            "n_facts": n_facts, "sha256": sha,
            "fitter_parity_cos": parity_vals,
            "param_dtype": meta["param_dtype"],
            "expected_dtype": meta["expected_dtype"],
            "launch_commit": launch_commit}


def near_duplicate_scan(
    statements: "list[str]", seeds: "tuple[int, ...] | list[int]",
    groups: "list[int] | None" = None,
    high_threshold: float = 0.6,
) -> "list[dict]":
    """Residual-leakage check on the USED fact statements under the SAME
    (group-aware) halves the fits use. Same-group members can no longer span
    halves by construction — the scan certifies that (n_same_group_split
    must be 0) and quantifies the remaining cross-half lexical overlap."""
    if groups is None:
        groups = duplicate_groups(statements)
    feats, channel = similarity_features(statements)
    n = len(feats)
    rows: list[dict] = []
    for seed in seeds:
        a, b = fact_halves(n, seed, groups=groups)
        in_a = set(a.tolist())
        same_group_split = sum(
            1 for gid in set(groups)
            if len({(i in in_a) for i, g in enumerate(groups) if g == gid}) > 1)
        b_idx = list(b)
        maxima = []
        argmax = []
        for i in a:
            ti = feats[i]
            best, best_j = 0.0, -1
            for j in b_idx:
                inter = len(ti & feats[j])
                if inter:
                    jac = inter / len(ti | feats[j])
                    if jac > best:
                        best, best_j = jac, j
            maxima.append(best)
            argmax.append((int(i), int(best_j), best))
        m = np.array(maxima)
        top = sorted(argmax, key=lambda t: -t[2])[:3]
        rows.append({
            "seed": int(seed), "n_half_a": int(len(a)), "n_half_b": int(len(b)),
            "n_same_group_split": int(same_group_split),
            "channel": channel,
            "max_jaccard_mean": float(m.mean()),
            "max_jaccard_p95": float(np.percentile(m, 95)),
            "max_jaccard_max": float(m.max()),
            "n_above_threshold": int((m > high_threshold).sum()),
            "threshold": high_threshold,
            "top_pairs": ";".join(f"{i}~{j}:{v:.2f}" for i, j, v in top if j >= 0),
        })
    return rows
