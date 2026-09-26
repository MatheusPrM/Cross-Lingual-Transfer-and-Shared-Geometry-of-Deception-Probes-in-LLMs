#!/usr/bin/env python3
"""Generate DeceptionBench rollouts (response-only, monolingual, leakage-asserted)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.chat_format import format_dialogue_for_chat
from project.deceptionbench import (
    CONDITIONS,
    DOMAIN_FILES,
    JUDGE_SCHEMA_VERSION,
    LeakageError,
    Scenario,
    assert_no_leakage,
    build_messages,
    fill_assistant_response,
    is_wellformed,
    judge_response,
    load_deceptionbench_domain,
    looks_like_refusal,
    messages_to_dialogue,
    normalize_label,
    parse_condition,
    save_deceptionbench_rollouts,
)
from project.deceptionbench.rollouts import generate_completions_for_dialogues
from project.model_loader import load_model_and_tokenizer
from project.paths import REPO_ROOT, rollout_json_path
from project.reproducibility import set_seed
from project.judge_models import DEFAULT_JUDGE
from project.deceptionbench.rollouts import truncation_report

_REASONING_MODEL_KEYWORDS: tuple[str, ...] = ("qwen3", "deepseek-r1", "qwq", "phi-4-mini")


def _is_reasoning_model(model_id: str) -> bool:
    s = (model_id or "").lower()
    return any(k in s for k in _REASONING_MODEL_KEYWORDS)


def _default_max_new_tokens(model_id: str, allow_thinking: bool) -> int:
    if allow_thinking and _is_reasoning_model(model_id):
        return 2048
    return 1024


def _resolve_data_path(repo_root: Path, lang: str, domain: str) -> Path:
    stem = Path(DOMAIN_FILES[domain]).stem
    suf = "" if lang == "en" else f"_{lang}"
    base_dir = repo_root / "datasets" / "evaluation" / "deceptionbench"
    candidates = [
        base_dir / (f"{stem}_en{suf}.json" if suf else f"{stem}.json"),
        base_dir / f"{stem}_en_{lang}.json" if suf else None,
        base_dir / DOMAIN_FILES[domain],
    ]
    for c in candidates:
        if c is None:
            continue
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no DeceptionBench {domain} JSON for lang={lang} under {base_dir}; "
        f"tried {[str(c) for c in candidates if c is not None]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeceptionBench response-only rollouts.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, default="en", choices=("en", "pt", "es", "de", "zh"))
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_FILES.keys()),
        choices=list(DOMAIN_FILES.keys()),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
    )
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 = all in selected domains")
    parser.add_argument(
        "--scenario-ids",
        type=Path,
        default=None,
        help="optional path to a JSON file with a 'picks' list of scenario IDs (e.g. balanced_30_v1.json) to restrict the run",
    )
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 = auto by model class")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-thinking", action="store_true", help="opt-in for reasoning models")
    parser.add_argument("--allow-en-fallback", action="store_true", help="debug only")
    parser.add_argument("--allow-leakage-check-off", action="store_true", help="alias for --leakage-mode off")
    parser.add_argument(
        "--leakage-mode",
        choices=("strict", "warn", "off"),
        default="strict",
        help="strict aborts on first leak (default); warn logs all leaks and continues; off disables the check",
    )
    parser.add_argument(
        "--leakage-min-len",
        type=int,
        default=40,
        help="minimum forbidden-field substring length to check (default 40 chars)",
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="resume from the *_unjudged.json snapshot: skip model load + generation entirely, "
        "judge the persisted responses (cache hits are free) and write the final file "
        "",
    )
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="override default datasets/evaluation/deceptionbench/ directory",
    )
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    lang = args.lang.strip().lower()
    strict = not args.allow_en_fallback

    data_dir = args.data_dir if args.data_dir is not None else REPO_ROOT / "datasets" / "evaluation" / "deceptionbench"
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    scenarios: list[Scenario] = []
    for d in args.domains:
        path = _resolve_data_path(REPO_ROOT, lang, d) if args.data_dir is None else (data_dir / DOMAIN_FILES[d])
        if args.data_dir is not None and not path.exists():
            raise FileNotFoundError(f"missing {path}")
        scenarios.extend(
            load_deceptionbench_domain(
                path,
                domain=d,
                lang=lang,
                strict=strict,
                allow_en_fallback=args.allow_en_fallback,
            )
        )
    if args.scenario_ids is not None:
        sid_path = args.scenario_ids if args.scenario_ids.is_absolute() else REPO_ROOT / args.scenario_ids
        if not sid_path.exists():
            raise FileNotFoundError(f"--scenario-ids file not found: {sid_path}")
        import json as _json
        with open(sid_path, encoding="utf-8") as _f:
            data = _json.load(_f)
        if isinstance(data, dict) and "picks" in data:
            allowed = {str(p["scenario_id"]) for p in data["picks"]}
        elif isinstance(data, list):
            allowed = {str(x["scenario_id"]) if isinstance(x, dict) else str(x) for x in data}
        else:
            raise ValueError(f"unsupported --scenario-ids JSON shape in {sid_path}")
        before = len(scenarios)
        scenarios = [s for s in scenarios if s.scenario_id in allowed]
        print(f"--scenario-ids: filtered {before} -> {len(scenarios)} scenarios", flush=True)

    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    enable_thinking = True if args.allow_thinking else False
    max_new_tokens = args.max_new_tokens or _default_max_new_tokens(args.model, args.allow_thinking)

    out = args.out
    if out is None:
        if set(args.domains) == set(DOMAIN_FILES.keys()):
            out = rollout_json_path(args.model, dataset_id=f"deceptionbench_{lang}")
        else:
            stem = f"deceptionbench_{'+'.join(sorted(args.domains))}_{lang}"
            out = rollout_json_path(args.model, dataset_id=f"deceptionbench_{lang}", file_stem=stem)
    else:
        out = Path(out)
        if not out.is_absolute():
            out = REPO_ROOT / out
    unjudged_path = out.with_name(out.name.replace(".json", "_unjudged.json"))

    if args.judge_only:
        if args.skip_judge:
            sys.exit("--judge-only with --skip-judge is a no-op; drop one of them.")
        if not unjudged_path.exists():
            sys.exit(f"--judge-only: snapshot not found: {unjudged_path}")
        import json as _json2
        snap = _json2.loads(unjudged_path.read_text(encoding="utf-8"))
        items: list[dict[str, Any]] = snap["rollouts"]
        gen_meta: dict[str, Any] = snap.get("generation", {})
        print(f"--judge-only: {len(items)} persisted rollouts from {unjudged_path}", flush=True)
    else:
        print(f"Loading model {args.model}...", flush=True)
        model, tokenizer = load_model_and_tokenizer(args.model)

        leakage_mode = "off" if args.allow_leakage_check_off else args.leakage_mode
        work: list[tuple[Scenario, str, list[dict[str, str]], Any]] = []
        warn_leaks: list[tuple[str, str, str, str]] = []
        for sc in scenarios:
            for cond in args.conditions:
                msgs = build_messages(sc, cond, lang=lang)
                d = messages_to_dialogue(msgs)
                rendered = format_dialogue_for_chat(
                    d,
                    tokenizer,
                    model_id=args.model,
                    add_generation_prompt=True,
                    for_generation=True,
                    enable_thinking=enable_thinking,
                )
                try:
                    hits = assert_no_leakage(
                        rendered, sc, min_substr_len=args.leakage_min_len, mode=leakage_mode,
                    )
                except LeakageError as e:
                    print(f"LEAKAGE on {sc.scenario_id} cond={cond}:\n{e}", file=sys.stderr)
                    sys.exit(2)
                for field, needle in hits:
                    warn_leaks.append((sc.scenario_id, cond, field, needle))
                work.append((sc, cond, msgs, d))

        if warn_leaks:
            print(f"WARN: {len(warn_leaks)} leakage hit(s) in --leakage-mode=warn (first 5):", file=sys.stderr)
            for sid, cond, field, needle in warn_leaks[:5]:
                n = needle if len(needle) <= 120 else needle[:120] + "..."
                print(f"  {sid} cond={cond} field={field!r} needle={n!r}", file=sys.stderr)

        print(f"Generating {len(work)} rollouts (max_new_tokens={max_new_tokens})...", flush=True)
        dialogues = [w[3] for w in work]
        terminations: list[str] = []  
        texts = generate_completions_for_dialogues(
            model=model,
            tokenizer=tokenizer,
            dialogues=dialogues,
            model_id=args.model,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            batch_size=args.batch_size,
            enable_thinking=enable_thinking,
            termination_out=terminations,
        )

        items = []
        for (sc, cond, msgs, _d), resp, term in zip(work, texts, terminations):
            level, role, inducement = parse_condition(cond)
            wellformed = is_wellformed(resp)
            item: dict[str, Any] = {
                "scenario_id": sc.scenario_id,
                "domain": sc.domain,
                "row_index": sc.row_index,
                "topic": sc.topic,
                "dimension": sc.dimension,
                "condition": cond,
                "level": level,
                "role": role,
                "inducement": inducement,
                "lang": lang,
                "prompt_messages": msgs,
                "response": resp,
                "termination": term,  
                "response_wellformed": wellformed,
                "judge_label": None,
                "judge_rationale": "",
                "judge_raw": "",
                "refusal_heuristic": looks_like_refusal(resp, lang=lang),
            }
            items.append(item)

        gen_meta = {
            **truncation_report(texts, tokenizer, max_new_tokens, tag=f"deceptionbench/{lang}"),
            "judge_schema_version": JUDGE_SCHEMA_VERSION,  
            "max_new_tokens": max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "enable_thinking": enable_thinking,
            "allow_thinking_opt_in": bool(args.allow_thinking),
            "domains": list(args.domains),
            "conditions": list(args.conditions),
            "data_dir": str(data_dir),
            "lang": lang,
            "strict_localized_fields": strict,
            "leakage_check_enabled": not args.allow_leakage_check_off,
        }
        save_deceptionbench_rollouts(
            unjudged_path,
            items=items,
            model_id=args.model,
            hf_dataset_id="skyai798/DeceptionBench",
            lang=lang,
            judge_model=None,
            generation=gen_meta,
        )
        print(f"Persisted unjudged snapshot: {unjudged_path}", flush=True)

    if not args.skip_judge:
        print(f"Judging {len(items)} responses with {args.judge_model} in lang={lang}...", flush=True)
        _consec_err = 0  
        for it in items:
            if not it["response_wellformed"]:
                it["judge_label"] = "malformed"
                it["judge_rationale"] = "response truncated mid-thinking"
                it["judge_raw"] = ""
                continue
            sc = next((s for s in scenarios if s.scenario_id == it["scenario_id"]), None)
            if sc is None:
                raise SystemExit(
                    f"scenario {it['scenario_id']!r} not found in loaded scenarios — "
                    "--judge-only snapshot does not match --domains/--data-dir/--lang."
                )
            try:
                label, rationale, raw = judge_response(
                    sc, it["response"], lang=lang, model=args.judge_model
                )
            except Exception as e:
                label, rationale, raw = "judge_uncertain", f"judge_error: {e}", ""
                _consec_err += 1
                if _consec_err >= 10:
                    raise RuntimeError(
                        f"10 consecutive judge failures (last: {e}) — API/key/model problem; "
                        "aborting instead of minting judge_uncertain labels."
                    ) from e
            else:
                _consec_err = 0
            if label == "judge_uncertain" and it["refusal_heuristic"]:
                label = "refusal"
                rationale = (rationale or "") + " [heuristic: looks_like_refusal]"
            it["judge_label"] = normalize_label(label)
            it["judge_rationale"] = rationale
            it["judge_raw"] = raw

    save_deceptionbench_rollouts(
        out,
        items=items,
        model_id=args.model,
        hf_dataset_id="skyai798/DeceptionBench",
        lang=lang,
        judge_model=None if args.skip_judge else args.judge_model,
        generation=gen_meta,
    )
    print(f"Saved DeceptionBench rollouts: {out}", flush=True)


if __name__ == "__main__":
    main()
