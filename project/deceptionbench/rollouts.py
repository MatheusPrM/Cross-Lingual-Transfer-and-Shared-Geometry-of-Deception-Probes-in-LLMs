"""DeceptionBench rollout JSON format (one row per scenario x condition, response-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Any, Optional


DECEPTIONBENCH_ROLLOUT_FAMILY = "deceptionbench"
DECEPTIONBENCH_SCHEMA_VERSION = 1


def save_deceptionbench_rollouts(
    path: Path,
    *,
    items: list[dict[str, Any]],
    model_id: str,
    hf_dataset_id: str,
    lang: str,
    judge_model: str | None,
    generation: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_family": DECEPTIONBENCH_ROLLOUT_FAMILY,
        "deceptionbench_schema_version": DECEPTIONBENCH_SCHEMA_VERSION,
        "hf_dataset_id": hf_dataset_id,
        "lang": lang,
        "model_id": model_id,
        "judge_model": judge_model,
        "generation": generation,
        "rollouts": items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_deceptionbench_rollouts(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"DeceptionBench rollouts must be a JSON object: {path}")
    if data.get("dataset_family") != DECEPTIONBENCH_ROLLOUT_FAMILY:
        raise ValueError(
            f"Not a DeceptionBench rollout (expected dataset_family={DECEPTIONBENCH_ROLLOUT_FAMILY!r}): {path}"
        )
    items = data.get("rollouts", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid rollouts list in {path}")
    meta = {k: v for k, v in data.items() if k != "rollouts"}
    return [x for x in items if isinstance(x, dict)], meta


def truncation_report(texts, tokenizer, max_new_tokens, *, tag=""):
    """Budget-exhaustion check: responses tokenizing to ~max_new_tokens were likely cut off mid-generation."""
    n_trunc = 0
    for s in texts:
        try:
            n_tok = len(tokenizer(s or "", add_special_tokens=False)["input_ids"])
        except Exception:
            continue
        if n_tok >= int(max_new_tokens) - 16:
            n_trunc += 1
    rate = n_trunc / max(len(texts), 1)
    if rate > 0.1:
        print(
            f"WARNING{f' [{tag}]' if tag else ''}: {n_trunc}/{len(texts)} generations ({rate:.0%}) hit the "
            f"{max_new_tokens}-token budget — refusal/label rates may be truncation-contaminated.",
            flush=True,
        )
    return {"n_truncated": n_trunc, "truncation_rate": round(rate, 4)}


def generate_completions_for_dialogues(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dialogues: list[Dialogue],
    model_id: str,
    max_new_tokens: int = 180,
    temperature: float = 0.0,
    do_sample: bool = False,
    batch_size: int = 8,
    enable_thinking: Optional[bool] = None,
    assistant_prefill: Optional[str] = None,
    termination_out: Optional[list] = None,
) -> list[str]:
    """Batch-generate one completion string per dialogue; assistant_prefill forces opening tokens and is prepended to the stored text.
    """
    import torch
    from project.chat_format import format_dialogue_for_chat
    prefill = assistant_prefill or ""  
    tokenizer.padding_side = "left"  
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
    device = next(model.parameters()).device
    outputs: list[str] = [""] * len(dialogues)
    if termination_out is not None:
        termination_out.clear()
        termination_out.extend([""] * len(dialogues))
    step = max(1, batch_size)
    for start in range(0, len(dialogues), step):
        end = min(start + step, len(dialogues))
        batch = dialogues[start:end]
        formatted = [
            format_dialogue_for_chat(
                d,
                tokenizer,
                model_id=model_id,
                add_generation_prompt=True,
                for_generation=True,
                enable_thinking=enable_thinking,
            )
            for d in batch
        ]
        if prefill:
            formatted = [f + prefill for f in formatted]  
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True,
            add_special_tokens=False,  
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            gen_out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        prompt_len = input_ids.shape[1]
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            eos = tokenizer.eos_token_id
            eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
        gen_cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if gen_cfg_eos is not None:
            eos_ids |= set(gen_cfg_eos) if isinstance(gen_cfg_eos, (list, tuple)) else {int(gen_cfg_eos)}
        for j in range(input_ids.shape[0]):
            new_ids = gen_out[j, prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            outputs[start + j] = (prefill + text) if prefill else text  
            if termination_out is not None:
                produced = new_ids.tolist()
                hit_eos = any(t in eos_ids for t in produced)
                termination_out[start + j] = "eos" if (hit_eos or len(produced) < max_new_tokens) else "cap"
    return outputs