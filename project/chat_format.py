"""
Model-specific chat formatting.
"""

from __future__ import annotations

from typing import Optional

from project.types import Dialogue, Message

_EOT_SUFFIX_BY_FAMILY: dict[str, str] = {
    "llama": "<|eot_id|>",
    "gemma": "<end_of_turn>\n",
    "mistral": "</s>",
    "phi": "<|end|>",
    "cohere": "<|END_OF_TURN_TOKEN|>",
}

_FOLD_SYSTEM_FAMILIES: frozenset[str] = frozenset({"gemma", "mistral"})


def get_template_family(tokenizer: object, model_id: Optional[str] = None) -> Optional[str]:
    """Infers the chat-template family."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template:
        if "<|turn>" in template or "<turn|>" in template:
            return "gemma4"
        if "<start_of_turn>" in template or "<end_of_turn>" in template:
            return "gemma"
        if "<|header_start|>" in template or "<|header_end|>" in template or "<|eot|>" in template:
            return "llama4"
        if "<|start_header_id|>" in template or "<|end_header_id|>" in template or "<|eot_id|>" in template:
            return "llama"
        if "<|START_OF_TURN_TOKEN|>" in template or "<|CHATBOT_TOKEN|>" in template:
            return "cohere"  

    if model_id:
        try:
            from project.models import get_model_info

            info = get_model_info(model_id)
            if info is not None:
                if info.family == "gemma" and model_id.startswith("gemma-4"):
                    return "gemma4"
                if info.family == "llama" and model_id.startswith("llama-4"):
                    return "llama4"
                return info.family
        except Exception:
            pass

    name = getattr(tokenizer, "name_or_path", "") or ""
    if not name:
        return None

    name_lower = name.lower()

    if "meta-llama" in name_lower or "llama" in name_lower:
        return "llama"
    if "google/gemma" in name_lower or "gemma" in name_lower:
        return "gemma"
    if "mistralai" in name_lower or "mistral" in name_lower:
        return "mistral"
    if "qwen" in name_lower:
        return "qwen"
    if "aya" in name_lower or "cohere" in name_lower or "command-r" in name_lower:
        return "cohere"
    if "phi" in name_lower or "microsoft" in name_lower:
        return "phi"
    if "deepseek" in name_lower:
        return "deepseek"

    return None


def get_eot_suffix(family: Optional[str]) -> Optional[str]:
    """Returns EOT suffix."""
    if family is None:
        return None
    return _EOT_SUFFIX_BY_FAMILY.get(family)


def _merge_consecutive_roles(dialogue: Dialogue) -> Dialogue:
    """Merges consecutive messages with same role."""
    if not dialogue:
        return dialogue

    out: list[Message] = []
    for m in dialogue:
        if out and out[-1].role == m.role:
            prev = out[-1]
            out[-1] = Message(
                role=prev.role,
                content=(prev.content + " " + m.content).strip(),
                detect=prev.detect or m.detect,
            )
        else:
            out.append(m)

    return out


def preprocess_dialogue_for_template(
    dialogue: Dialogue,
    family: Optional[str] = None,
) -> list[dict[str, str]]:
    """Prepares dialogue for chat template."""
    if not dialogue:
        return []

    fold_system = family in _FOLD_SYSTEM_FAMILIES if family else False
    processed: list[dict[str, str]] = []

    if fold_system and dialogue[0].role == "system":
        processed.append({"role": "user", "content": dialogue[0].content + "\n\n"})
        dialogue = dialogue[1:]

    for m in dialogue:
        if processed and processed[-1]["role"] == m.role:
            processed[-1]["content"] += m.content
        else:
            processed.append({"role": m.role, "content": m.content})

    return processed


def format_dialogue_for_chat(
    dialogue: Dialogue,
    tokenizer: object,
    *,
    model_id: Optional[str] = None,
    add_generation_prompt: bool = False,
    for_generation: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Formats dialogue into chat string."""
    family = get_template_family(tokenizer, model_id)
    messages = preprocess_dialogue_for_template(dialogue, family)

    if not messages:
        return ""

    if (add_generation_prompt and messages[-1].get("role") == "assistant"
            and not (messages[-1].get("content") or "").strip()):
        messages = messages[:-1]
        if not messages:
            return ""

    has_chat_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )

    if has_chat_template:
        template_kwargs: dict[str, object] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None and family == "qwen":
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError as _exc:
            template_kwargs.pop("enable_thinking", None)
            text = tokenizer.apply_chat_template(messages, **template_kwargs)

        if isinstance(text, list):
            text = text[0] if text else ""

        last_role = messages[-1].get("role", "")
        last_content = (messages[-1].get("content") or "").strip()

        if for_generation and last_role == "assistant" and last_content == "":
            eot = get_eot_suffix(family)
            if eot and text.endswith(eot):
                text = text[: -len(eot)]

        return text

    merged = _merge_consecutive_roles(dialogue)
    parts = [f"<|{m.role}|>\n{m.content}" for m in merged]
    return "\n".join(parts)


def set_chat_template_if_needed(tokenizer: object, model_id: Optional[str] = None) -> None:
    """Sets template if tokenizer lacks one."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return

    family = get_template_family(tokenizer, model_id)
    if family is None:
        return

    pass