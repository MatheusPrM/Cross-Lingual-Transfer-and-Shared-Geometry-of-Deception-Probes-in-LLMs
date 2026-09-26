"""
Model loading helpers. Optional cache and quantization.
"""

import os
from pathlib import Path
from typing import Any, Optional

from project.models import get_model_info


def _resolve_device_map() -> Any:
    raw = os.environ.get("PROJECT_DEVICE_MAP", "").strip()
    if raw:
        if raw == "auto":
            return "auto"
        if raw.isdigit():
            return int(raw)
        return raw
    try:
        import accelerate  # noqa: F401
    except ImportError:
        return None
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_cache_dir(cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Returns cache dir (arg → project env → HF env → default)."""
    if cache_dir is not None:
        return Path(cache_dir)

    val = os.environ.get("PROJECT_MODEL_CACHE")
    if val:
        return Path(val)

    for env in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        val = os.environ.get(env)
        if val:
            return Path(val)

    return None


def get_load_kwargs(
    model_id: str,
    cache_dir: Optional[Path] = None,
    use_4bit_for_large: Optional[bool] = None,
    large_threshold_b: int = 30,
) -> dict[str, Any]:
    """Build kwargs for HF model loading."""
    info = get_model_info(model_id)
    if info is None:
        raise ValueError(f"Unknown model_id: {model_id}. Add it to MODEL_REGISTRY.")

    device_map = _resolve_device_map()

    if not info.revision:
        raise ValueError(f"{model_id}: registry entry has no pinned revision")
    out: dict[str, Any] = {
        "hf_id": info.hf_id,
        "trust_remote_code": info.trust_remote_code,
        "device_map": device_map,
        "dtype": "auto",
        "revision": info.revision,
    }

    cache = get_cache_dir(cache_dir)
    if cache is not None:
        out["cache_dir"] = cache

    if use_4bit_for_large is None:
        use_4bit_for_large = os.environ.get("PROJECT_USE_4BIT", "1").strip().lower() in (
            "1", "true", "yes"
        )

    if use_4bit_for_large and info.size_b >= large_threshold_b:
        try:
            from transformers import BitsAndBytesConfig
            out["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            print(f"[model_loader] bitsandbytes unavailable — loading UNquantized", flush=True)

    return out


def load_model_and_tokenizer(
    model_id: str,
    cache_dir: Optional[Path] = None,
    use_4bit_for_large: Optional[bool] = None,
    omit_model: bool = False,
) -> tuple[Any, Any]:
    """Loads model and tokenizer by registry id."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = get_load_kwargs(
        model_id,
        cache_dir=cache_dir,
        use_4bit_for_large=use_4bit_for_large,
    )
    hf_id = kwargs.pop("hf_id")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=kwargs.get("cache_dir"),
        revision=kwargs.get("revision"),
        trust_remote_code=kwargs.get("trust_remote_code", True),
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    if omit_model:
        return None, tokenizer

    import gc

    import torch

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)

    if next(model.parameters()).device.type == "meta":
        retry_kw = {**kwargs, "low_cpu_mem_usage": False}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(hf_id, **retry_kw)

    if next(model.parameters()).device.type == "meta":
        raise RuntimeError(
            "Model weights are still on torch.device('meta') after load. "
            "Generation requires real weights on CPU/GPU. "
            "Try PROJECT_DEVICE_MAP=0 for a single GPU, or PROJECT_DEVICE_MAP=auto if you need "
            "CPU/disk offload for a large checkpoint."
        )

    if kwargs.get("device_map") is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        device_str = str(next(model.parameters()).device)
        if device == "cuda":
            print(f"Model loaded on GPU ({device_str})", flush=True)
        else:
            print(f"Model loaded on CPU ({device_str})", flush=True)
    else:
        device_str = str(next(model.parameters()).device)
        print(f"Model loaded on {device_str}", flush=True)

    model.eval()
    return model, tokenizer

def model_precision_info(model: Any) -> dict[str, Any]:
    import torch as _torch

    try:
        param_dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        param_dtype = "unknown"
    quantized = bool(getattr(model, "is_loaded_in_4bit", False)
                     or getattr(model, "is_loaded_in_8bit", False))
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    return {
        "param_dtype": param_dtype,
        "quantized_4bit": quantized or (qcfg is not None),
        "project_use_4bit_env": os.environ.get("PROJECT_USE_4BIT", "1"),
        "project_4bit_threshold_env": os.environ.get("PROJECT_4BIT_THRESHOLD", ""),
        "cuda_available": bool(_torch.cuda.is_available()),
        "device_name": (_torch.cuda.get_device_name(0)
                        if _torch.cuda.is_available() else "cpu"),
    }


def resolved_model_revision(model_id: str) -> Optional[str]:
    """The pinned HF revision this model_id resolves to, for artifact stamping."""
    info = get_model_info(model_id)
    if info is not None and getattr(info, "revision", None):
        return info.revision
    return None
