"""
Utility module for handling FP8 block-quantized models (compressed-tensors format).

Provides detection, loading, and FP8-safe perturbation helpers so the main
gaussian_profiler.py can apply Gaussian noise to quantized weights directly.
The quantization scales (weight_scale) are never modified — perturbation
operates on the FP8 weight values themselves.
"""

import json
import os
import shutil
import tempfile
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

# FP8 constants
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0

_TOKENIZER_ASSET_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]


def copy_tokenizer_assets(source_model_dir: str, dest_model_dir: str) -> None:
    """
    Copy common tokenizer/generation assets into a saved model directory.

    This is useful when saving to a temporary directory, so downstream loaders
    (e.g., vLLM/transformers) can initialize without needing the original path.
    """
    for fname in _TOKENIZER_ASSET_FILES:
        src = os.path.join(source_model_dir, fname)
        if os.path.exists(src):
            try:
                shutil.copy(src, dest_model_dir)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_quantized_model(model_path: str) -> bool:
    """
    Check whether a model directory contains a compressed-tensors quantized
    model by inspecting its config.json.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        qconfig = config.get("quantization_config", {})
        return qconfig.get("quant_method") == "compressed-tensors"
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _sanitize_recursive_scale_zp(obj) -> None:
    """Remove null scale_dtype/zp_dtype (in-place). Does not modify config_groups."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in ("scale_dtype", "zp_dtype") and obj[k] is None:
                del obj[k]
            else:
                _sanitize_recursive_scale_zp(obj[k])
    elif isinstance(obj, list):
        for item in obj:
            _sanitize_recursive_scale_zp(item)


def sanitize_quantization_config_on_disk(model_dir: str) -> None:
    """
    Sanitize config.json on disk for vLLM and other loaders that reject null
    scale_dtype/zp_dtype. Only removes those keys; config_groups stays as dict.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        qc = config_dict.get("quantization_config")
        if isinstance(qc, dict):
            _sanitize_recursive_scale_zp(qc)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    except (json.JSONDecodeError, OSError):
        pass


def _sanitize_quantization_config_for_loading(config_dict: dict) -> None:
    """
    Modify config dict in-place to work around QuantizationConfig validation errors.
    Removes null scale_dtype/zp_dtype (rejected as 'Extra inputs').
    config_groups stays as dict (transformers and vLLM both expect dict).
    """
    qc = config_dict.get("quantization_config")
    if not isinstance(qc, dict):
        return
    _sanitize_recursive_scale_zp(qc)


def load_quantized_model(path: str, device_hint: str = "cuda") -> torch.nn.Module:
    """
    Load an FP8 compressed-tensors model. Sanitizes quantization_config before
    loading to avoid pydantic validation errors in newer compressed-tensors/transformers.
    """
    device = "cuda" if torch.cuda.is_available() and device_hint == "cuda" else "cpu"

    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    _sanitize_quantization_config_for_loading(config_dict)

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    model_type = config_dict.get("model_type", "llama")
    config_class = CONFIG_MAPPING[model_type]
    config = config_class.from_dict(config_dict)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# FP8-safe perturbation helpers
# ---------------------------------------------------------------------------

def generate_fp8_noise(weight: torch.Tensor) -> torch.Tensor:
    """
    Generate Gaussian noise matching the statistics of an FP8 weight tensor.

    Casts the FP8 weight to bfloat16 to compute mean/std, then generates
    noise in bfloat16.  Returns a bfloat16 noise tensor.
    """
    with torch.no_grad():
        w_float = weight.data.to(torch.bfloat16)
        mean_val = w_float.mean().item()
        std_val = w_float.std().item()
        noise = torch.normal(
            mean=mean_val, std=std_val, size=w_float.shape, device=w_float.device
        )
        return noise.to(torch.bfloat16)


def apply_fp8_perturbation(
    param: torch.nn.Parameter, noise: torch.Tensor, perturbation: str
) -> None:
    """
    Apply a perturbation to an FP8 weight parameter in-place.

    1. Cast weight to bfloat16.
    2. Apply the perturbation (add / avg / replace).
    3. Clamp to the representable FP8 range [-448, 448].
    4. Cast back to float8_e4m3fn and write in-place.

    The associated weight_scale is never touched.
    """
    with torch.no_grad():
        w = param.data.to(torch.bfloat16)

        if perturbation == "add":
            w.add_(noise)
        elif perturbation == "avg":
            w.add_(noise).div_(2.0)
        elif perturbation == "replace":
            w = noise
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")

        w.clamp_(-FP8_MAX, FP8_MAX)
        param.data.copy_(w.to(FP8_DTYPE))


# ---------------------------------------------------------------------------
# 4-bit quantization helpers (BitsAndBytes)
# ---------------------------------------------------------------------------

def quantize_model_dir_to_4bit_bnb(
    input_model_dir: str,
    tmp_root: str,
    *,
    source_assets_dir: Optional[str] = None,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
    device_map: str = "auto",
) -> str:
    """
    Create a temporary 4-bit (BitsAndBytes) model directory from an existing
    full-precision model directory.

    Returns:
        Path to a newly created directory under tmp_root containing a 4-bit model.

    Notes:
        - This relies on transformers+bitsandbytes integration.
        - The output format is the standard HF directory with a quantization_config
          embedded such that `from_pretrained()` can restore 4-bit weights.
    """
    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            f"BitsAndBytesConfig is unavailable; cannot quantize to 4-bit. Error: {e}"
        )

    quant_type = (quant_type or "").lower()
    if quant_type not in ("nf4", "fp4"):
        raise ValueError(f"Unsupported 4-bit quant_type: {quant_type} (expected nf4 or fp4)")

    compute_dtype = (compute_dtype or "").lower()
    if compute_dtype in ("bf16", "bfloat16"):
        compute_torch_dtype = torch.bfloat16
    elif compute_dtype in ("fp16", "float16"):
        compute_torch_dtype = torch.float16
    else:
        raise ValueError(
            f"Unsupported compute_dtype: {compute_dtype} (expected bfloat16 or float16)"
        )

    os.makedirs(tmp_root, exist_ok=True)
    out_dir = tempfile.mkdtemp(dir=tmp_root)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=bool(double_quant),
        bnb_4bit_compute_dtype=compute_torch_dtype,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            input_model_dir,
            quantization_config=bnb_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
    except Exception as e:
        # Fallback when accelerate/device_map isn't available in the environment.
        model = AutoModelForCausalLM.from_pretrained(
            input_model_dir,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        if torch.cuda.is_available():
            model.to("cuda")
        else:
            raise RuntimeError(
                f"4-bit quantization requires CUDA; failed to load with device_map and no GPU available. "
                f"Original error: {e}"
            )
    model.eval()

    try:
        model.save_pretrained(out_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(out_dir)

    assets_dir = source_assets_dir or input_model_dir
    copy_tokenizer_assets(assets_dir, out_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir
