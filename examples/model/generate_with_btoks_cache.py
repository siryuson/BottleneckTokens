"""Generate text from a BToks bottleneck-token KV cache.

Example:
    python examples/model/generate_with_btoks_cache.py \
        --image-path docs/assets/example.jpg \
        --base-model-path Qwen/Qwen2-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from omegaconf import OmegaConf

from vlm2emb.config import load_config
from vlm2emb.data.processors import batch_processor_outputs, create_processor_wrapper
from vlm2emb.inference.btoks import generate_from_btoks_cache
from vlm2emb.model import VLM2Emb, create_model_and_processor
from vlm2emb.modules.btoks import extract_btoks_kv

DEFAULT_CONFIG_PATH = "configs/presets/btoks_qwen2vl_2b_v2.yaml"
DEFAULT_LOCAL_IMAGE = "docs/assets/example.jpg"
IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_GENERATION_PROMPT = "<|im_start|>assistant\n"
DEFAULT_GENERATION_SUFFIX = "<|im_end|>"


def _default_image_path() -> str | None:
    """Prefer the bundled example image when available."""
    if Path(DEFAULT_LOCAL_IMAGE).exists():
        return DEFAULT_LOCAL_IMAGE
    return None


def _override_runtime_paths(config: dict[str, Any], base_model_path: str | None) -> dict[str, Any]:
    """Override processor/backbone source paths for local runs."""
    if not base_model_path:
        return config

    if isinstance(config.get("processor"), dict):
        if "processor_name" in config["processor"]:
            config["processor"]["processor_name"] = base_model_path

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        modules = model_cfg.get("modules")
        if isinstance(modules, dict):
            for mod_cfg in modules.values():
                if isinstance(mod_cfg, dict) and "model_name_or_path" in mod_cfg:
                    mod_cfg["model_name_or_path"] = base_model_path
    return config


def _set_attn_implementation(config: Any, attn_implementation: str | None) -> Any:
    """Override backbone attention implementation in runtime configs."""
    if not attn_implementation:
        return config

    if isinstance(config, dict):
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            modules = model_cfg.get("modules")
        else:
            modules = config.get("modules")
    else:
        modules = getattr(config, "modules", None)

    if isinstance(modules, dict):
        for mod_cfg in modules.values():
            if isinstance(mod_cfg, dict) and "attn_implementation" in mod_cfg:
                mod_cfg["attn_implementation"] = attn_implementation

    return config


def _resolve_attn_implementation(
    device: torch.device,
    attn_implementation: str | None,
) -> str | None:
    """Resolve an attention implementation that can run on the target device."""
    if attn_implementation:
        return attn_implementation
    if device.type == "cpu":
        return "eager"
    return None


def _resolve_generation_protocol(
    config: dict[str, Any],
    *,
    generation_prompt_override: str | None,
    generation_suffix_override: str | None,
) -> tuple[str, str]:
    """Resolve generation prompt/suffix from config defaults with CLI overrides."""
    train_cfg = config.get("train")
    args_cfg = train_cfg.get("args") if isinstance(train_cfg, dict) else None

    generation_prompt = (
        generation_prompt_override
        if generation_prompt_override is not None
        else (
            args_cfg.get("generation_prefix_text", DEFAULT_GENERATION_PROMPT)
            if isinstance(args_cfg, dict)
            else DEFAULT_GENERATION_PROMPT
        )
    )
    generation_suffix = (
        generation_suffix_override
        if generation_suffix_override is not None
        else (
            args_cfg.get("generation_suffix_text", DEFAULT_GENERATION_SUFFIX)
            if isinstance(args_cfg, dict)
            else DEFAULT_GENERATION_SUFFIX
        )
    )
    return generation_prompt, generation_suffix


def _load_processor_from_checkpoint(path: str) -> Any:
    """Load processor/tokenizer from a checkpoint path."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _is_vlm2emb_artifact(path: str | None) -> bool:
    """Return true when the path looks like a saved vlm2emb/BToks artifact."""
    if not path:
        return False

    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return False

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("model_type") in {"vlm2emb", "btoks"}


def _load_runtime(
    *,
    config_path: str,
    checkpoint_path: str | None,
    base_model_path: str | None,
    device: torch.device,
    attn_implementation: str | None,
) -> tuple[Any, Any, str | None, dict[str, Any]]:
    """Load a runtime model and processor for the example."""
    config = OmegaConf.to_container(load_config(config_path), resolve=True)
    config = _override_runtime_paths(config, base_model_path)
    resolved_attn_implementation = _resolve_attn_implementation(device, attn_implementation)
    config = _set_attn_implementation(config, resolved_attn_implementation)

    if checkpoint_path:
        from peft import PeftConfig, PeftModel

        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
        except Exception:
            model_config = VLM2Emb.config_class.from_pretrained(checkpoint_path)
            model_config = _set_attn_implementation(
                model_config,
                resolved_attn_implementation,
            )
            model = VLM2Emb.from_pretrained(checkpoint_path, config=model_config)
            processor = _load_processor_from_checkpoint(checkpoint_path)
            return _prepare_model_for_inference(model, device), processor, checkpoint_path, config

        resolved_base_model_path = base_model_path or getattr(
            peft_config,
            "base_model_name_or_path",
            None,
        )
        if _is_vlm2emb_artifact(resolved_base_model_path):
            model_config = VLM2Emb.config_class.from_pretrained(resolved_base_model_path)
            model_config = _set_attn_implementation(
                model_config,
                resolved_attn_implementation,
            )
            model = VLM2Emb.from_pretrained(resolved_base_model_path, config=model_config)
            processor = _load_processor_from_checkpoint(resolved_base_model_path)
        else:
            config = _override_runtime_paths(config, resolved_base_model_path)
            config = _set_attn_implementation(config, resolved_attn_implementation)
            model, processor = create_model_and_processor(config)
        model = PeftModel.from_pretrained(model, checkpoint_path).merge_and_unload()
        try:
            processor = _load_processor_from_checkpoint(checkpoint_path)
        except Exception:
            pass
        return _prepare_model_for_inference(model, device), processor, resolved_base_model_path, config

    model, processor = create_model_and_processor(config)
    return _prepare_model_for_inference(model, device), processor, base_model_path, config


def _prepare_model_for_inference(model: Any, device: torch.device) -> Any:
    """Move model to device and align module dtypes for inference."""
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    return model.eval()


def _prepare_encode_text(text: str, image_path: str | None) -> str:
    """Ensure multimodal prompts include the image token."""
    if not image_path or IMAGE_TOKEN in text:
        return text
    return f"{IMAGE_TOKEN}\n{text}"


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields to the target device."""
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if value is None:
            continue
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--image-path", default=_default_image_path())
    parser.add_argument(
        "--encode-text",
        default="Represent the visual content in detail.",
        help=(
            "Semantic text paired with the encode sample. When --image-path is set, "
            "the processor wrapper will normalize the visual boundary automatically"
        ),
    )
    parser.add_argument(
        "--generation-prompt",
        default=None,
        help="Optional override for generation prefix text; defaults to train.args.generation_prefix_text",
    )
    parser.add_argument(
        "--generation-suffix",
        default=None,
        help="Optional override for generation suffix text; defaults to train.args.generation_suffix_text",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Override backbone attention implementation; defaults to eager on CPU",
    )
    return parser


def main() -> None:
    """Run the bottleneck-cache generation example."""
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)

    model, processor, resolved_base_model_path, runtime_config = _load_runtime(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        base_model_path=args.base_model_path,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    generation_prompt, generation_suffix = _resolve_generation_protocol(
        runtime_config,
        generation_prompt_override=args.generation_prompt,
        generation_suffix_override=args.generation_suffix,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    wrapper = create_processor_wrapper(processor=processor)

    image = None
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")

    encode_text = _prepare_encode_text(args.encode_text, args.image_path)
    encode_samples = wrapper(
        [encode_text],
        [[image]] if image is not None else [None],
    )
    encode_batch = _move_to_device(batch_processor_outputs(encode_samples, wrapper), device)

    prompt_batch = tokenizer(
        [generation_prompt],
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_batch = _move_to_device(prompt_batch, device)

    modules_dict = model._modules_dict
    backbone = modules_dict["backbone"] if "backbone" in modules_dict else None
    if backbone is None or not hasattr(backbone, "model"):
        raise ValueError("Expected a backbone module with an underlying HF model")

    im_end_token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        candidate = tokenizer.convert_tokens_to_ids(generation_suffix)
        if isinstance(candidate, int) and candidate >= 0:
            im_end_token_id = candidate

    with torch.inference_mode():
        encode_out = model.encode(**encode_batch, return_cache=True)
        btoks_kv = extract_btoks_kv(
            past_key_values=encode_out["past_key_values"],
            btoks_token_mask=encode_out["btoks_token_mask"],
        )
        generated_ids = generate_from_btoks_cache(
            encode_out=encode_out,
            backbone_model=backbone.model,
            prompt_batch=prompt_batch,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            stop_token_ids=[im_end_token_id] if im_end_token_id is not None else None,
        )

    completion_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    full_text = tokenizer.batch_decode(
        torch.cat([prompt_batch["input_ids"], generated_ids], dim=1),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("=== BToks Bottleneck Cache Generation ===")
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint_path or '(none)'}")
    print(f"Base model: {resolved_base_model_path or '(from checkpoint/config)'}")
    print(f"Image: {args.image_path or '(text-only)'}")
    print(f"Encode text: {encode_text}")
    print(f"BToks KV length: {btoks_kv.get_seq_length() if btoks_kv is not None else 0}")
    print(f"Generation prompt: {generation_prompt}")
    print(f"Generation suffix: {generation_suffix}")
    print(f"Completion: {completion_text}")
    print(f"Full text: {full_text}")


if __name__ == "__main__":
    main()
