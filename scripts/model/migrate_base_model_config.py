"""Migrate base_model config artifacts to the new reference schema."""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoConfig


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    """Resolve malformed local-like Qwen paths to official repo ids."""
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    basename = os.path.basename(model_name_or_path.rstrip("/"))
    if basename.startswith("Qwen") and "/" not in basename:
        return f"Qwen/{basename}"
    return model_name_or_path


def load_backbone_config(module_config: dict) -> dict:
    """Load serialized backbone config from model_name_or_path."""
    load_kwargs = {}
    for key in ("revision", "subfolder", "token", "use_auth_token", "trust_remote_code"):
        if key in module_config:
            load_kwargs[key] = module_config[key]
    hf_config = AutoConfig.from_pretrained(
        resolve_model_name_or_path(module_config["model_name_or_path"]),
        **load_kwargs,
    )
    return hf_config.to_dict()


def migrate_config(path: Path) -> bool:
    """Inject backbone_config for referenced backbone modules when missing."""
    data = json.loads(path.read_text())
    modules = data.get("modules", {})
    changed = False

    for _name, module_config in modules.items():
        if not isinstance(module_config, dict):
            continue
        if "model_name_or_path" not in module_config:
            continue
        resolved_path = resolve_model_name_or_path(module_config["model_name_or_path"])
        if resolved_path != module_config["model_name_or_path"]:
            module_config["model_name_or_path"] = resolved_path
            changed = True
        if "backbone_config" in module_config:
            continue
        module_config["backbone_config"] = load_backbone_config(module_config)
        changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return changed


def iter_target_configs(paths: list[str], root: str | None) -> list[Path]:
    """Resolve explicit paths or discover base_model config.json under root."""
    if paths:
        result = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                result.extend(sorted(path.glob("*/base_model/config.json")))
            else:
                result.append(path)
        return result
    if root is None:
        raise ValueError("Either --root or --path must be provided.")
    return sorted(Path(root).glob("*/base_model/config.json"))


def main() -> None:
    """Run migration CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="./outputs",
        help="Root directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Explicit config.json file or experiment directory to migrate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which files need migration.",
    )
    args = parser.parse_args()

    targets = iter_target_configs(args.path, args.root)
    changed_paths: list[Path] = []
    for path in targets:
        data = json.loads(path.read_text())
        modules = data.get("modules", {})
        needs_change = any(
            isinstance(module_config, dict)
            and (
                (
                    "model_name_or_path" in module_config
                    and "backbone_config" not in module_config
                )
                or (
                    "model_name_or_path" in module_config
                    and resolve_model_name_or_path(module_config["model_name_or_path"])
                    != module_config["model_name_or_path"]
                )
            )
            for module_config in modules.values()
        )
        if not needs_change:
            continue
        changed_paths.append(path)
        if not args.dry_run:
            migrate_config(path)

    print(f"targets={len(targets)} changed={len(changed_paths)} dry_run={args.dry_run}")
    for path in changed_paths:
        print(path)


if __name__ == "__main__":
    main()
