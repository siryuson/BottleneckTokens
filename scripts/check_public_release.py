#!/usr/bin/env python
"""Check the working tree for content that must not enter a public release."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

TEXT_DENYLIST = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"code\.byted",
        r"\bbytedance\b",
        r"\bbyted\b",
        r"/mnt(?:/|$)",
        r"/opt/tiger",
        r"\bsunsiyu\b",
        r"\barnold\b",
        r"\bhdfs\b",
        r"\btos\b",
        r"\bceph\b",
        r"\bfeishu\b",
        r"\blark\b",
        r"\bwecom\b",
        r"\bwebhook\b",
        r"\byour-org\b",
        r"\bYOUR_ORG\b",
        r"docs/internal",
        r"archive/",
        r"api[_-]?key\s*[:=]",
        r"\bpassword\s*[:=]",
    ]
]

FORBIDDEN_NAMES = {
    ".env",
    "checkpoint",
    "checkpoints",
    "logs",
    "wandb",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".pyc",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "data",
    "datasets",
    "outputs",
    "runs",
    "tmp",
    "openspec",
    ".codex",
}

SKIP_FILES = {"check_public_release.py"}


def git_candidate_files(root: Path) -> list[Path] | None:
    """Return tracked and untracked-not-ignored files when inside a git tree."""
    command = [
        "git",
        "-C",
        str(root),
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return [root / line for line in completed.stdout.splitlines() if line]


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir() and path.name in SKIP_DIRS:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        yield path


def check(root: Path) -> list[str]:
    findings: list[str] = []
    candidate_files = git_candidate_files(root)
    paths = candidate_files if candidate_files is not None else list(iter_files(root))
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(root)
        if path.name in SKIP_FILES:
            continue
        if path.name in FORBIDDEN_NAMES:
            findings.append(f"forbidden name: {rel}")
            continue
        if path.is_dir():
            continue
        if path.suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden binary/cache suffix: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in TEXT_DENYLIST:
                if pattern.search(line):
                    findings.append(f"{rel}:{line_no}: {pattern.pattern}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    findings = check(args.root.resolve())
    if findings:
        print("\n".join(findings))
        return 1
    print("public release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
