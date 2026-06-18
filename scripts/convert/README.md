# Conversion Scripts

This directory contains the maintained dataset conversion and validation tools.

## Project-Level Boundary

The project now follows a strict three-layer boundary:

1. **source**
- raw download sources
- raw split / directory layout
- no model-facing prompt or token semantics

2. **conversion**
- aggregate raw sources
- fix data facts: source freeze, path normalization, broken-sample handling, qrels / label / page-index fixes, stable IDs
- write stable Lance storage and metadata

3. **runtime dataset**
- read Lance
- apply prompt/token/newline/layout logic that should remain adjustable without regenerating storage

Rule:
- conversion owns **storage correctness**
- runtime datasets own **input flexibility**

## Why MMEB Evaluation Is Special

MMEB-V2 evaluation data is not a normal “store raw data as-is” dataset:
- it aggregates multiple upstream sources
- it contains historical source drift and packaging problems
- strict / canonical semantics must be frozen explicitly

Therefore the MMEB evaluation converters in this directory are allowed to apply
one-time storage fixes to guarantee stable Lance artifacts.

Even for MMEB:
- prompt wording
- token placement
- newline / whitespace formatting

should stay runtime-adjustable whenever possible.

## Current Structure

- `src/vlm2emb/data/conversion/mmeb_v2/`
  - maintained MMEB-V2 evaluation conversion implementation
  - `datasets.py`: dataset/source/pipeline source-of-truth
  - `image.py`: image-task conversion pipelines
  - `video.py`: video-task conversion pipelines
  - `visdoc.py`: visual-document conversion pipelines
  - `common.py` / `bindings.py`: shared storage and source-resolution helpers
- `src/vlm2emb/cli/convert_mmeb_v2.py`
  - maintained user-facing MMEB-V2 conversion CLI
- `scripts/convert/eval/mmeb_v2.py`
  - thin shell that forwards to the maintained CLI entrypoint
- `scripts/convert/`
  - validation and audit helpers
- `scripts/convert/train/`
  - training dataset storage conversion tools

## Rules

- Shared source/path/schema/metadata logic should live in `src/vlm2emb/data/conversion/mmeb_v2/common.py` and `bindings.py`.
- Dataset/source/pipeline declarations should live in `src/vlm2emb/data/conversion/mmeb_v2/datasets.py`.
- Task converters should focus on source parsing, data-fact fixes, and record construction only.
- Avoid baking runtime-only formatting policy into conversion unless MMEB storage stability requires it.
- Evaluation-side `strict / canonical` logic belongs only to the top-level MMEB converters.
- Training-side converters under `train/` are storage conversion tools. They do not own MMEB evaluation semantics.
- Historical machine-specific shell wrappers are intentionally removed. Prefer the Python tools here.
