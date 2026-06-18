# Training Conversion Tools

This subdirectory contains retained training-data conversion tools.

Scope:

- Convert raw parquet / jsonl / frame directories into Lance-compatible storage.
- Preserve data integrity and fail early on missing assets or partial conversion.
- Do not implement MMEB evaluation `strict / canonical` logic here.

Tools:

- `convert_mmeb_train_to_lance.py`
- `convert_vidore_train_to_lance.py`
- `convert_visrag_train_to_lance.py`
- `convert_llavahound_train_to_lance.py`
- `convert_parquet_to_lance.py`
- `convert_jsonl_to_lance.py`
- `convert_video_frames_to_lance.py`

Current rewrite boundary:

- Operator-facing scripts in this directory only parse arguments, validate
  paths, print summaries, and call library conversion functions.
- Raw-preserving MMEB conversion lives in
  `src/vlm2emb/data/conversion/mmeb_train.py`.
- Raw-preserving ViDoRe conversion lives in
  `src/vlm2emb/data/conversion/vidore_train.py`.
- Raw-preserving VisRAG conversion lives in
  `src/vlm2emb/data/conversion/visrag_train.py`.
- Raw-preserving LLaVA-Hound conversion lives in
  `src/vlm2emb/data/conversion/llavahound_train.py`.
- Basic training conversion outputs do not generate `manifest.json`,
  `README.md`, `dataset_infos.json`, or split inventory files.

Example:

```bash
export VLM2EMB_OUTPUT_ROOT=./data
export LLAVAHOUND_SOURCE_ROOT=/path/to/ShareGPTVideo_train_video_and_instruction

python scripts/convert/train/convert_vidore_train_to_lance.py \
  --input-dir ./data/raw/vidore/colpali_train_set \
  --output-dir "${VLM2EMB_OUTPUT_ROOT}/vidore_colpali_train_set" \
  --subset-output-base "${VLM2EMB_OUTPUT_ROOT}" \
  --json
```

```bash
python scripts/convert/train/convert_visrag_train_to_lance.py \
  --input-dir ./data/raw/VisRAG-Ret-Train-In-domain-data \
  --output-dir "${VLM2EMB_OUTPUT_ROOT}/openbmb_VisRAG-Ret-Train-In-domain-data" \
  --subset-output-base "${VLM2EMB_OUTPUT_ROOT}" \
  --json
```

```bash
python scripts/convert/train/convert_llavahound_train_to_lance.py \
  --input-dir "${LLAVAHOUND_SOURCE_ROOT}" \
  --output-dir "${VLM2EMB_OUTPUT_ROOT}/ShareGPTVideo_train_video_and_instruction" \
  --json
```
