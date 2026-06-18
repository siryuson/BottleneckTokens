# Runtime Transform Audit

This document records the audit results for the current training/evaluation runtime transform logic. Two sample views are checked:

- raw storage samples: the original text/media fields from Lance/Parquet
- dataset output samples: the runtime samples returned by `Dataset.__getitem__()` or by `build_mmeb_eval_dataset(...)/EvalLanceDataset(transform=...)`

Audit goals:

- determine which runtime transforms are necessary input normalization
- determine which runtime transforms have become duplicate text formatting on top of already-formatted storage

## Summary

### Training families

Training-side runtime transform is still necessary because raw storage is not always the final model-input format:

| Family | Verdict | Why |
|------|------|------|
| MMEB-train | keep | raw query still uses `<|image_1|>` and runtime normalizes it to `<|image_pad|>` |
| ViDoRe | keep | raw query is plain text and runtime injects the image token |
| VisRAG | keep | raw query lacks the source-specific prompt, and the positive side lacks the target text/token |
| LLaVA-Hound | keep | raw instructions are still conversation / `<video>` style and runtime converts them into retrieval layouts |

### MMEB eval

The `MMEB benchmark` should no longer depend on a shared, unconditional eval-time text transform. Both `strict` and `canonical` should use dataset-owned transforms, invoked only when the specific family explicitly declares them, while `canonical` must still preserve the final legacy canonical behavior.

Reason:

- conversion should not bake final query/candidate text by default
- `strict` now follows the "raw content + reconstruction metadata + runtime reconstruction" model
- the visual-token / layout / whitespace / family-template rules historically approved in `canonical` remain valid; Phase 10 is about moving them to the proper layer, not deleting them
- the image side of `MMEB-V2 eval` is still a composite benchmark materialization, where media and annotations may come from MMEB-V2 and MMEB-eval respectively; that source assembly is not the same thing as runtime text transforms

The clearest historical failure was `MomentSeeker`:

- the raw image-conditioned query is `<|image_pad|> ...`
- the runtime query transform injects `<|video_pad|>` again because the dataset belongs to the `video_*` family
- the final runtime query becomes `<|video_pad|>\n<|image_pad|> ...`
- this leaves a stray `video token` inside the batch and triggers Qwen2-VL `video tokens != video features`

Current effective contract after the earlier closure plus completed Phase 10 work:
- the shared layer only keeps validate, materialization, and common utilities
- default runtime behavior now lives in dataset files instead of a shared family-level formatting protocol
- any token/media mismatch must fail-fast
- no-media query cleanup only remains as legacy compatibility handling, not a runtime downgrade policy
- `strict` image / video / visdoc already reconstruct parser-visible text through dataset-owned runtime hooks
- the remaining canonical work should be framed as restoring the full legacy canonical repair set; the only newly introduced rule is that trailing newlines are no longer forced globally, and newline retention must be audited per dataset

Some MMEB eval examples below still show historical storage layouts. They remain here to explain why default runtime semantics had to move back into dataset files; they should not be read as evidence that legacy canonical behavior was discarded. They remain audit references when restoring canonical end behavior.

## Scope

### Training

- `MmebTrainDataset`
- `VidoreTrainDataset`
- `VisragTrainDataset`
- `LlavaHoundTrainDataset`

### Evaluation

- `MMEBBenchmark`
- `RetrievalEvalDataset`
- representative subsets from `MMEB-V2`:
  - `video-tasks/MSR-VTT`
  - `video-tasks/MomentSeeker`
  - `image-tasks/MSCOCO_i2t`
  - `visdoc-tasks/VisRAG_ChartQA`

## Training family samples

### MMEB-train

raw sample from `MMEB-train/data/ImageNet_1K/train.lance`:

```text
qry      = "<|image_1|>\nRepresent the given image for classification\n"
pos_text = "plane, carpenter's plane, woodworking plane"
```

dataset output:

```text
query.text      = "<|image_pad|>\nRepresent the given image for classification\n"
query.images    = 1
positive.text   = "plane, carpenter's plane, woodworking plane"
positive.images = 0
```

Verdict:

- training runtime only normalizes the legacy `<|image_1|>` token into `<|image_pad|>`
- this is necessary normalization rather than duplicate text formatting

Relevant code:
`src/vlm2emb/data/datasets/mmeb_train.py` (around line 83)

### ViDoRe

raw sample from `vidore_colpali_train_set/data/train.lance`:

```text
query  = "Comparing panels a, b, c, and d, which statement best describes the data variance?"
answer = "D"
```

dataset output:

```text
query.text      = "<|image_pad|> Comparing panels a, b, c, and d, which statement best describes the data variance?"
query.images    = 1
positive.text   = "D"
positive.images = 0
```

Verdict:

- raw storage does not contain the image token
- runtime is responsible for converting a single-image query into model-ready text

Relevant code:
`src/vlm2emb/data/datasets/vidore.py` (around line 89)

### VisRAG

raw sample from `openbmb_VisRAG-Ret-Train-In-domain-data/data/train.lance`:

```text
query  = "which are the countries with Chinese speaking population other than China according to this infographic?"
source = "InfoVQA"
```

dataset output:

```text
query.text =
"This query is related to retrieving an infographic ..."

positive.text =
"An infographic with structured data, charts, and annotations. <|image_pad|>"
```

Verdict:

- source-specific prompt injection and positive-side image token injection happen at runtime
- raw storage is not yet in final model-input form, so this runtime layer is still justified

Relevant code:
`src/vlm2emb/data/datasets/visrag.py` (around line 86)

### LLaVA-Hound

raw sample from `ShareGPTVideo_train_video_and_instruction/data/video_instruction/train/sft/video_caption_300k.lance`:

```text
human.value = "<video>\nWrite an in-depth depiction of the video, covering all its aspects."
gpt.value   = "The video opens with a view of an indoor court ..."
```

dataset output:

`caption_retrieval` mode:

```text
query.text    = "<|video_pad|>\nWrite an in-depth depiction of the video, covering all its aspects.\n"
query.images  = 8
positive.text = "The video opens with a view of an indoor court ...\n"
```

`video_retrieval` mode:

```text
query.text      = "Find a video that contains the following visual content: The video opens ...\n"
query.images    = 0
positive.text   = "Understand the content of the provided video: <|video_pad|>\n"
positive.images = 8
```

Verdict:

- raw instructions are still conversation / `<video>` style
- runtime is responsible for mapping the conversation into retrieval query/positive layouts

Relevant code:
`src/vlm2emb/data/datasets/llavahound.py` (around line 105)

## MMEB eval samples

### MSR-VTT (historical storage example)

raw Lance:

```text
queries[0].text      = "Find a video that contains the following visual content: ..."
queries[0].images    = 0
candidates[0].text   = "<|video_pad|>\nUnderstand the content of the provided video.\n"
candidates[0].images = 8
```

Verdict:

- historical storage directly baked the final query/candidate text
- this explains why shared benchmark runtime transforms created duplicate text formatting
- Phase 10 does not preserve this storage shape as the desired end state; instead it moves this text responsibility into dataset-owned transforms while preserving the legacy canonical end behavior

### MSCOCO_i2t (historical storage example)

raw Lance:

```text
queries[0].text      = "<|image_pad|>\nFind an image caption describing the given everyday image.\n"
queries[0].images    = 1
candidates[0].text   = "2 Zebras standing next to each other in plaines."
candidates[0].images = 0
```

Verdict:

- historical storage already included the image token
- this shows why image-family runtime text transforms must be declared per dataset family rather than injected blindly at the benchmark layer

### VisRAG_ChartQA (historical storage example)

raw Lance:

```text
queries[0].text      = "Find a document image that matches the given query: ..."
queries[0].images    = 0
candidates[0].text   = "<|image_pad|>\nUnderstand the content of the provided document image.\n"
candidates[0].images = 1
```

Verdict:

- historical storage already baked the query prompt and candidate token/prompt
- future canonical should no longer treat this as the conversion default
- whether these prompts/tokens are injected must be decided explicitly by the family/profile runtime contract

### MomentSeeker

`queries.lance` modality distribution:

| Type | Count |
|------|------:|
| `video` query | 400 |
| `image` query | 400 |
| `text` query | 802 |
| `mixed token` query | 0 |

raw Lance examples:

```text
video query:
  "<|video_pad|>\nFind the clip that corresponds to the given sentence and video segment: ..."
  n_images = 8

image query:
  "<|image_pad|>\nSelect the video clip that aligns with the given text and image: ..."
  n_images = 1

text query:
  "Find the clip that corresponds to the given text: ..."
  n_images = 0
```

After `build_mmeb_eval_dataset(..., transform_kwargs={\"runtime_mode\": ...})`:

```text
image query:
  "<|video_pad|>\n<|image_pad|>\nSelect the video clip that aligns with the given text and image: ..."
```

Verdict:

- raw Lance is fully correct and contains no mixed token samples
- the mixed token state is introduced by the benchmark runtime transform
- this is the direct cause of the `video tokens != video features` failure

## Current eval-time transform inventory

Current eval-side text runtime behavior uses transform-named helpers; the old `render.py` compatibility re-export has been removed:

1. `transform_eval_sample_with_media_token()`
   - injects modality tokens based on the presence of `images`
   - location: `src/vlm2emb/data/datasets/transforms.py`
   - the old `render_eval_sample_with_media_token()` compatibility path has been removed

2. `transform_eval_sample_with_text_prefix()`
   - injects query prefixes
   - location: `src/vlm2emb/data/datasets/transforms.py`
   - the old `render_eval_sample_with_text_prefix()` compatibility path has been removed

3. `MMEBBenchmark` + `build_mmeb_eval_dataset()`
   - the benchmark only forwards subset, artifact path, and `transform_kwargs` into the MMEB-V2 runtime builder
   - parser modules own subset-specific text transforms through `build_query_transform()` / `build_candidate_transform()`
   - locations: `src/vlm2emb/evaluation/benchmarks/mmeb.py` and `src/vlm2emb/data/datasets/mmeb_v2/loader.py`

## Recommendation

For `MMEB benchmark`:

- disable eval-time text transforms
- use raw Lance text directly as query/candidate text
- keep benchmark-materialization data-fact repairs and composite-source assembly where they are still required
- keep non-text runtime logic:
  - image decoding
  - video frame sampling
  - `_index` tracking

In other words:

- disable `query_transform/candidate_transform` when they mutate text, inject tokens, or inject prefixes
- do not keep “force one trailing newline” or “force-remove trailing newline” as a default eval-conversion responsibility
- keep media decoding and frame sampling

Training-side runtime transforms should not be removed together with MMEB eval, because training raw storage is still not uniformly in final input format.
