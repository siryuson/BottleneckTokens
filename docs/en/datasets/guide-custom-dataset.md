# Dataset Onboarding Guide

This document explains how to add new training datasets or evaluation datasets to the current BToks codebase.

> Related references:
> - [Format Specification](./format-spec.md)
> - [Data Pipeline](./architecture-data-pipeline.md)

## 1. Decide the Layer First

Before adding a dataset, decide which layer owns the change:

1. **source layer**
- raw download source
- raw splits / raw directory layout
- no model-facing semantics

2. **conversion layer**
- aggregate raw sources
- fix data facts: source freeze, path normalization, broken-sample handling, qrels/label/page-index fixes, stable IDs
- produce Lance storage and metadata

3. **runtime dataset layer**
- read samples from Lance
- apply runtime-adjustable prompt/token/newline/layout logic
- do not change already-frozen data facts

Rule:
- **storage correctness belongs to conversion**
- **input flexibility belongs to runtime**

## 2. Default Onboarding Strategy

### 2.1 Every Dataset Gets Its Own Entry Point

Every new dataset should have its own dataset entry point for:
- isolated maintenance
- isolated configuration
- isolated special cases

### 2.2 Reuse Shared Schemas And Archetype Helpers Underneath

Do not duplicate a full implementation for every dataset. Prefer:

- a dataset-specific registry name and entry class
- shared task-archetype helpers, common schemas, and thin dataset wrappers underneath

Examples:
- training Lance datasets should use `SampleLanceDataset` directly; introduce a training-specific subclass only after it has real training-specific behavior
- retrieval evaluation reuses the `RetrievalEvalDataset` container together with `EvalLanceDataset`
- `VideoEvalLanceDataset` remains only as a compatibility wrapper; new mainline paths such as `MMEB-V2 eval` should prefer `EvalLanceDataset + explicit transform`

For new datasets, prefer the following two-step path:

1. clarify the source / conversion / runtime boundaries
2. produce training `TrainSample` or retrieval `queries/candidates/qrels` through a thin runtime adapter

This keeps future dataset additions in the "thin adapter / parser" shape instead of forcing changes into the trainer/evaluator core.

The current boundary is:
- the shared layer only keeps validate and common utilities
- dataset-specific default runtime behavior lives in dataset files
- the shared layer reduces duplication, but it does not own dataset default text-transform policy anymore

### 2.3 Prefer Task Archetypes As The Shared Boundary

For future dataset onboarding, decide the **task archetype** first instead of starting from a family bucket.

The current preferred archetypes are:

| archetype | typical task_type |
|-----------|-------------------|
| `classification` | `image_classification`, `video_classification` |
| `question_answer` | `image_question_answer`, `video_question_answer`, `document_question_answer` |
| `retrieval` | `image_retrieval`, `video_retrieval` |
| `moment_retrieval` | `video_moment_retrieval` |
| `grounding` | `image_visual_grounding` |
| `document_retrieval` | `document_retrieval`, some `visdoc_ood` cases |
| `entity_retrieval` | `multimodal_retrieval`, `entity_retrieval` |

Recommended decision order:

1. decide the archetype first
2. then decide whether an existing family schema / base class can be reused
3. keep dataset-owned runtime logic inside the dataset module

Rules:

- the archetype owns only the minimal shared semantic structure
- the family owns only schema/layout-level reuse
- the dataset file keeps prompt logic, metadata mapping, special cases, and historical compatibility behavior
- ready datasets should be discovered through explicit dataset registry / config entries; the project no longer keeps a generic manifest catalog compatibility package

## 3. When Conversion Is Required

By default, keep format changes in runtime so prompt/token adjustments do not require regenerating data.

However, the following belong in conversion:

- aggregating multiple raw sources
- repairing media paths or broken samples
- fixing qrels / labels / page-index
- materializing scattered media into stable Lance storage
- for special cases such as `MMEB-V2 eval`, freezing strict / canonical data facts

Outside `MMEB-V2 eval`, evaluation datasets should not adopt dual strict/canonical storage by default. Keep original content and schema whenever possible.

## 4. MMEB-V2 Is a Special Case

MMEB-V2 is not a simple “store raw data as-is” dataset. It currently involves:
- multi-source aggregation
- historical version drift
- path, media, qrels, and page-index issues

Therefore MMEB-V2 is allowed to fix these **data-fact issues** in conversion so that storage remains stable.

Even for MMEB-V2, the following should still remain runtime-adjustable whenever possible:
- prompt wording
- token injection
- newline / whitespace
- model-specific text transforms

## 5. Training Dataset Onboarding

### 5.1 Unified Output Format

Migrated training Datasets output `TrainSample` from `src/vlm2emb/data/schema.py`:

```python
class MediaInput(TypedDict, total=False):
    kind: str
    content: Any
    metadata: dict[str, Any]


class MultiModalInput(TypedDict):
    text: str
    media: list[MediaInput]


class TrainSample(TypedDict, total=False):
    query: MultiModalInput
    positive: MultiModalInput
    negative: MultiModalInput
    metadata: dict[str, Any]
```

Requirements:
- the single encoding object only stores `text` and `media`
- `query / positive / negative` are relation-layer fields
- `metadata` stores passthrough values such as dataset name, subset, and source row

### 5.2 Reuse Existing Lance Schemas First

The training stack currently supports 4 official Lance schemas:

| Family | Registry name | Characteristic |
|--------|---------------|----------------|
| MMEB | `mmeb_train` | raw sample table + image side table |
| ViDoRe | `vidore_train` | single-table with embedded document image |
| VisRAG | `visrag_train` | single-table with document image + source |
| LLaVA-Hound | `llavahound_train` | dual-table: frames + video_instruction |

Only add a new schema when the existing ones are clearly insufficient.

Reusing a schema does not mean reusing one family-wide default runtime semantics block.  
Schema reuse and runtime semantic ownership should be evaluated separately.

### 5.3 Minimal Steps for a Training Loader

1. Add a loader under `src/vlm2emb/data/datasets/`
2. Reuse `TrainSample` from `src/vlm2emb/data/schema.py` instead of redefining the sample schema inside the loader
3. Register it with `@AutoDataset.register(...)`
4. Make `__getitem__()` return `TrainSample`
5. Import it in `src/vlm2emb/data/datasets/__init__.py`
6. Add YAML config pointing to the Lance root

Minimal example:

```python
from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import SampleLanceDataset


def transform_my_train_record(row):
    return {
        "query": {"text": row["query"], "media": []},
        "positive": {"text": row["positive"], "media": []},
        "negative": {"text": "", "media": []},
        "metadata": {"dataset_name": "my_train_dataset"},
    }


@AutoDataset.register("my_train_dataset")
class MyTrainDataset(SampleLanceDataset):
    def __init__(self, lance_path: str):
        super().__init__(
            lance_path,
            read_columns=["query", "positive"],
            transform=transform_my_train_record,
        )
```

Notes:
- New datasets should declare `read_columns`
- `required_columns` remains only as a compatibility fallback for unmigrated legacy families

## 6. Retrieval Evaluation Dataset Onboarding

### 6.1 Standard Storage Layout

The standard storage format for retrieval evaluation datasets is fixed as:

- `queries`
- `candidates`
- `qrels`

That is, the Lance three-file layout:

```text
dataset_root/
├── queries.lance
├── candidates.lance
├── qrels.lance
└── metadata.json
```

### 6.2 Unified Runtime Entry

The unified retrieval-eval container remains:

- `src/vlm2emb/data/datasets/eval.py` -> `RetrievalEvalDataset`

But the concrete builders are now split by dataset responsibility:

- generic Lance reading is handled by `EvalLanceDataset` / `LanceDataset`
- `MMEB-V2 eval` enters the parser-driven runtime through `src/vlm2emb/data/datasets/mmeb_v2/loader.py` -> `build_mmeb_eval_dataset(...)`
- the benchmark only selects subsets, resolves artifact paths, and forwards `runtime_mode`; it no longer owns parser-specific defaults

This stack supports:
- image/video decoding
- video frame sampling
- query/candidate-level runtime transforms

Therefore:
- data facts such as qrels/source/media/id should be fixed in conversion
- prompt/token/newline should remain in runtime transforms whenever possible
- conversion outputs should include deterministic dataset/sample/relation IDs plus a machine-readable manifest
- token/media mismatch must fail-fast instead of falling back silently at runtime

### 6.3 Runtime-Adjustable Formatting

The sample-oriented Lance base now supports runtime transforms through:

- `src/vlm2emb/data/datasets/base.py` -> `SampleLanceDataset`
- `src/vlm2emb/data/datasets/base.py` -> `EvalLanceDataset`
- `src/vlm2emb/data/datasets/mmeb_v2/loader.py` -> `build_mmeb_eval_dataset(...)`

This allows the system to apply, without regenerating Lance:
- token injection
- whitespace/newline normalization
- model-specific text transforms

For `MMEB-V2 eval`, the runtime transform layer also owns:
- explicit query/candidate raw-column contracts per subset
- static dispatch from subset name to parser module
- parser-visible `origin` / `canonical` text assembly, instead of a shared heuristic formatter

Notes:
- a no-media query may still go through legacy token cleanup for backward compatibility, but that is not a runtime downgrade policy
- if a sample declares visual tokens without media, or media without the matching tokens, the runtime must still fail-fast

Engineering constraint:
- any transform attached to a dataset, benchmark, or evaluator that may cross `DataLoader(num_workers>0)` or multiprocessing training/eval workers must be a **pickle-safe callable**
- do not use local `lambda`s or local closures for default runtime transforms
- allowed forms should be limited to:
  - top-level functions
  - top-level `dataclass` callables
  - `functools.partial(top_level_function, ...)`
- during review, check both semantic correctness and whether the transform can be serialized into worker processes

## 7. Training Conversion Tooling

Training-data conversion currently remains under `scripts/convert/train/`.

Official Python tools:

| Script | Purpose |
|--------|---------|
| `convert_mmeb_train_to_lance.py` | MMEB-train -> Lance |
| `convert_parquet_to_lance.py` | parquet -> Lance |
| `convert_jsonl_to_lance.py` | JSONL -> Lance |
| `convert_video_frames_to_lance.py` | frame directories -> Lance |
| `convert_llavahound_train_to_lance.py` | raw-preserving LLaVA-Hound -> Lance |

These tools only own **training-data storage conversion**. They do not own MMEB evaluation strict/canonical semantics.

## 8. Checklist

### 8.1 General

- [ ] Did you decide the source / conversion / runtime ownership first?
- [ ] Did you provide a dataset-specific entry point?
- [ ] Did you reuse an existing shared schema / archetype helper instead of adding another family hierarchy?

### 8.2 Training

- [ ] Can one of the existing 4 Lance schemas be reused?
- [ ] Does the dataset return `TrainSample`?
- [ ] Are images decoded inside the dataset instead of the collator?
- [ ] Is the default `sample_transform` a pickle-safe callable that can cross multiprocessing DataLoader workers?

### 8.3 Retrieval Eval

- [ ] Does it use the standard `queries + candidates + qrels` layout?
- [ ] Are data-fact fixes frozen in conversion?
- [ ] Are prompt/token/newline rules kept in runtime transforms whenever possible?
- [ ] Do query/candidate transforms avoid local `lambda`s / closures?

### 8.4 Tooling

- [ ] Is there already a Python converter that can be reused?
- [ ] Did you avoid adding a new hardcoded shell wrapper?
