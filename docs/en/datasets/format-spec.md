# Dataset Format Specification

This document defines the standard formats for training and evaluation datasets in the BToks framework, and clarifies the source / conversion / runtime boundaries used for onboarding new datasets and future extension.

## 1. Project-Level Data Guidelines

### 1.1 Three-Layer Responsibility Boundary

Every dataset onboarding flow should distinguish three layers first:

1. **source layer**
- raw download source
- raw split and raw directory layout
- no model-facing prompt / token semantics

2. **conversion layer**
- aggregate raw sources
- fix data facts: source freeze, path normalization, broken-sample handling, qrels/label/page-index fixes, stable IDs
- produce stable Lance storage; basic training conversion preserves raw fields by default and does not generate manifest, README, dataset_infos, or split inventory

3. **runtime dataset layer**
- read samples from Lance
- apply runtime-adjustable prompt/token/newline/layout logic without changing stored facts

Rule:
- **data facts are frozen in conversion**
- **input formatting stays adjustable in runtime**
- **all maintained Lance conversion scripts must use `max_bytes_per_file=2 * 1024 * 1024 * 1024`**
- **do not use `max_rows_per_file` as the physical Lance file sharding policy; `max_rows_per_group` may remain only as an internal row-group tuning knob**

### 1.2 MMEB-V2 Eval as a Special Case

MMEB-V2 is not a simple “store raw data as-is” dataset. It currently involves:
- multi-source aggregation
- historical version drift
- path, media, broken-sample, and qrels issues
- strict / canonical semantics that must be frozen explicitly

Therefore MMEB-V2 is allowed to apply one-time conversion-layer fixes to guarantee stable storage, but neither `strict` nor `canonical` should keep prompt/token/newline/layout as conversion-default model-facing behavior.

Additional note:

- `MMEB-V2 eval` benchmark materialization may combine multiple sources; on the image side this explicitly includes MMEB-V2 media together with MMEB-eval annotations.
- This composite-source assembly belongs to the benchmark/data-fact boundary rather than to runtime text transforms.
- `strict` storage should preserve raw content together with metadata required for runtime reconstruction, while `canonical` storage should preserve data-fact canonicalization results. Final prompt/token/newline/layout belongs to runtime in both cases.

### 1.3 Onboarding Principles for New Datasets

- Every new dataset should have a **dataset-specific entry point**.
- Underneath that entry point, prefer **shared schemas, task-archetype helpers, and thin dataset wrappers** over extending a thicker family-class hierarchy.
- The standard storage format for retrieval evaluation is fixed as `queries + candidates + qrels`.
- Conversion owns storage correctness; runtime datasets own input flexibility.
- If a change mainly affects the text seen by the model rather than stored data facts, it should not live in conversion by default.

### 1.4 Family vs. Archetype Responsibilities

In this project, `family` and `task archetype` are different layers:

- `family`: primarily about storage layout, upstream source shape, and local helper reuse
- `task archetype`: primarily about the minimal semantic structure of query / candidate / positive / negative / qrels

Future shared boundaries should be built around **task archetypes**, not around a larger family-wide default runtime semantics center.

Rules:

- the archetype owns the minimal shared contract, runtime view, and validation primitive
- the family owns only schema/layout-level reuse
- the dataset-owned module keeps source-specific prompts, metadata mapping, and historical compatibility behavior
- mixture weights, oversampling, and bidirectional expansion do not belong to the archetype or family layers and should move into a separate mixture compiler later
- ready datasets should be discovered through explicit dataset registry / config entries; the project no longer keeps a generic manifest catalog compatibility package

## 2. Training Dataset Format

### 2.0 Basic Training Raw Storage / Runtime Transform Boundary

The new main path for basic training datasets is raw-preserving conversion plus runtime transform:

- conversion preserves source-table fields whenever possible and only aggregates many small media files into Lance side tables.
- conversion does not normalize different datasets into one unified training-field schema; it may rename tables, reorganize layout, and materialize derived media, while query / positive / negative semantics are interpreted by the runtime Dataset.
- runtime Dataset code owns `TrainSample` construction, visual-token normalization, media decoding, and `MultiModalInput` output for each side.
- Video training data uses `data/videos.lance` for aggregated raw videos and `data/frames.lance` for pre-extracted derived frames. The default training runtime reads `data/frames.lance` and should fail when it is missing, unless the config explicitly requests the raw-video compatibility path.
- the four migrated basic training families (MMEB, ViDoRe, VisRAG, LLaVA-Hound) no longer depend on `materialize_training_view`; the old `src/vlm2emb/data/contracts/` package has been removed.
- sample validation must either live in a unified validate boundary or be covered by standalone validation; runtime must not grow a partial ad hoc validator.
- datasets that have not migrated yet may keep legacy helpers temporarily; new implementations must not keep spreading legacy `render` / materialized-view naming.

### 2.1 Unified Output Format: TrainSample

All migrated training datasets output `TrainSample` as defined in `src/vlm2emb/data/schema.py`:

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
    negative: MultiModalInput  # text="" + media=[] means no negative
    metadata: dict[str, Any]
```

`TrainSample` is the top-level training relation. `MultiModalInput` is the unified `text + media` input for each side. `media` is not a top-level `TrainSample` field; it belongs to each side's `MultiModalInput`.

### 2.2 Four Lance Schemas

#### 2.2.1 mmeb_train (MMEB Image Training)

Raw sample table plus image side table:

- **Sample table** (`data/{subset}/{split}.lance`):
  | Column | Type | Description |
  |--------|------|-------------|
  | * | * | Raw parquet fields are preserved as-is, for example `qry`, `qry_image_path`, `pos_text`, `pos_image_path`, `neg_text`, `neg_image_path` |

- **Images table** (`data/images/{subset}.lance`):
  | Column | Type | Description |
  |--------|------|-------------|
  | path | string | Raw parquet image path string and runtime join key |
  | image | binary | Image binary data |

- **Implementation**: `src/vlm2emb/data/datasets/mmeb_train.py`
- **Registry name**: `mmeb_train`
- **Subsets**: 20 (ImageNet_1K, N24News, ...)

#### 2.2.2 vidore_train (ViDoRe Document Retrieval Training)

Single-table structure:

- **Data table** (`data/train.lance`):
  | Column | Type | Description |
  |--------|------|-------------|
  | image | struct{bytes: binary, path: string} | Document page image |
  | query | string | Query text |
  | answer | string | Positive-side answer text |

- **Implementation**: `src/vlm2emb/data/datasets/vidore.py`
- **Registry name**: `vidore_train`
- **Provenance**: 2-layer (original → Lance)

#### 2.2.3 visrag_train (VisRAG Document Retrieval Training)

Single-table structure with source-specific prompt templates:

- **Data table** (`data/train.lance`):
  | Column | Type | Description |
  |--------|------|-------------|
  | query | string | Query text |
  | image | struct{bytes: binary, path: string} | Document page image |
  | source | string | Data source (10 types: NeurIPS Papers, Textbooks, ICML Papers, Manuallib, ArxivQA, ChartQA, MP-DocVQA, InfoVQA, PlotQA, SlideVQA) |

- **Prompt template**: Different prompt prefix selected based on `source` field
- **Implementation**: `src/vlm2emb/data/datasets/visrag.py`
- **Registry name**: `visrag_train`
- **Provenance**: 2-layer (original → Lance)

#### 2.2.4 llavahound_train (LLaVA-Hound Video Training)

Dual-table structure:

- **Frames table** (`data/train_300k.lance`):
  | Column | Type | Description |
  |--------|------|-------------|
  | video_id | string | Video ID |
  | frame_idx | string | Frame ordering key |
  | image | binary | Frame image binary data |

- **Instruction table** (`data/video_instruction/{subset}.lance`):
  | Column | Type | Description |
  |--------|------|-------------|
  | id | string | Sample ID |
  | video | string | `video_id` used to join the frame table |
  | conversations | list[struct] | Conversation list |

- **Three modes**:
  - `caption_retrieval`: Video→caption retrieval (video_caption_300k)
  - `video_retrieval`: Caption→video retrieval (video_caption_300k-video)
  - QA mode: Question-answer pairs (video_qa_240k)
- **Implementation**: `src/vlm2emb/data/datasets/llavahound.py`
- **Registry name**: `llavahound_train`
- **Provenance**: 2-layer (original → Lance)

### 2.3 Multimodal Token Constants

Defined in `src/vlm2emb/data/datasets/const.py`:

| Constant | Value | Usage |
|----------|-------|-------|
| `STANDARD_IMAGE_TOKEN` | `<\|image_pad\|>` | Standard format (Qwen). New datasets SHALL use this token |
| `PHI3V_IMAGE_TOKEN` | `<\|image_1\|>` | Original MMEB format (Phi-3V), for compatibility only |
| `STANDARD_VIDEO_TOKEN` | `<\|video_pad\|>` | Video frame token |

> **Rule**:
> - Project-level runtime text formatting should prefer `STANDARD_IMAGE_TOKEN` / `STANDARD_VIDEO_TOKEN`
> - Historical tokens such as `PHI3V_IMAGE_TOKEN` remain compatibility inputs only
> - `MMEB-V2 eval` should no longer treat legacy-token replacement, token-layout cleanup, or trailing-newline policy as conversion-default responsibilities; these model-visible input surfaces are governed by the runtime surface rules below
> - New non-MMEB datasets should keep their original content and schema by default rather than inheriting project-wide strict/canonical token semantics

### 2.4 Runtime Surface Rules

This section merges the former `runtime-surface-rules.md`. It defines the **text/media surface** that the model actually sees, including:

- whether visual tokens appear and where they appear
- whether visual tokens use a space or newline before body text
- how instruction and body are joined
- whether trailing newlines are kept
- how option blocks, paragraph blocks, and trailing blanks are preserved

Scope:

- runtime text transforms and canonical convergence for evaluation data
- runtime text transforms and prompt materialization for training data
- discussions about conversion/runtime ownership
- text/media surface audits when onboarding new datasets

This does not mean:

- every training dataset must mechanically follow the same surface
- every parser must share one implementation
- every subset must share the same default values

Training can use this section as the default reference baseline and still apply controlled perturbations for robustness, such as toggling style-level switches, sampling different instruction/body separators, or lightly randomizing semantically equivalent surfaces. Training must not break runtime invariants, especially `visual_token_alignment`.

#### 2.4.1 origin and plan canonical

The reference target for `origin` is **original parser behavior**, not raw Lance fields copied as-is.

That means:

- if the original parser assembles a prompt, `origin` follows that prompt
- if the original parser appends a newline, `origin` follows that surface
- if the original parser uses dataset-local instructions instead of raw `qry_instruction`, `origin` follows the parser output

Going forward, dataset-level defaults should keep only two viewpoints:

- `origin`
- `plan canonical`

We should no longer keep “current canonical / current non-origin” as a long-term third table dimension. If current implementation still differs from `plan canonical`, explain that difference in prose instead of adding another persistent column.

#### 2.4.2 Core rule vocabulary

The project-level rule vocabulary is currently fixed to 10 rules:

| Rule | Meaning |
| --- | --- |
| `trailing_newline` | Whether trailing newline is kept, and in what form |
| `whitespace_normalization` | Normalization of extra spaces, bad joins, trailing spaces, etc. |
| `blankline_trim` | Trimming extra blank lines |
| `instruction_body_separator` | Connection style between instruction and body |
| `choice_prefix_cleanup` | Cleanup of option prefixes like `(A)`, `A.`, `1.` |
| `terminal_format_artifact_cleanup` | Cleanup of trailing formatting artifacts, punctuation tails, or trailing spaces |
| `visual_token_variant_normalization` | Normalization of legacy token variants such as `<|image_1|>`, `<image>`, `<video>` |
| `visual_token_alignment` | Token/media count, type, and order consistency |
| `visual_token_placement` | Whether a visual token is prepended, standalone, or embedded |
| `visual_token_separator` | Whether a visual token uses a space or newline before body text |

`visual_token_alignment` is not a style toggle. It is a **runtime invariant**. It requires:

- token count matches media slot count
- token type matches media type
- token order matches media order

Therefore:

- media without token: must be repaired or fail-fast
- token without media: must be removed or fail-fast
- repeated token: treated as alignment violation

Training and evaluation must both keep this invariant.

#### 2.4.3 Project-level plan canonical conclusions

Two `plan canonical` conclusions currently matter most.

First, if a visual token is a standalone modality placeholder rather than part of sentence semantics, it should occupy its own line:

```text
<|image_pad|>
body...
```

or:

```text
<|video_pad|>
body...
```

instead of:

```text
<|video_pad|> body...
```

Second, once body text already forms a complete prompt block, whether it is a single instruction sentence, a multi-line option block, a dialog block, or a compact `(A) yes; (B) no.` structure, `plan canonical` should generally prefer:

- `trailing_newline=ensure_single`

Exceptions:

- if `origin` already has mixed trailing states, `origin` should remain `preserve`
- some special queries such as `VisDial.query` are better modeled as `normalize_blank_tail`

#### 2.4.4 Four trailing_newline states

`trailing_newline` is currently split into four states:

| State | Meaning |
| --- | --- |
| `preserve` | Keep the original parser/content trailing state as-is |
| `strip` | Explicitly require no trailing newline |
| `ensure_single` | Explicitly require exactly one trailing newline |
| `normalize_blank_tail` | Collapse excessive blank tail while preserving end-of-block intent |

The clearest current `normalize_blank_tail` example is `VisDial.query`.

#### 2.4.5 Current rough default judgments

This section is a project-level default summary. It does not replace subset-level ledgers.

**image classification / image QA**

- `query`
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- `candidate`
  - usually does not need `visual_token_separator`
  - `trailing_newline=preserve`

**image retrieval / grounding**

- most image-backed prompt/query forms lean toward:
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- template-style image candidates usually lean toward:
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
  - if the candidate body is a generic instruction owned by the parser or config, it should be a complete sentence, such as `Represent the given cropped image of the object.`, not a period-less fragment.
- class-name, single-word, or phrase-like candidates usually lean toward:
  - `trailing_newline=strip`
- complete sentences, multi-sentence captions, or candidates assembled from an instruction usually lean toward:
  - `trailing_newline=ensure_single`
- borderline answer / caption cases should be decided in the dataset-specific config and sample review, not by a mechanical length rule.

Key exceptions:

- `VisDial.query`
  - `instruction_body_separator=newline`
  - `trailing_newline=normalize_blank_tail`
- `VisDial.candidate`
  - `origin` keeps the single-line image-token-plus-instruction form
  - `plan canonical` converges to standalone image token plus a single trailing newline
- `WebQA / EDIS.query`
  - stripping visual tokens from text-only queries is no longer treated as a canonical style choice
  - it should be treated as a `visual_token_alignment` invariant repair and record `surface_repairs` provenance
- `VizWiz.query`
  - `plan canonical` enables narrow `whitespace_normalization`
  - it only removes stray trailing spaces and keeps the parser-owned single trailing newline

MMEB-train reuses these surface rules as the new Dataset output standard. The currently confirmed
training-side exceptions are:

- `WebQA.query`
  - legacy image tokens left in text-only queries are handled as a `visual_token_alignment` repair
  - `trailing_newline=ensure_single`
- `ChartQA.positive` / `ChartQA.negative`
  - enable narrow `whitespace_normalization` for answer text
  - only strip trailing horizontal whitespace, without changing newlines or answer content
- `MSCOCO.positive`
  - template-style image positive
  - visual token on its own line
  - `trailing_newline=ensure_single`
- `VisDial.query`
  - `instruction_body_separator=newline`
  - `trailing_newline=normalize_blank_tail`
- `VisDial.negative`
  - when the negative side has an image but empty raw text, restore it as an image token on its own line
  - `trailing_newline=ensure_single`

**video classification**

- `query`
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- `candidate`
  - mostly short labels
  - `trailing_newline=strip`

Typical shape:

```text
<|video_pad|>
Recognize the category of the video content.
```

**video QA**

- `query`
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- `candidate`
  - mostly answer text
  - `trailing_newline=strip`

Internal option-block newlines in `Video-MME / MVBench / NExTQA` are body structure and must be preserved. This rule is about the final trailing newline, not internal option-block newlines.

**video retrieval / moment retrieval**

- `MSR-VTT / MSVD / DiDeMo / VATEX / YouCook2`
  - `query.trailing_newline=ensure_single`
  - `candidate.visual_token_separator=newline`
  - `candidate.trailing_newline=ensure_single`
- `QVHighlight / Charades-STA / MomentSeeker`
  - `query.visual_token_separator=newline`
  - `query.trailing_newline=ensure_single`
  - `candidate.visual_token_separator=newline`
  - `candidate.trailing_newline=ensure_single`

**visdoc**

- `query`
  - `origin` is usually closer to `preserve`
  - `plan canonical` converges to `trailing_newline=ensure_single`
- `candidate`
  - `origin` is usually `visual_token_separator=space, trailing_newline=strip`
  - `plan canonical` is usually `visual_token_separator=newline, trailing_newline=ensure_single`

Key exceptions:

- `MMLongBench-page`
- `MMLongBench-doc`

These two subsets have mixed trailing states in raw queries, so for now we freeze:

- `origin=query preserve`
- `plan canonical=query ensure_single`

#### 2.4.6 Non-origin handling must be explicit

Any runtime surface handling that differs from `origin` must be:

1. explicitly named
2. explicitly commented
3. explicitly documented
4. tied to a specific dataset / subset

We should not keep:

- hidden helper repairs
- undocumented parser branches
- loader-level silent surface rewrites

### 2.5 MMEB-V2 Eval strict / canonical Relationship

- This section only applies to `MMEB-V2 eval`.
- `strict` is the MMEB-V2 reference conversion/evaluation contract. Its goal is to reproduce the output semantics of the corresponding the public reference parser contract parser through auditable runtime reconstruction.
- `canonical` is the production reprocessed branch for `MMEB-V2 eval`; it should not be generalized as the default storage semantics for ordinary training or evaluation datasets.
- `strict / canonical` may still own benchmark-materialization data-fact repairs. Those repairs belong to the conversion layer.
- Model-visible prompt, token layout, newline, option layout, and whitespace behavior belong to the runtime surface layer and follow `2.4 Runtime Surface Rules`.
- `strict` conversion should preserve raw content, media, qrels, identities, and metadata needed for runtime reconstruction; `strict` runtime is responsible for reproducing parser-visible text.
- `canonical` may repartition responsibilities between conversion and runtime, but the combined final output must follow the documented canonical data-fact repairs and runtime surface rules.
- Training datasets and other ordinary evaluation datasets do not use this strict/canonical storage contract; they keep original content by default and rely on runtime transforms for prompt/token adjustments.

Allowed MMEB-V2 canonical **data-fact repairs**:

- source freeze and composite-source materialization
- broken-sample removal, duplicate cleanup, and stable identity fixing
- qrels / label / page-index / candidate-identity repairs
- Path alias resolution: only when different paths point to the same underlying asset

Disallowed MMEB-V2 canonical repairs:

- ad hoc query / candidate wording rewrites or templating that were never documented
- unaudited prompt / token / newline / blank-line edits
- candidate order changes
- qrels semantic rewrites

### 2.6 Current MMEB-V2 Data-Fact Repair Coverage and Migration Note

The following items remain valid conversion-layer data-fact repairs:

| Category | Dataset / Scope | Canonical behavior |
|----------|-----------------|-------------------|
| Data | `ViDoSeek-page` | Use `siyrus/BToks-ViDoSeek-page-fixed` and apply the page-index fix |
| Data | `MMLongBench-page` | Use `siyrus/BToks-MMLongBench-page-fixed` and apply the page-index fix |
| Data | `RefCOCO-Matching` | Deduplicate repeated `candidate_id` entries in qrels |
| Data | `MomentSeeker` | Freeze to the deduplicated `siyrus/BToks-MomentSeeker` 1.6k official variant |
| Data | `MMLongBench-doc` | Explicitly keep graded qrels instead of switching to binary qrels |
| Data | `ViDoRe_esg_reports_human_labeled_v2` | Prefer the `VLM2Vec` mirror as canonical source |
| Path | `video_retrieval` family | Normalize extracted frame paths with the same semantics and never fall back to raw video |

The following items are no longer described as conversion data-fact repairs. They belong to runtime surface rules:

| Scope | Current ownership |
|-------|-------------------|
| Legacy image tokens left in text-only `WebQA` / `EDIS` queries | `visual_token_alignment` invariant repair; not a canonical-only style rule |
| `VisDial` query instruction/body newline | `instruction_body_separator` + `blankline_trim` / `normalize_blank_tail` |
| Model-visible legacy-token replacement | `visual_token_variant_normalization` |
| Standalone-token text-layout cleanup | `visual_token_separator` / `visual_token_placement` |
| Trailing newline, extra blank line, dirty join formatting | `trailing_newline` / `blankline_trim` / `whitespace_normalization` |

### 2.7 MMEB-V2 Upstream Issue Coverage

The current `MMEB-V2 eval` `strict / canonical` pipeline already covers the following public upstream issues:

| Source | Issue | Current handling |
|--------|-------|------------------|
| VLM2Vec `#167` | `ViDoSeek-page` / `MMLongBench-page` page-index bug | `strict` preserves the original behavior; `canonical` switches to fixed sources and applies the fix |
| MMEB-V2 discussion `#5` | `MomentSeeker` path sensitivity for `query_images` / `video_frames` | Both `strict` and `canonical` stay `frame-only` and keep path semantics compatible |
| MMEB-V2 discussion `#9` | `video_ret.tar.gz` extracts with a retained path prefix | `strict` allows path aliases; `canonical` normalizes frame paths with the same semantics |
| Official `MomentSeeker` variants | `1.8k` vs deduplicated `1.6k` source ambiguity | `strict` freezes the source id; `canonical` fixes it to `siyrus/BToks-MomentSeeker` |
| VLM2Vec `#188` | `esg_reports_human_labeled_v2` uploaded image count does not match the corpus | `strict` trusts parser/source-of-truth rows; `canonical` prefers the `VLM2Vec` mirror source |
| MMEB-V2 discussion `#8` | Truncated / corrupted VisDoc images | `strict` does not silently replace media; `canonical` avoids confirmed packaging issues through a more stable source, without rewriting media content |

Paper or leaderboard discrepancies are outside the scope of this conversion layer.

### 2.8 Training Conversion Tooling Boundary

Training conversion tooling currently remains under `scripts/convert/train/`.

Python converters that are intentionally retained:

| Script | Role |
|--------|------|
| `convert_mmeb_train_to_lance.py` | MMEB-train to Lance |
| `convert_parquet_to_lance.py` | parquet dataset to Lance |
| `convert_jsonl_to_lance.py` | JSONL dataset to Lance |
| `convert_video_frames_to_lance.py` | video frame directories to Lance |
| `convert_llavahound_train_to_lance.py` | raw-preserving LLaVA-Hound dual tables to Lance |

Constraints:
- These scripts are for **training data** storage conversion. They do not own MMEB evaluation strict/canonical logic.
- Historical shell wrappers were machine-specific wrappers and have been removed from the maintained tool layer.
- New train conversion logic should go into the Python tool layer rather than adding more hardcoded shell entrypoints.
- All maintained `scripts/convert/**` Python converters must explicitly pass `max_bytes_per_file=2 * 1024 * 1024 * 1024` when writing Lance datasets.
- Training conversion must not use `max_rows_per_file` for physical file splitting; if tuning is needed, keep it at the `max_rows_per_group` level only.

## 3. Evaluation Dataset Format

### 3.1 MMEB-V2 Unified Lance Three-File Layout

This section describes the fixed storage layout for `MMEB-V2 eval`. It is not a project-wide hard requirement for every evaluation dataset.

The 78 `MMEB-V2 eval` subsets share the following directory structure:

```
{dataset_name}/
├── queries.lance/        # Query table
├── candidates.lance/     # Candidate table
├── qrels.lance/          # Relevance judgment table
└── metadata.json         # Metadata
```

Additional constraints:
- `queries.lance`, `candidates.lance`, and `qrels.lance` are Lance dataset directories whose underlying physical data files are all capped at `2GB` per file
- `MMEB-V2` conversion must not rely on `max_rows_per_file` for physical file splitting

Runtime loading constraints:
- `MMEB-V2 eval` no longer relies on generic schema scanning to guess columns; the maintained mainline entry is `src/vlm2emb/data/datasets/mmeb_v2/loader.py` -> `build_mmeb_eval_dataset(...)`
- the loader resolves the parser module by subset first, then reads the exact query/candidate raw columns declared by that parser
- the benchmark only forwards runtime knobs such as `runtime_mode` and `num_frames`; prompt wording, newline policy, option layout, and token separators now live in the dataset-local parser modules

#### queries.lance

| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique query identifier |
| text | string | Query text (with instruction) |
| images | list[binary] | Query image list (may be empty) |

#### candidates.lance

| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique candidate identifier |
| text | string | Candidate text |
| images | list[binary] | Candidate image list (may be empty) |

#### qrels.lance

| Column | Type | Description |
|--------|------|-------------|
| query_id | string | Associated query ID |
| mode | string | Evaluation mode: `"sparse"` or `"exhaustive"` |
| candidate_ids | list[string] | Related candidate ID list |
| candidate_scores | list[float32] | Corresponding relevance scores |

#### metadata.json

```json
{
    "name": "ImageNet-1K",
    "task_type": "image_classification",
    "num_queries": 1000,
    "num_candidates": 1000
}
```

### 3.2 Evaluation Mode (qrels.mode)

Evaluation mode is **per-query level** — different queries within the same dataset may have different modes:

| mode | Meaning | Evaluation Method |
|------|---------|-------------------|
| `"sparse"` | Global evaluation | Query ranked against **all** candidates |
| `"exhaustive"` | Local evaluation | Query ranked only against its `candidate_ids` |

### 3.3 task_type Enumeration

| Modality | task_type | Dataset Count |
|----------|-----------|---------------|
| image | `image_classification` | 10 |
| image | `image_question_answer` | 10 |
| image | `image_retrieval` | 12 |
| image | `image_visual_grounding` | 4 |
| video | `video_classification` | 5 |
| video | `video_question_answer` | 5 |
| video | `video_retrieval` | 5 |
| video | `video_moment_retrieval` | 3 |
| visdoc | `visdoc_vidore_v1` | 10 |
| visdoc | `visdoc_vidore_v2` | 4 |
| visdoc | `visdoc_visrag` | 6 |
| visdoc | `visdoc_ood` | 4 |

### 3.4 Aggregation Metrics

| Modality | Aggregation Metric |
|----------|--------------------|
| image | `hit@1` |
| video | `hit@1` |
| visdoc | `ndcg@5` |

## 4. Naming Conventions

| Context | Format | Example |
|---------|--------|---------|
| Training subset (in config) | Underscore | `ImageNet_1K`, `VisualNews_t2i` |
| Eval subset (in DATASET_REGISTRY) | Hyphen | `ImageNet-1K`, `VisualNews_t2i` |
| Documentation filename | kebab-case | `imagenet-1k.md`, `visualnews-t2i.md` |
| Dataset registry name | Task-specific snake_case | `mmeb_train`, `vidore_train` |
| Eval directory name | Matches DATASET_REGISTRY | `ImageNet-1K/`, `MSR-VTT/` |

## 5. New Dataset Onboarding Checklist

### 5.1 Training Dataset

- [ ] Select or create Lance schema (refer to Section 2.2 for the 4 existing schemas)
- [ ] Implement transform outputting `TrainSample`
- [ ] Register dataset class with `@AutoDataset.register()`
- [ ] Add import in `src/vlm2emb/data/datasets/__init__.py`
- [ ] Create YAML config file (refer to `configs/datasets/mmeb_train.yaml`)
- [ ] Use `STANDARD_IMAGE_TOKEN` instead of `PHI3V_IMAGE_TOKEN`
- [ ] Prefer runtime transforms over regenerating storage when only input formatting needs to change
- [ ] Is the default runtime transform a pickle-safe callable?
- [ ] Create provenance documentation card (under `docs/*/datasets/train/`)

### 5.2 Evaluation Dataset

- [ ] Decide whether the issue belongs to source / conversion / runtime first
- [ ] For retrieval, use the standard `queries + candidates + qrels` layout
- [ ] Freeze data-fact fixes (source/media/qrels/id) in conversion
- [ ] Keep prompt/token/newline formatting in runtime transforms whenever possible
- [ ] Do query/candidate transform hooks avoid local `lambda`s / closures and remain serializable across multiprocessing workers?
- [ ] Write a conversion script only when needed (refer to `scripts/convert/convert_*.py`)
- [ ] Convert to unified Lance three-file layout (queries.lance / candidates.lance / qrels.lance)
- [ ] Explicitly set `max_bytes_per_file=2 * 1024 * 1024 * 1024` when writing Lance datasets
- [ ] Generate metadata.json (including name, task_type, num_queries, num_candidates)
- [ ] Register in `DATASET_REGISTRY` (`src/vlm2emb/evaluation/benchmarks/mmeb.py`)
- [ ] Create provenance documentation card (under `docs/*/datasets/eval/`)
