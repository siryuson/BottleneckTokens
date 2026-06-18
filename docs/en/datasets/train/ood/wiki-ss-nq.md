# Wiki-SS-NQ Training Dataset

> Wikipedia screenshot question-answer retrieval dataset used for text-to-image retrieval training.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_retrieval` |
| Usage | Training / OOD evaluation |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | Wiki-SS-NQ |
| Paper | Wiki-SS-NQ: Wikipedia Screenshot Natural Questions |
| HuggingFace | [Tevatron/wiki-ss-nq](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) |
| Local Query Root | `./data/converted` |
| Local Corpus Root | `./data/converted` |
| License | Refer to the upstream dataset |

### Local Audit Summary

- `train.jsonl`: `44,002` queries
- `test.jsonl`: `3,610` queries
- screenshot corpus: `1,267,874` screenshots
- all `1000` MMEB-V2 eval queries appear inside local `train.jsonl`
- MMEB-V2 eval candidate `docid`s (`10,694`) do not overlap local train positive `docid`s

The standardized root therefore keeps two views:

- `official_train_raw`: preserves the local raw training query-positive pairs
- `official_train`: the safe default training view after removing the `1000` eval-overlap query texts

### Raw Schema

`train.jsonl`:

| Field | Type | Description |
|------|------|------|
| `query_id` | string | Query identifier |
| `query` | string | Raw question text |
| `positive_passages` | list[object] | Positive screenshot passages with `docid/title/text` |
| `negative_passages` | list[object] | Negative text passages, not directly consumed by the current training loader |
| `answers` | list[string] | Reference answers |

`wiki-ss-corpus-new/train`:

| Field | Type | Description |
|------|------|------|
| `image` | PIL.Image / bytes | Wikipedia screenshot |
| `docid` | string | Sequential corpus row id; training positives use the filename stem from `image.path` |

---

## BToks Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `wikissnq_train` |
| Implementation | `src/vlm2emb/data/datasets/wikissnq_train.py` |
| Registry Name | `wikissnq_train` |
| Conversion Script | `scripts/convert/train/convert_wikissnq_to_lance.py` |
| Standardized Root | `./data/converted` |

### Lance Layout

- `data/images.lance`
  - shared screenshot corpus
  - fields: `path: string`, `image: binary`
  - `path` uses the filename stem from `image.path`, aligned with training `positive_docid`
- `data/official_train_raw.lance`
  - one row per `query x positive_passage`
  - positives are exploded from `positive_passages`
- `data/official_train.lance`
  - default safe training view with MMEB-V2 eval-overlap queries removed
- `data/official_train_without_mmeb_v2_eval.lance`
  - equivalent to `official_train`; the name makes the MMEB-V2 eval exclusion explicit
- `data/exclusions/mmeb_v2_eval.lance`
  - filtered query-positive pairs kept for audit, not used as default training input

### Current Sample Counts

- `official_train_raw`: `340,094`
- `official_train`: `332,518`
- `official_train_without_mmeb_v2_eval`: `332,518`
- `exclusions/mmeb_v2_eval`: `7,576`
- `images`: `1,267,874`

### Training-side Mapping

- query: `Find the document image that can answer the given query: {query}`
- positive: `<|image_pad|>\nRepresent the given image`
- negative: empty text (placeholder)

### Transform Parameters

The default transform is owned by the `wikissnq_train` runtime. `transform.query.*`, `transform.positive.*`, and `transform.negative.*` expose the default behavior in config and do not depend on the old text-format path.

Runtime only owns model-visible surface rules such as instruction/body joining, visual token placement, and trailing newlines. It does not detokenize or otherwise clean the source query text, so upstream tokenization artifacts such as `n 't` remain preserved.

### Config Example

```yaml
Wiki-SS-NQ:
  type: wikissnq_train
  path: ./data/converted
  dataset_name: Wiki-SS-NQ
  split: official_train_without_mmeb_v2_eval
  weight: 1
  transform:
    query:
      instruction: "Find the document image that can answer the given query:"
      instruction_body_separator: space
      trailing_newline: ensure_single
    positive:
      instruction: "Represent the given image"
      visual_token_placement: own_line
      trailing_newline: ensure_single
    negative:
      empty: empty_multimodal_input
```

### Sample

Example query:

```text
Find the document image that can answer the given query: who founded the red cross to relieve the suffering of the war wounded
```

Example positive:

```text
<|image_pad|>
Represent the given image
```

Notes:

- the query carries no image
- the positive is one Wikipedia screenshot
- `official_train` already excludes samples whose query text overlaps MMEB-V2 / Wiki-SS-NQ eval queries
