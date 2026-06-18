# ViDoRe DocVQA

> DocVQA retrieval training subset in the ViDoRe family.

| Property | Value |
|------|-----|
| Modality | visdoc |
| Task Type | `document_retrieval` |
| Usage | Training |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | `vidore/colpali_train_set` (docvqa subset) |
| Paper | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| Subset Name | docvqa |
| License | See original dataset |

### Schema

| Field | Type | Description |
|------|------|------|
| image | struct{bytes, path} | Document page image |
| query | string | Query text |
| answer | string | Positive text answer |

### Sample

> Full examples are provided in the BToks Lance section below.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `vidore_train` |
| Implementation | `src/vlm2emb/data/datasets/vidore.py` |
| Registry Name | `vidore_train` |
| Lance Path | `./data/converted` |
| Filter | `source = "docvqa"` |

### Lance Layout

**train.lance (Single Table)**:

| Column | Type | Description |
|------|------|------|
| image | struct{bytes: binary, path: string} | Document page image |
| image_filename | string | Original image filename |
| query | string | Query text |
| answer | string | Positive text answer |
| source | string | Source subset |
| options | string | Auxiliary upstream metadata |
| page | string | Page identifier when available |
| model | string | Upstream model metadata |
| prompt | string | Upstream prompt metadata |
| answer_type | string | Answer-type metadata |

### Training-side Mapping
- query: runtime transform turns it into `"<|image_pad|> {query}"` plus the single page image
- positive: `answer` as a text-only positive sample
- negative: empty text placeholder
- `source / page / answer_type` and related columns stay in storage metadata and are not part of the minimal training view

### Sample

The examples below are sampled directly from `./data/converted` with `source = "docvqa"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- Other fields keep their Lance values.

#### Example 1

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 63908,
        "sha1": "3ca701571c57",
        "format": "PNG",
        "size": [
          1709,
          2293
        ]
      }
    },
    "image_filename": "0fd47b51ae9248ef36669b8619b1223f268edae3e7a44ac1e6cebbbfaaf69f96",
    "query": "What is the date?\nYour answer should be very brief.",
    "answer": "OCTOBER 17, 1995.",
    "source": "docvqa",
    "options": null,
    "page": null,
    "model": null,
    "prompt": null,
    "answer_type": null
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 1142718,
        "sha1": "5a1beea60ca5",
        "format": "PNG",
        "size": [
          1776,
          2274
        ]
      }
    },
    "image_filename": "888b7dc9b346313f46c431c0b45ab723adc2b3e9a5a5f391702e797f90e6e659",
    "query": "To whom should the form be returned?\nEnsure brevity in your answer. ",
    "answer": "James F. Glenn.",
    "source": "docvqa",
    "options": null,
    "page": null,
    "model": null,
    "prompt": null,
    "answer_type": null
  }
}
```
