# ViDoRe TAT-DQA

> TAT-DQA retrieval training subset in the ViDoRe family.

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
| Original Name | `vidore/colpali_train_set` (tatdqa subset) |
| Paper | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| Subset Name | tatdqa |
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
| Filter | `source = "tatdqa"` |

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

The examples below are sampled directly from `./data/converted` with `source = "tatdqa"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- Other fields keep their Lance values.

#### Example 1

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 516804,
        "sha1": "b28528c51549",
        "format": "PNG",
        "size": [
          1650,
          2150
        ]
      }
    },
    "image_filename": "data/downloaded_datasets/tatdqa/train/7b920fbe828615563dcc4230356c0282.pdf",
    "query": "What was the increase in income before income taxes in Semiconductor Test driven by?",
    "answer": "['by an increase in semiconductor tester sales for 5G infrastructure and image sensors, partially offset by a decrease in sales in the automotive and analog test segments.']",
    "source": "tatdqa",
    "options": null,
    "page": "1",
    "model": "",
    "prompt": "",
    "answer_type": "span"
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
        "bytes": 274117,
        "sha1": "9a10a7ab0b91",
        "format": "PNG",
        "size": [
          1700,
          2200
        ]
      }
    },
    "image_filename": "data/downloaded_datasets/tatdqa/train/d775277402669fd93e81ed268607ba0c.pdf",
    "query": "What was the % change in the free cash flow from 2017 to 2018?",
    "answer": "-456.96",
    "source": "tatdqa",
    "options": null,
    "page": "1",
    "model": "",
    "prompt": "",
    "answer_type": "arithmetic"
  }
}
```
