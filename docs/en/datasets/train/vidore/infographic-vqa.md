# ViDoRe Infographic-VQA

> Infographic-VQA retrieval training subset in the ViDoRe family.

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
| Original Name | `vidore/colpali_train_set` (Infographic-VQA subset) |
| Paper | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| Subset Name | Infographic-VQA |
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
| Filter | `source = "Infographic-VQA"` |

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

The examples below are sampled directly from `./data/converted` with `source = "Infographic-VQA"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- Other fields keep their Lance values.

#### Example 1

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 71446,
        "sha1": "7f5a0b4db620",
        "format": "JPEG",
        "size": [
          590,
          843
        ]
      }
    },
    "image_filename": "e205f5ce26ecf2bd455ed1034eaf7764c873a63056b6e3e41a3a953843c0fbc3",
    "query": "How many tips are mentioned for prevention of coronavirus?\nAnswer briefly.",
    "answer": "10.",
    "source": "Infographic-VQA",
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
        "bytes": 190966,
        "sha1": "bd5eef9d78e7",
        "format": "JPEG",
        "size": [
          700,
          1700
        ]
      }
    },
    "image_filename": "9a71009d0e95bff815db2547e23c56b8c73e50ff332d1b722bdbf37c81f187ec",
    "query": "What is the inverse of the percentage of total job advertisements in London?\nGive a very brief answer.",
    "answer": "78.",
    "source": "Infographic-VQA",
    "options": null,
    "page": null,
    "model": null,
    "prompt": null,
    "answer_type": null
  }
}
```
