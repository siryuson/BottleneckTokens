# ViDoRe ArXiv QA

> ArXiv paper QA retrieval training subset in the ViDoRe family.

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
| Original Name | `vidore/colpali_train_set` (arxiv_qa subset) |
| Paper | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| Subset Name | arxiv_qa |
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
| Filter | `source = "arxiv_qa"` |

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

The examples below are sampled directly from `./data/converted` with `source = "arxiv_qa"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- Other fields keep their Lance values.

#### Example 1

```json
{
  "data/train.lance": {
    "image": {
      "path": "1810.07757_2.jpg",
      "bytes": {
        "bytes": 167632,
        "sha1": "229b31fafc52",
        "format": "JPEG",
        "size": [
          1000,
          1600
        ]
      }
    },
    "image_filename": "images/1810.07757_2.jpg",
    "query": "Comparing panels a, b, c, and d, which statement best describes the data variance?",
    "answer": "D",
    "source": "arxiv_qa",
    "options": "['A. The variance of the data decreases from panel a to panel d.', 'B. The variance of the data increases from panel a to panel d.', 'C. The data presents no variance in any of the panels.', 'D. The variance of the data is inconsistent across the panels.', '-']",
    "page": "",
    "model": "gpt4V",
    "prompt": "",
    "answer_type": null
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "image": {
      "path": "1711.09320_2.jpg",
      "bytes": {
        "bytes": 187866,
        "sha1": "d17043b3b2fa",
        "format": "JPEG",
        "size": [
          1705,
          1592
        ]
      }
    },
    "image_filename": "images/1711.09320_2.jpg",
    "query": "What trend can be observed in figure a when the strain (ε) is increased from 0% to 8%?",
    "answer": "C",
    "source": "arxiv_qa",
    "options": "['A. The value of γ_sl increases linearly.', 'B. The value of γ_sl remains constant.', 'C. The value of γ_sv decreases linearly.', 'D. The value of γ_sv increases exponentially.', '-']",
    "page": "",
    "model": "gpt4V",
    "prompt": "",
    "answer_type": null
  }
}
```
