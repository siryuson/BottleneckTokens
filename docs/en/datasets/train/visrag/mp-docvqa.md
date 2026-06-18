# VisRAG MP-DocVQA

> MP-DocVQA retrieval training subset in the VisRAG family.

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
| Original Name | `openbmb/VisRAG-Ret-Train-In-domain-data` (MP-DocVQA subset) |
| Paper | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| Subset Name | MP-DocVQA |
| License | See original dataset |

### Schema

| Field | Type | Description |
|------|------|------|
| query | string | Query text |
| image | struct{bytes, path} | Document page image |
| source | string | Data source |

### Sample

> Full examples are provided in the BToks Lance section below.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `visrag_train` |
| Implementation | `src/vlm2emb/data/datasets/visrag.py` |
| Registry Name | `visrag_train` |
| Lance Path | `./data/converted` |
| Filter | `source = "MP-DocVQA"` |

### Lance Layout

**train.lance (Single Table)**:

| Column | Type | Description |
|------|------|------|
| query | string | Query text |
| image | struct{bytes: binary, path: string} | Document page image |
| source | string | Source category |

### Training-side Mapping
- query: runtime injects a source-specific query prompt based on `source`
- positive: runtime injects a source-specific target prompt and attaches the single page image
- negative: empty text placeholder
- Lance stores `query / image / source` only, without materializing prompts ahead of time

### Sample

The examples below are sampled directly from `./data/converted` with `source = "MP-DocVQA"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- `query` keeps the raw Lance value instead of the runtime prompt-expanded form.

#### Example 1

```json
{
  "data/train.lance": {
    "query": "What is the mean value of children below 15 yrs?",
    "image": {
      "path": "2561.png",
      "bytes": {
        "bytes": 249493,
        "sha1": "7364bb730f86",
        "format": "JPEG",
        "size": [
          1685,
          2187
        ]
      }
    },
    "source": "MP-DocVQA"
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "query": "What is the dosage of Temik used for SI Method at Planting time?",
    "image": {
      "path": "2563.png",
      "bytes": {
        "bytes": 402856,
        "sha1": "feaaec468e65",
        "format": "JPEG",
        "size": [
          1689,
          2194
        ]
      }
    },
    "source": "MP-DocVQA"
  }
}
```
