# VisRAG InfoVQA

> InfoVQA retrieval training subset in the VisRAG family.

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
| Original Name | `openbmb/VisRAG-Ret-Train-In-domain-data` (InfoVQA subset) |
| Paper | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| Subset Name | InfoVQA |
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
| Filter | `source = "InfoVQA"` |

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

The examples below are sampled directly from `./data/converted` with `source = "InfoVQA"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- `query` keeps the raw Lance value instead of the runtime prompt-expanded form.

#### Example 1

```json
{
  "data/train.lance": {
    "query": "which are the countries with Chinese speaking population other than China according to this infographic?",
    "image": {
      "path": "1.png",
      "bytes": {
        "bytes": 262214,
        "sha1": "cac24738166d",
        "format": "JPEG",
        "size": [
          564,
          3040
        ]
      }
    },
    "source": "InfoVQA"
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "query": "How many people will be there on earth by 2100?",
    "image": {
      "path": "3.png",
      "bytes": {
        "bytes": 307471,
        "sha1": "8a68d32d2d35",
        "format": "JPEG",
        "size": [
          900,
          2239
        ]
      }
    },
    "source": "InfoVQA"
  }
}
```
