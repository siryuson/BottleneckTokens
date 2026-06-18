# VisRAG ArXiv QA

> ArXiv QA retrieval training subset in the VisRAG family.

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
| Original Name | `openbmb/VisRAG-Ret-Train-In-domain-data` (ArxivQA subset) |
| Paper | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| Subset Name | ArxivQA |
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
| Filter | `source = "ArxivQA"` |

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

The examples below are sampled directly from `./data/converted` with `source = "ArxivQA"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- `query` keeps the raw Lance value instead of the runtime prompt-expanded form.

#### Example 1

```json
{
  "data/train.lance": {
    "query": "In subfigure (b), which relationship does the blue dotted line indicate?",
    "image": {
      "path": "129.png",
      "bytes": {
        "bytes": 185227,
        "sha1": "57ba31515f18",
        "format": "JPEG",
        "size": [
          1680,
          1484
        ]
      }
    },
    "source": "ArxivQA"
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "query": "Which feature selection method(s) consistently included the feature 'CD44+ CD25+' in the model?",
    "image": {
      "path": "131.png",
      "bytes": {
        "bytes": 274409,
        "sha1": "b51d6669f045",
        "format": "JPEG",
        "size": [
          2016,
          1276
        ]
      }
    },
    "source": "ArxivQA"
  }
}
```
