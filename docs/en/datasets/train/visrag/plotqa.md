# VisRAG PlotQA

> PlotQA retrieval training subset in the VisRAG family.

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
| Original Name | `openbmb/VisRAG-Ret-Train-In-domain-data` (PlotQA subset) |
| Paper | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| Subset Name | PlotQA |
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
| Filter | `source = "PlotQA"` |

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

The examples below are sampled directly from `./data/converted` with `source = "PlotQA"` applied.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- `query` keeps the raw Lance value instead of the runtime prompt-expanded form.

#### Example 1

```json
{
  "data/train.lance": {
    "query": "How many groups of bars are there ?",
    "image": {
      "path": "257.png",
      "bytes": {
        "bytes": 32739,
        "sha1": "18061e626b38",
        "format": "PNG",
        "size": [
          1319,
          700
        ]
      }
    },
    "source": "PlotQA"
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "query": "In the year 2010, what is the difference between the number of enrolments in primary education and number of enrolments in secondary general education ?",
    "image": {
      "path": "259.png",
      "bytes": {
        "bytes": 38816,
        "sha1": "c126f78b119b",
        "format": "PNG",
        "size": [
          1224,
          650
        ]
      }
    },
    "source": "PlotQA"
  }
}
```
