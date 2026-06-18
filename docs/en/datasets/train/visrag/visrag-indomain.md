# VisRAG In-domain

> VisRAG in-domain document retrieval training subset, directly converted to BToks Lance single-table format, with source-specific prompts injected by source.
> `query/image/source` are stored facts; source-specific prompts and `<|image_pad|>` handling are runtime transform policy.

| Property | Value |
|------|-----|
| Modality | visdoc |
| Task Type | `document_retrieval` |
| Usage | Training |
| Provenance Layers | 2 layers |
| Training Weight | 12 |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | `openbmb/VisRAG-Ret-Train-In-domain-data` |
| Paper | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| Primary Authors | Yushi Hu, Ziyang Luo, Yixin Cao, Shuai Yan, Jiazhen Gu, Fei Huang, Jingren Zhou, Zhoujun Li |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| License | See original dataset |

### Schema

| Field | Type | Description |
|------|------|------|
| query | string | Query text |
| image | struct{bytes, path} | Document page image |
| source | string | Data source (10 categories) |

Supported sources (10 categories): NeurIPS Papers, Textbooks, ICML Papers, Manuallib, ArxivQA, ChartQA, MP-DocVQA, InfoVQA, PlotQA, SlideVQA.

### Multi-source Samples (Format Examples)
- `source=NeurIPS Papers`: Query side concatenates "top conference paper retrieval" type prompts; positive side is paper page image.
- `source=ChartQA`: Query side concatenates "chart retrieval" type prompts; positive side is chart image.
- `source=MP-DocVQA`: Query side concatenates "multi-page document localization" type prompts; positive side is document page image.

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

### Lance Layout

**train.lance (Single Table)**:

| Column | Type | Description |
|------|------|------|
| query | string | Query text |
| image | struct{bytes: binary, path: string} | Document page image |
| source | string | Source category |

### Training-side Mapping
- query: runtime transform emits `"{source_prompt}{query}\n"`
- positive: runtime transform emits `"<|image_pad|>\n{target_prompt}\n"` and attaches the single page image
- negative: empty text placeholder
- Lance stores `query / image / source` only, without materializing prompts ahead of time

### Sample

The examples below are sampled directly from `./data/converted`.

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

#### Example 3

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
