# N24News

> Multimodal news classification dataset covering 24 news categories.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_classification` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | N24News |
| Paper | N24News: A New Dataset for Multimodal News Classification ([arXiv](https://arxiv.org/abs/2108.13327)) |
| Primary Authors | Zhen Wang, Xu Shan, Xiangxie Zhang, Jie Yang |
| HuggingFace | [linxy/Multimodal-N24News](https://huggingface.co/datasets/linxy/Multimodal-N24News) |
| License | See original dataset |

### Schema
Image + news text/title + category label; unified as image queries and candidate labels in MMEB.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `N24News` |
| Split | `test` |
| Eval Parser | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/N24News/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json` (unified three-table structure).

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_21`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given news image with the following caption for domain classification: Anthony Pidgeon\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 189554,
          "sha1": "1c213c84dcf6",
          "format": "JPEG",
          "size": [
            1050,
            549
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_21",
      "text": "Travel",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_21"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### Example 2
`query_id=q_1`, `positive_candidate_id=c_17`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given news image with the following caption for domain classification: Ms. Goodman styled Amber Valletta with wings for a 1993 shoot by Peter Lindbergh for Harper's Bazaar.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 138959,
          "sha1": "004cebd3131c",
          "format": "JPEG",
          "size": [
            1050,
            550
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_17",
      "text": "Style",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_17"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
