# HatefulMemes

> Multimodal hateful expression detection dataset for testing image-text semantics and bias robustness.

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
| Original Name | Hateful Memes Challenge Dataset |
| Paper | The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes ([arXiv](https://arxiv.org/abs/2005.04790)) |
| Primary Authors | Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, Davide Testuggine |
| HuggingFace | [neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes) |
| License | See original dataset |

### Schema
Image, text, and binary label; unified as a query/candidate retrieval-style evaluation structure in MMEB.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `HatefulMemes` |
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
| Directory | `MMEB-V2/image-tasks/HatefulMemes/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json` (unified three-table structure).

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 20806,
          "sha1": "6208f633356e",
          "format": "JPEG",
          "size": [
            283,
            400
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1",
      "text": "Yes",
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
      "c_1"
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
`query_id=q_1`, `positive_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 38697,
          "sha1": "f0f97a6ac779",
          "format": "JPEG",
          "size": [
            550,
            366
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1",
      "text": "Yes",
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
      "c_1"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
