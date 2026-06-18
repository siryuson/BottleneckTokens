# ImageNet-A

> Adversarial/hard natural-distribution ImageNet subset for testing robustness generalization.

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
| Original Name | ImageNet-A |
| Paper | Natural Adversarial Examples ([arXiv](https://arxiv.org/abs/1907.07174)) |
| Primary Authors | Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, Dawn Song |
| HuggingFace | [Maysee/tiny-imagenet-a](https://huggingface.co/datasets/Maysee/tiny-imagenet-a) |
| License | See original dataset |

### Schema
Image + ImageNet class label (hard examples).

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `ImageNet-A` |
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
| Directory | `MMEB-V2/image-tasks/ImageNet-A/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_90`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image for classification\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 15440,
          "sha1": "11a2813d1749",
          "format": "JPEG",
          "size": [
            236,
            314
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_90",
      "text": "Rottweiler",
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
      "c_90"
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
`query_id=q_1`, `positive_candidate_id=c_90`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image for classification\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 7255,
          "sha1": "c0de3dc2b653",
          "format": "JPEG",
          "size": [
            300,
            300
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_90",
      "text": "Rottweiler",
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
      "c_90"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
