# ImageNet-R

> Out-of-domain rendered-style ImageNet subset (painting/cartoon, etc.) for evaluating out-of-distribution robustness.

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
| Original Name | ImageNet-R |
| Paper | Are we done with ImageNet? ([arXiv](https://arxiv.org/abs/2006.07159)) |
| Primary Authors | Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Austin Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, Justin Gilmer |
| HuggingFace | [yerevann/imagenet-r](https://huggingface.co/datasets/yerevann/imagenet-r) |
| License | See original dataset |

### Schema
Image + ImageNet class label (rendered style-shift examples).

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `ImageNet-R` |
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
| Directory | `MMEB-V2/image-tasks/ImageNet-R/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_67`

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
          "bytes": 16148,
          "sha1": "6b287ddef517",
          "format": "JPEG",
          "size": [
            450,
            470
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_67",
      "text": "flute",
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
      "c_67"
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
`query_id=q_1`, `positive_candidate_id=c_67`

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
          "bytes": 4072,
          "sha1": "a199cbd4c74b",
          "format": "JPEG",
          "size": [
            257,
            194
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_67",
      "text": "flute",
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
      "c_67"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
