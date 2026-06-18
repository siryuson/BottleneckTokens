# Place365

> Scene recognition benchmark for evaluating model discrimination of geographic and scene semantics.

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
| Original Name | Places365 |
| Paper | Places: A 10 million Image Database for Scene Recognition ([arXiv](https://arxiv.org/abs/1610.02055)) |
| Primary Authors | Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, Antonio Torralba |
| HuggingFace | [Eth1et/places365](https://huggingface.co/datasets/Eth1et/places365) |
| License | See original dataset |

### Schema
Image + scene label.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `Place365` |
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
| Directory | `MMEB-V2/image-tasks/Place365/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_55`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nIdentify the scene shown in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 12954,
          "sha1": "9f46d14d531a",
          "format": "JPEG",
          "size": [
            256,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_55",
      "text": "berth",
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
      "c_55"
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
`query_id=q_1`, `positive_candidate_id=c_262`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nIdentify the scene shown in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 16390,
          "sha1": "cfad8b3516e1",
          "format": "JPEG",
          "size": [
            256,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_262",
      "text": "pharmacy",
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
      "c_262"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
