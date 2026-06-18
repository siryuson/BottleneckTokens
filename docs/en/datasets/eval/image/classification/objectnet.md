# ObjectNet

> Object recognition dataset targeting real-world capture-condition shifts, emphasizing robustness to viewpoint and background changes.

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
| Original Name | ObjectNet |
| Paper | ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models ([arXiv](https://arxiv.org/abs/1909.13000)) |
| Primary Authors | Andrey Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Joshua Tenenbaum, Boris Katz |
| HuggingFace | [blanchon/ObjectNet](https://huggingface.co/datasets/blanchon/ObjectNet) |
| License | See original dataset |

### Schema
Image + object category label.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `ObjectNet` |
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
| Directory | `MMEB-V2/image-tasks/ObjectNet/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_98`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nIdentify the object shown in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 30176,
          "sha1": "9c2673f0852b",
          "format": "JPEG",
          "size": [
            964,
            544
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_98",
      "text": "tray",
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
      "c_98"
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
`query_id=q_1`, `positive_candidate_id=c_52`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nIdentify the object shown in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 137440,
          "sha1": "d3375b364678",
          "format": "JPEG",
          "size": [
            1084,
            1924
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_52",
      "text": "mug",
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
      "c_52"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
