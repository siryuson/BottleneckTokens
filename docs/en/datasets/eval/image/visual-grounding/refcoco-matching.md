# RefCOCO-Matching

> RefCOCO matching subset where the goal is to match referring expressions among candidate objects.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_visual_grounding` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset
### Source Info
| Property | Value |
|------|-----|
| Original Name | RefCOCO-Matching |
| Paper | Generation and Comprehension of Unambiguous Object Descriptions ([arXiv](https://arxiv.org/abs/1511.02283)) |
| Primary Authors | Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, Kevin Murphy |
| HuggingFace | [jxu124/refcoco-benchmark](https://huggingface.co/datasets/jxu124/refcoco-benchmark) |
| License | See original dataset |

### Schema
Image, referring expression, and candidate target matching (multiple targets in same image).

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `RefCOCO-Matching` |
| Split | `test` |
| Eval Parser | `image_i2i_vg` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/RefCOCO-Matching/` |
| task_type | `image_visual_grounding` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_561`, `negative_candidate_id=c_562`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"bowl behind the others can only see part\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 43142,
          "sha1": "6f679b125e46",
          "format": "JPEG",
          "size": [
            640,
            428
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_561",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"Dish in top right corner\"\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 43142,
            "sha1": "6f679b125e46",
            "format": "JPEG",
            "size": [
              640,
              428
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_562",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"front bowl w/carrots in it\"\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 43142,
            "sha1": "6f679b125e46",
            "format": "JPEG",
            "size": [
              640,
              428
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_561"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```

#### Example 2
`query_id=q_1`, `positive_candidate_id=c_1892`, `negative_candidate_id=c_1891`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"little girl\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 67273,
          "sha1": "91db79adf516",
          "format": "JPEG",
          "size": [
            640,
            481
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1892",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"pink\"\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 67273,
            "sha1": "91db79adf516",
            "format": "JPEG",
            "size": [
              640,
              481
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1891",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"green woman\"\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 67273,
            "sha1": "91db79adf516",
            "format": "JPEG",
            "size": [
              640,
              481
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_1892"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
