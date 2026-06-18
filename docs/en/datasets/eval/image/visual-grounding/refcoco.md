# RefCOCO

> Referring-expression understanding dataset for target localization and image-text matching evaluation.

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
| Original Name | RefCOCO |
| Paper | Generation and Comprehension of Unambiguous Object Descriptions ([arXiv](https://arxiv.org/abs/1511.02283)) |
| Primary Authors | Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, Kevin Murphy |
| HuggingFace | [jxu124/refcoco-benchmark](https://huggingface.co/datasets/jxu124/refcoco-benchmark) |
| License | See original dataset |

### Schema
Image, referring expression, and target region annotations.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `RefCOCO` |
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
| Directory | `MMEB-V2/image-tasks/RefCOCO/` |
| task_type | `image_visual_grounding` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_2033`, `negative_candidate_id=c_4986`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions. \"bowl behind the others can only see part\"\n",
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
      "id": "c_2033",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 2963,
            "sha1": "5b9ac77aa5f4",
            "format": "JPEG",
            "size": [
              172,
              116
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_4986",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 2782,
            "sha1": "416a76bf5bb8",
            "format": "JPEG",
            "size": [
              119,
              123
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
      "c_2033"
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
`query_id=q_1`, `positive_candidate_id=c_8220`, `negative_candidate_id=c_4937`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions. \"front bowl w/carrots in it\"\n",
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
      "id": "c_8220",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 26362,
            "sha1": "837cdcee2d8e",
            "format": "JPEG",
            "size": [
              455,
              284
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_4937",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 29450,
            "sha1": "09484421a47b",
            "format": "JPEG",
            "size": [
              235,
              295
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
      "c_8220"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
