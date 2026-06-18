# MSCOCO (Visual Grounding)

> COCO-based visual grounding/matching evaluation subset.

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
| Original Name | MS COCO |
| Paper | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| Primary Authors | Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| License | See original dataset |

### Schema
Image and caption text, paired with candidate images for visual-grounding-style retrieval.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `MSCOCO` |
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
| Directory | `MMEB-V2/image-tasks/MSCOCO/` |
| task_type | `image_visual_grounding` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1609`, `negative_candidate_id=c_2343`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nCrop the image to to isolate the object labeled as \"dog\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 170295,
          "sha1": "714a7089e714",
          "format": "JPEG",
          "size": [
            529,
            640
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1609",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 1520,
            "sha1": "545be211d39a",
            "format": "JPEG",
            "size": [
              38,
              29
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2343",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 11951,
            "sha1": "c75eceacf713",
            "format": "JPEG",
            "size": [
              238,
              186
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
      "c_1609"
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
`query_id=q_1`, `positive_candidate_id=c_4033`, `negative_candidate_id=c_1763`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nCrop the image to to isolate the object labeled as \"dog\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 89601,
          "sha1": "739294454342",
          "format": "JPEG",
          "size": [
            640,
            480
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4033",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 17549,
            "sha1": "4d031657f81b",
            "format": "JPEG",
            "size": [
              152,
              280
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1763",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 1267,
            "sha1": "4ea6d84fd2b3",
            "format": "JPEG",
            "size": [
              22,
              45
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
      "c_4033"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
