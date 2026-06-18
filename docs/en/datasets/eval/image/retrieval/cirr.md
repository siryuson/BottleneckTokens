# CIRR

> Composed image retrieval dataset where queries combine a reference image and text modification instructions.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_retrieval` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset
### Source Info
| Property | Value |
|------|-----|
| Original Name | CIRR |
| Paper | CIRR: Composed Image Retrieval through Referring Relationship ([arXiv](https://arxiv.org/abs/2109.14024)) |
| Primary Authors | Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, Rogerio Feris |
| HuggingFace | [xswu/cirr](https://huggingface.co/datasets/xswu/cirr) |
| License | See original dataset |

### Schema
Query image, text modification description, and candidate image set.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `CIRR` |
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
| Directory | `MMEB-V2/image-tasks/CIRR/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_105`, `negative_candidate_id=c_1979`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nGiven an image, find a similar everyday image with the described changes: Show three bottles of soft drink.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 14234,
          "sha1": "370b012a1ac0",
          "format": "JPEG",
          "size": [
            256,
            416
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_105",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 22491,
            "sha1": "cc7772155abd",
            "format": "JPEG",
            "size": [
              256,
              340
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1979",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 14445,
            "sha1": "5b052143957d",
            "format": "JPEG",
            "size": [
              256,
              384
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
      "c_105"
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
`query_id=q_1`, `positive_candidate_id=c_1372`, `negative_candidate_id=c_150`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nGiven an image, find a similar everyday image with the described changes: Fewer paper towels per pack.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 15115,
          "sha1": "10d01f6ce01b",
          "format": "JPEG",
          "size": [
            384,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1372",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 19802,
            "sha1": "400c566d0533",
            "format": "JPEG",
            "size": [
              256,
              512
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_150",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9566,
            "sha1": "56a9b2bd8605",
            "format": "JPEG",
            "size": [
              341,
              256
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
      "c_1372"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
