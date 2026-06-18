# WebQA

> Text-to-image retrieval subset derived from multi-hop multimodal QA.

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
| Original Name | WebQA |
| Paper | WebQA: Multihop and Multimodal QA ([arXiv](https://arxiv.org/abs/2109.00590)) |
| Primary Authors | Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, Yonatan Bisk |
| HuggingFace | [Stanford/web_questions](https://huggingface.co/datasets/Stanford/web_questions) |
| License | See original dataset |

### Schema
Question, evidence text, and candidate image set.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `WebQA` |
| Split | `test` |
| Eval Parser | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/WebQA/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_477`, `negative_candidate_id=c_814`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals in a cup shape?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_477",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: 2020-05-08 15 17 05 Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 19200,
            "sha1": "a1b41cccf4f8",
            "format": "JPEG",
            "size": [
              341,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_814",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Louvre - French sculptures - Room 25 - 03.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 15435,
            "sha1": "eebff019af25",
            "format": "JPEG",
            "size": [
              386,
              256
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
      "c_477"
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
`query_id=q_1`, `positive_candidate_id=c_2136`, `negative_candidate_id=c_1188`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find a Wikipedia image that answers this question: What water-related object is sitting in front of the Torre del Reloj?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2136",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Torre del Reloj de la Plaza Colón de Antofagasta (1).\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 18024,
            "sha1": "b7dcd3976d9e",
            "format": "JPEG",
            "size": [
              256,
              301
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1188",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Isaac White Stanford University Cardinals versus University of San Diego Torendos , Mens Basketball, Chase Center, San Francisco, 2019 Al Attles Classic, Basketball Hall of Fame , CA,.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 15761,
            "sha1": "1e65c6a0cd0c",
            "format": "JPEG",
            "size": [
              256,
              313
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
      "c_2136"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
