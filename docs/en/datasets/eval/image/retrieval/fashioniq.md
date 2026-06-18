# FashionIQ

> Fashion image retrieval dataset based on natural-language feedback.

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
| Original Name | Fashion IQ |
| Paper | Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback ([arXiv](https://arxiv.org/abs/1905.12794)) |
| Primary Authors | Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, Rogerio Feris |
| HuggingFace | [McAuley-Lab/Amazon-Customer-Reviews-Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Customer-Reviews-Dataset) |
| License | See original dataset |

### Schema
Reference image, textual feedback, and target image.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `FashionIQ` |
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
| Directory | `MMEB-V2/image-tasks/FashionIQ/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_2930`, `negative_candidate_id=c_3283`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nFind an image to match the fashion image and style note: Is shiny and silver with shorter sleeves and fit and flare.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 9165,
          "sha1": "9c48d6effccd",
          "format": "JPEG",
          "size": [
            256,
            332
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2930",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 7212,
            "sha1": "01a3eb6f3c3d",
            "format": "JPEG",
            "size": [
              256,
              332
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3283",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 8242,
            "sha1": "a2a5b55d1d97",
            "format": "JPEG",
            "size": [
              256,
              356
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
      "c_2930"
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
`query_id=q_1`, `positive_candidate_id=c_4212`, `negative_candidate_id=c_2044`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nFind an image to match the fashion image and style note: Is grey with black design and is a light printed short dress.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 8965,
          "sha1": "faa293c84050",
          "format": "JPEG",
          "size": [
            256,
            306
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4212",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 8777,
            "sha1": "4126fa19f31a",
            "format": "JPEG",
            "size": [
              256,
              320
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2044",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9211,
            "sha1": "fbb11f14fad3",
            "format": "JPEG",
            "size": [
              256,
              332
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
      "c_4212"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
