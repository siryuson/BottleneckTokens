# VisualNews I2T

> News image-text retrieval subset (image-to-text).

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
| Original Name | VisualNews |
| Paper | Visual News: Benchmark and Challenges in News Image Captioning ([arXiv](https://arxiv.org/abs/2010.03743)) |
| Primary Authors | Fuxiao Liu, Yinghan Wang, Tianlu Wang, Vicente Ordonez |
| HuggingFace | [FuxiaoLiu/visualnews](https://huggingface.co/datasets/FuxiaoLiu/visualnews) |
| License | See original dataset |

### Schema
News image, news title/description, and metadata.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `VisualNews_i2t` |
| Split | `test` |
| Eval Parser | `image_i2t` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/VisualNews_i2t/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_8351`, `negative_candidate_id=c_15164`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nFind a caption for the news in the given photo.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 11525,
          "sha1": "a9c09e155019",
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
      "id": "c_8351",
      "text": "Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_15164",
      "text": "Some of the 112 portraits already completed.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_8351"
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
`query_id=q_1`, `positive_candidate_id=c_310`, `negative_candidate_id=c_15694`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nFind a caption for the news in the given photo.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 20929,
          "sha1": "3ae38f6edd64",
          "format": "JPEG",
          "size": [
            398,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_310",
      "text": "A Maryland State Police cruiser sits at a blocked southbound entrance on the BaltimoreWashington Parkway that accesses Fort Meade.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_15694",
      "text": "Syracuse Orange head coach Jim Boeheim looks on during the first half against the Georgetown Hoyas at Verizon Center.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_310"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
