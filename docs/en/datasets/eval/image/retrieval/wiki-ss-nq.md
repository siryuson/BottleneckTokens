# Wiki-SS-NQ

> Text-to-image retrieval subset based on Wikipedia screenshot corpus.

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
| Original Name | Wiki-SS-NQ |
| Paper | Unifying Multimodal Retrieval via Document Screenshot Embedding ([arXiv](https://arxiv.org/abs/2406.11251)) |
| Primary Authors | Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, Jimmy Lin |
| HuggingFace | [Tevatron/wiki-ss-nq](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) |
| License | See original dataset |

### Schema
Natural-language question, page screenshot candidates, and relevance annotations.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `Wiki-SS-NQ` |
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
| Directory | `MMEB-V2/image-tasks/Wiki-SS-NQ/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1482`, `negative_candidate_id=c_2775`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find the document image that can answer the given query: who needs to know about the jet stream\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1482",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 186874,
            "sha1": "256d45696c03",
            "format": "JPEG",
            "size": [
              980,
              980
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2775",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 81037,
            "sha1": "197e031f464b",
            "format": "JPEG",
            "size": [
              980,
              980
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
      "c_1482"
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
`query_id=q_1`, `positive_candidate_id=c_7031`, `negative_candidate_id=c_929`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find the document image that can answer the given query: who founded the red cross to relieve the suffering of the war wounded\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_7031",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 156902,
            "sha1": "87a79e84422d",
            "format": "JPEG",
            "size": [
              980,
              980
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_929",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 217585,
            "sha1": "acc47505b96d",
            "format": "JPEG",
            "size": [
              980,
              980
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
      "c_7031"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
