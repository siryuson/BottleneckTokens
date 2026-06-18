# TextVQA

> Visual question answering dataset focused on reading and reasoning over text in images.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_question_answer` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset
### Source Info
| Property | Value |
|------|-----|
| Original Name | TextVQA |
| Paper | Towards VQA Models That Can Read ([arXiv](https://arxiv.org/abs/1904.08920)) |
| Primary Authors | Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach |
| HuggingFace | [lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) |
| License | See original dataset |

### Schema
Image, question, OCR text, and answer.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `TextVQA` |
| Split | `test` |
| Eval Parser | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/TextVQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1209`, `negative_candidate_id=c_2583`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: what is the brand of this camera?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 47139,
          "sha1": "5e72db6ed822",
          "format": "JPEG",
          "size": [
            1024,
            664
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1209",
      "text": "dakota",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2583",
      "text": "rose",
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
      "c_1209"
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
`query_id=q_1`, `positive_candidate_id=c_1148`, `negative_candidate_id=c_1263`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: what does the small white text spell?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 80334,
          "sha1": "aaaea1a3a7e2",
          "format": "JPEG",
          "size": [
            1024,
            683
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1148",
      "text": "copenhagen",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1263",
      "text": "dino buzzati",
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
      "c_1148"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
