# A-OKVQA

> Open-ended visual QA dataset combining commonsense reasoning with visual evidence.

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
| Original Name | A-OKVQA |
| Paper | A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge ([arXiv](https://arxiv.org/abs/2206.01718)) |
| Primary Authors | Dustin Schwenk, Jack Hessel, Ari Holtzman, Ronan Le Bras, Chandra Bhagavatula, Yejin Choi |
| HuggingFace | [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) |
| License | See original dataset |

### Schema
Image, question, direct answer, rationale, and related fields.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `A-OKVQA` |
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
| Directory | `MMEB-V2/image-tasks/A-OKVQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_140`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is in the motorcyclist's mouth?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 54735,
          "sha1": "a6288593a2ab",
          "format": "JPEG",
          "size": [
            640,
            569
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_140",
      "text": "cigarette",
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
      "c_140"
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
`query_id=q_1`, `positive_candidate_id=c_774`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which number birthday is probably being celebrated?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 70271,
          "sha1": "3d8607931911",
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
      "id": "c_774",
      "text": "thirty",
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
      "c_774"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
