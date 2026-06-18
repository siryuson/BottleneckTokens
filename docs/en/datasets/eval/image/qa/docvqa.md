# DocVQA

> Document image QA benchmark focused on page text and layout understanding.

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
| Original Name | DocVQA (Task 1: VQA on Document Images) |
| Paper | ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction ([arXiv](https://arxiv.org/abs/1911.12482)) |
| Primary Authors | Minesh Mathew, Dimosthenis Karatzas, C. V. Jawahar |
| HuggingFace | [cmarkea/docvqa](https://huggingface.co/datasets/cmarkea/docvqa) |
| License | See original dataset |

### Schema
Document image, question text, and answer list.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `DocVQA` |
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
| Directory | `MMEB-V2/image-tasks/DocVQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_269`, `negative_candidate_id=c_721`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the ‘actual’ value per 1000, during the year 1975?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 276397,
          "sha1": "9ee7cdea0973",
          "format": "JPEG",
          "size": [
            2257,
            1764
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_269",
      "text": "0.28",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_721",
      "text": "1992",
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
      "c_269"
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
`query_id=q_1`, `positive_candidate_id=c_4388`, `negative_candidate_id=c_3025`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is name of university?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 333318,
          "sha1": "e1daaf1bed29",
          "format": "JPEG",
          "size": [
            808,
            1077
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4388",
      "text": "university of california",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3025",
      "text": "Postcards",
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
      "c_4388"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
