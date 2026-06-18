# GQA

> Visual QA dataset emphasizing compositional reasoning and scene-graph semantics.

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
| Original Name | GQA |
| Paper | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering ([arXiv](https://arxiv.org/abs/1902.09506)) |
| Primary Authors | Drew A. Hudson, Christopher D. Manning |
| HuggingFace | [Graphcore/gqa](https://huggingface.co/datasets/Graphcore/gqa) |
| License | See original dataset |

### Schema
Image, question, answer, and structured semantic information.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `GQA` |
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
| Directory | `MMEB-V2/image-tasks/GQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_5372`, `negative_candidate_id=c_6519`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is this bird called?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 49455,
          "sha1": "a21c0a1610b2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_5372",
      "text": "This is a parrot.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_6519",
      "text": "Yes, there is a clock that is gold.",
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
      "c_5372"
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
`query_id=q_1`, `positive_candidate_id=c_3343`, `negative_candidate_id=c_3782`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What color is the helmet in the middle of the image?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 51950,
          "sha1": "fa6868314a30",
          "format": "JPEG",
          "size": [
            500,
            333
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_3343",
      "text": "The helmet is light blue.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3782",
      "text": "The man is to the right of the American flag.",
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
      "c_3343"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
