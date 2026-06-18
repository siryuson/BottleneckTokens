# ChartQA

> Chart visual QA dataset emphasizing numerical and logical reasoning.

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
| Original Name | ChartQA |
| Paper | ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning ([arXiv](https://arxiv.org/abs/2203.10244)) |
| Primary Authors | Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, Enamul Hoque |
| HuggingFace | [ahmed-masry/ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) |
| License | See original dataset |

### Schema
Chart image, question, and answer.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `ChartQA` |
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
| Directory | `MMEB-V2/image-tasks/ChartQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_227`, `negative_candidate_id=c_150`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: How many food item is shown in the bar graph?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 62667,
          "sha1": "ca30ebb41d6a",
          "format": "JPEG",
          "size": [
            850,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_227",
      "text": "14",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_150",
      "text": "11.8",
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
      "c_227"
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
`query_id=q_1`, `positive_candidate_id=c_40`, `negative_candidate_id=c_1541`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the difference in value between Lamb and Corn?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 62667,
          "sha1": "ca30ebb41d6a",
          "format": "JPEG",
          "size": [
            850,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_40",
      "text": "0.57",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1541",
      "text": "Too little",
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
      "c_40"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
