# OK-VQA

> Visual QA dataset requiring external knowledge.

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
| Original Name | OK-VQA |
| Paper | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge ([arXiv](https://arxiv.org/abs/1906.00067)) |
| Primary Authors | Kenneth Marino, Mohammad Rastegari, Ali Farhadi, Roozbeh Mottaghi |
| HuggingFace | [HuggingFaceM4/OK-VQA](https://huggingface.co/datasets/HuggingFaceM4/OK-VQA) |
| License | See original dataset |

### Schema
Image, question, and answer set.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `OK-VQA` |
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
| Directory | `MMEB-V2/image-tasks/OK-VQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1830`, `negative_candidate_id=c_319`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What sport can you use this for?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 67015,
          "sha1": "f4094f7a48a7",
          "format": "JPEG",
          "size": [
            640,
            480
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1830",
      "text": "race",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_319",
      "text": "bake it",
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
      "c_1830"
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
`query_id=q_1`, `positive_candidate_id=c_2429`, `negative_candidate_id=c_1620`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Name the type of plant this is?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 50950,
          "sha1": "0f7fe258734f",
          "format": "JPEG",
          "size": [
            640,
            437
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2429",
      "text": "vine",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1620",
      "text": "octogon",
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
      "c_2429"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
