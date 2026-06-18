# Visual7W

> Visual QA dataset based on images with multiple QA formats.

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
| Original Name | Visual7W |
| Paper | Visual7W: Grounded Question Answering in Images ([arXiv](https://arxiv.org/abs/1511.07332)) |
| Primary Authors | Yuke Zhu, Oliver Groth, Michael Bernstein, Li Fei-Fei |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| License | See original dataset |

### Schema
Image, question, candidate answers, and annotated answer.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `Visual7W` |
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
| Directory | `MMEB-V2/image-tasks/Visual7W/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1344`, `negative_candidate_id=c_3163`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What color is the sidewalk?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 106591,
          "sha1": "cc861c687eff",
          "format": "JPEG",
          "size": [
            800,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1344",
      "text": "Gray.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3163",
      "text": "To receive orders.",
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
      "c_1344"
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
`query_id=q_1`, `positive_candidate_id=c_639`, `negative_candidate_id=c_2189`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Where is the man sitting?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 60190,
          "sha1": "e698932ba877",
          "format": "JPEG",
          "size": [
            800,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_639",
      "text": "At the computer.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2189",
      "text": "On the street.",
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
      "c_639"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
