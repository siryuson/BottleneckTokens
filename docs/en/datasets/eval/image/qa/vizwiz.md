# VizWiz

> Real-world visual QA dataset built from images captured by blind/low-vision users.

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
| Original Name | VizWiz-VQA |
| Paper | VizWiz Grand Challenge: Answering Visual Questions from Blind People ([arXiv](https://arxiv.org/abs/1802.08218)) |
| Primary Authors | Danna Gurari, Qing Li, Abigale Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, Jeffrey P. Bigham |
| HuggingFace | [lmms-lab/VizWiz-VQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA) |
| License | See original dataset |

### Schema
Image, question, answer annotations, and answerability information.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `VizWiz` |
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
| Directory | `MMEB-V2/image-tasks/VizWiz/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_883`, `negative_candidate_id=c_1276`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Can you tell me what this medicine is please?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 4314,
          "sha1": "3cf0301aced8",
          "format": "JPEG",
          "size": [
            121,
            162
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_883",
      "text": "night time",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1276",
      "text": "tonic wine",
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
      "c_883"
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
`query_id=q_1`, `positive_candidate_id=c_471`, `negative_candidate_id=c_67`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the title of this book?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 29759,
          "sha1": "8ed3f2e024f8",
          "format": "JPEG",
          "size": [
            484,
            648
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_471",
      "text": "dog years",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_67",
      "text": "advanced boot options",
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
      "c_471"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
