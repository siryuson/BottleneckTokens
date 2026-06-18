# ScienceQA

> Multimodal QA dataset for science-education scenarios.

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
| Original Name | ScienceQA |
| Paper | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering ([arXiv](https://arxiv.org/abs/2209.09513)) |
| Primary Authors | Pan Lu, Swaroop Mishra, Tang Yao, Daniel Q. Fu, John Tstiatsios, Michihiro Yasunaga, Xinyun Chen, Silvio Savarese, Hannaneh Hajishirzi, Luke Zettlemoyer, Wen-tau Yih |
| HuggingFace | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| License | See original dataset |

### Schema
Question, optional image, options, and answer.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `ScienceQA` |
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
| Directory | `MMEB-V2/image-tasks/ScienceQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_544`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which of the following could Gordon's test show?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 13169,
          "sha1": "a7a4b9a345b3",
          "format": "JPEG",
          "size": [
            302,
            232
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_544",
      "text": "how steady a parachute with a 1 m vent was at 200 km per hour",
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
      "c_544"
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
`query_id=q_1`, `positive_candidate_id=c_296`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the name of the colony shown?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 19696,
          "sha1": "e11723e214c6",
          "format": "JPEG",
          "size": [
            452,
            595
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_296",
      "text": "New Hampshire",
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
      "c_296"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
