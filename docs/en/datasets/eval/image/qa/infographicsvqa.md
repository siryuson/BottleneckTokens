# InfographicsVQA

> Visual QA dataset for infographic scenarios, emphasizing mixed text-visual and layout understanding.

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
| Original Name | InfographicsVQA |
| Paper | InfographicVQA ([arXiv](https://arxiv.org/abs/2104.12756)) |
| Primary Authors | Furkan Biten, Lluis Gomez, Marçal Rusinol, Dimosthenis Karatzas, C. V. Jawahar |
| HuggingFace | [vidore/infovqa_test_subsampled_beir](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir) |
| License | See original dataset |

### Schema
Infographic image, question, and answer set.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `InfographicsVQA` |
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
| Directory | `MMEB-V2/image-tasks/InfographicsVQA/` |
| task_type | `image_question_answer` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1446`, `negative_candidate_id=c_1507`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which social platform has heavy female audience?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 255643,
          "sha1": "c6815eb609f2",
          "format": "JPEG",
          "size": [
            969,
            2221
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1446",
      "text": "pinterest",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1507",
      "text": "runny nose, cough, sore throat",
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
      "c_1446"
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
`query_id=q_1`, `positive_candidate_id=c_1488`, `negative_candidate_id=c_1015`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which three business types is Pinterest good for?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 255643,
          "sha1": "c6815eb609f2",
          "format": "JPEG",
          "size": [
            969,
            2221
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1488",
      "text": "restaurants, interior design, wedding venues",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1015",
      "text": "doers, artists",
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
      "c_1488"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
