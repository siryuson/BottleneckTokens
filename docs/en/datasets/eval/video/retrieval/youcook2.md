# YouCook2

> Large-scale cooking procedure video dataset for process understanding and video retrieval.

| Property | Value |
|------|-----|
| Modality | video |
| Task type | `video_retrieval` |
| Purpose | Evaluation |
| Provenance layers | 3 |

## Original Dataset

### Source Information
| Property | Value |
|------|-----|
| Original name | YouCook2 |
| Paper | Towards Automatic Learning of Procedures from Web Instructional Videos (Zhou et al., 2018) ([arXiv](https://arxiv.org/abs/1703.09788)) |
| HuggingFace | [lmms-lab/YouCook2](https://huggingface.co/datasets/lmms-lab/YouCook2) |
| License | See the original dataset. |

### Schema
- `id`: video segment ID
- `sentence`: query description text

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [lmms-lab/YouCook2](https://huggingface.co/datasets/lmms-lab/YouCook2) |
| Split | `val` |

### Schema
- `id`: video ID
- `sentence`: text query

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/YouCook2/` |
| task_type | `video_retrieval` |
| Evaluation mode | global |

### Lance Layout
- `queries.lance`: text query (`id`, `text`, `images=[]`)
- `candidates.lance`: video-frame candidates (`id`, `text`, `images`)
- `qrels.lance`: sparse relevance annotations (mostly single-positive)

### Samples

The examples below are generated from the current runtime-canonical output backed by `./data/converted`.

- The `text` field shows the current runtime-canonical model-visible text; the parser may reconstruct visual-token layout, option blocks, and trailing newlines from raw Lance fields.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_0`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find a video that demonstrates the following action while making a recipe: pick the ends off the verdalago\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 12854,
            "sha1": "cf0a998989ca",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 1,
            "bytes": 6289,
            "sha1": "87984ac67bd7",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 2,
            "bytes": 5536,
            "sha1": "c30d39f30a0b",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 3,
            "bytes": 5529,
            "sha1": "f58264d9ef99",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 4,
            "bytes": 6196,
            "sha1": "a8cd7fdde05c",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 5,
            "bytes": 5818,
            "sha1": "6c550a496e00",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 6,
            "bytes": 5262,
            "sha1": "5b63adfde988",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 7,
            "bytes": 4750,
            "sha1": "296bc0d5e934",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_0"
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
`query_id=q_1`, `positive_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find a video that demonstrates the following action while making a recipe: combine lemon juice sumac garlic salt and oil in a bowl\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 19055,
            "sha1": "73a444282e3a",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 1,
            "bytes": 6016,
            "sha1": "f2a46df1e640",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 2,
            "bytes": 6867,
            "sha1": "4c60b990b5a3",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 3,
            "bytes": 7251,
            "sha1": "5a096cc81007",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 4,
            "bytes": 7029,
            "sha1": "cd7e17d7d07b",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 5,
            "bytes": 6357,
            "sha1": "2446d1bc4418",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 6,
            "bytes": 6784,
            "sha1": "25fba3a20536",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 7,
            "bytes": 6857,
            "sha1": "0b7c385ad32a",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_1"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
