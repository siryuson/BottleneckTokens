# DiDeMo

> Fine-grained description dataset for natural-language moment localization and video retrieval.

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
| Original name | DiDeMo |
| Paper | Localizing Moments in Video with Natural Language (Hendricks et al., 2017) ([arXiv](https://arxiv.org/abs/1708.01641)) |
| HuggingFace | [siyrus/BToks-DiDeMo](https://huggingface.co/datasets/siyrus/BToks-DiDeMo) |
| License | See the original dataset. |

### Schema
- `video`/`video_id`: video identifier
- `caption`: query description text

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-DiDeMo](https://huggingface.co/datasets/siyrus/BToks-DiDeMo) |
| Split | `test` |

### Schema
- `video`/`video_id`: video ID
- `caption`: text query

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/DiDeMo/` |
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
    "text": "Find a video that includes the following described scenes: we pan right off the mural and on to a building. the view changes from artwork to a building in this shot. the screen moves to something dark the camera pans from the end of the mural to another structure and then a building.\n",
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
            "bytes": 33910,
            "sha1": "81da46599371",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 17108,
            "sha1": "5c5428d9154c",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 15906,
            "sha1": "7f415fb148e6",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 7679,
            "sha1": "f38ac474f70e",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 9994,
            "sha1": "9c056fc80f06",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 4142,
            "sha1": "37be8c16cba8",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 7883,
            "sha1": "fc30e40b5131",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 4048,
            "sha1": "28206e905e4b",
            "format": "JPEG",
            "size": [
              640,
              480
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
    "text": "Find a video that includes the following described scenes: the sunset first comes into view. nothing but ocean in these clips first time sun glares on water first time.\n",
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
            "bytes": 23975,
            "sha1": "7de99ba4b5a1",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 8643,
            "sha1": "f3367c726be6",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 6991,
            "sha1": "c56362886634",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 7573,
            "sha1": "09025a0b4d47",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 7139,
            "sha1": "9e2078239104",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 8037,
            "sha1": "8b8c730f03ca",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 9635,
            "sha1": "8609d5301c4e",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 11791,
            "sha1": "e6b70a583563",
            "format": "JPEG",
            "size": [
              640,
              480
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
