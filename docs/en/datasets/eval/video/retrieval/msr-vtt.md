# MSR-VTT

> Large open-domain video-text retrieval benchmark covering diverse scenes and descriptions.

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
| Original name | MSR-VTT |
| Paper | MSR-VTT: A Large Video Description Dataset for Bridging Video and Language (Xu et al., 2016) ([CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html)) |
| HuggingFace | [siyrus/BToks-MSR-VTT](https://huggingface.co/datasets/siyrus/BToks-MSR-VTT) |
| License | See the original dataset. |

### Schema
- `video_id`: video identifier
- `caption`: query description text

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MSR-VTT](https://huggingface.co/datasets/siyrus/BToks-MSR-VTT) |
| Split | `test` (subset: `test_1k`) |

### Schema
- `video_id`: video ID
- `caption`: text query

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/MSR-VTT/` |
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
    "text": "Find a video that contains the following visual content: a woman creating a fondant baby and flower\n",
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
            "bytes": 12126,
            "sha1": "29da827bf5c8",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 1,
            "bytes": 15466,
            "sha1": "615bd8a88625",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 2,
            "bytes": 17522,
            "sha1": "f1b9c19bb4a7",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 3,
            "bytes": 11633,
            "sha1": "1a7afff0dbfd",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 4,
            "bytes": 9139,
            "sha1": "12e490bb9474",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 5,
            "bytes": 10201,
            "sha1": "dbfc60c73355",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 6,
            "bytes": 10398,
            "sha1": "be2e806d7925",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 7,
            "bytes": 9185,
            "sha1": "b5450ecc0e11",
            "format": "JPEG",
            "size": [
              298,
              224
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
    "text": "Find a video that contains the following visual content: baseball player hits ball\n",
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
            "bytes": 11502,
            "sha1": "c27da12ca45f",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 1,
            "bytes": 21016,
            "sha1": "433c4255cf54",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 2,
            "bytes": 6725,
            "sha1": "680defc550b8",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 3,
            "bytes": 13143,
            "sha1": "2d8a5a253696",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 4,
            "bytes": 12462,
            "sha1": "df98477abfd7",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 5,
            "bytes": 10881,
            "sha1": "d78a3239b244",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 6,
            "bytes": 14361,
            "sha1": "e70a12598135",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 7,
            "bytes": 11026,
            "sha1": "a22757d2eb0a",
            "format": "JPEG",
            "size": [
              298,
              224
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
