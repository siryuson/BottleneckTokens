# MSVD

> Early classic video description and retrieval dataset built from parallel multi-caption annotations.

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
| Original name | MSVD (Microsoft Video Description Corpus) |
| Paper | Collecting Highly Parallel Data for Paraphrase Evaluation (Chen & Dolan, 2011) ([ACL Anthology](https://aclanthology.org/P11-1020/)) |
| HuggingFace | [siyrus/BToks-MSVD](https://huggingface.co/datasets/siyrus/BToks-MSVD) |
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
| HuggingFace | [siyrus/BToks-MSVD](https://huggingface.co/datasets/siyrus/BToks-MSVD) |
| Split | `test` |

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
| Directory | `MMEB-V2/video-tasks/MSVD/` |
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
    "text": "Find the video snippet that corresponds to the given summary: two young men are playing table tennis\n",
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
            "bytes": 14503,
            "sha1": "5fe3ea8a459a",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 8562,
            "sha1": "5b7bc686ad4d",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 7795,
            "sha1": "5b8f98ba1a69",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 8014,
            "sha1": "70fe6028dba1",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 8070,
            "sha1": "e0ccb1d936ec",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 7932,
            "sha1": "c81173ebac9d",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 8011,
            "sha1": "503a09304f44",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 6852,
            "sha1": "64009cc0eb07",
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
    "text": "Find the video snippet that corresponds to the given summary: a man rides his horse in the dessert\n",
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
            "bytes": 22042,
            "sha1": "d82484bcfa2c",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 9181,
            "sha1": "4441cf9c6211",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 7925,
            "sha1": "7826eead9a53",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 8413,
            "sha1": "9ceff6ed8b17",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 7702,
            "sha1": "a102816c0dd2",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 8104,
            "sha1": "11446652c067",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 7615,
            "sha1": "008b5d5e7175",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 8532,
            "sha1": "02789bd1d01c",
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
