# VATEX

> High-quality multilingual video description and retrieval dataset for cross-lingual video semantic matching.

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
| Original name | VATEX |
| Paper | VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research (Wang et al., 2019) ([arXiv](https://arxiv.org/abs/1904.03493)) |
| HuggingFace | [siyrus/BToks-VATEX](https://huggingface.co/datasets/siyrus/BToks-VATEX) |
| License | See the original dataset. |

### Schema
- `videoID`: video identifier
- `enCap`: English caption list

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-VATEX](https://huggingface.co/datasets/siyrus/BToks-VATEX) |
| Split | `test` |

### Schema
- `videoID`: video ID
- `enCap`: text querycandidatedescription

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/VATEX/` |
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
    "text": "Select a video that fits the description provided: A young man is showing the polish, water. old soft cloth and brush needed to polish shoes with.\n",
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
            "bytes": 1618,
            "sha1": "ac7db4321cb4",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 1,
            "bytes": 8870,
            "sha1": "18e8b5a62201",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 2,
            "bytes": 8627,
            "sha1": "cd3f1e2d26e9",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 3,
            "bytes": 9246,
            "sha1": "774a103a7e7f",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 4,
            "bytes": 11619,
            "sha1": "1d3cc02f8f45",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 5,
            "bytes": 12574,
            "sha1": "96c512cc1e1a",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 6,
            "bytes": 15885,
            "sha1": "481329fdfbf2",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 7,
            "bytes": 13125,
            "sha1": "406ec3adc166",
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
    "text": "Select a video that fits the description provided: A woman explains how to sew a beanie as she sews a beanie in her hands.\n",
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
            "bytes": 5622,
            "sha1": "3e26306d44b5",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 17016,
            "sha1": "918a9186de22",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 17997,
            "sha1": "385cd73ad356",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 16883,
            "sha1": "1e4419fc592e",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 18213,
            "sha1": "31deb5f5a3e6",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 21265,
            "sha1": "f95c38cfa190",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 22182,
            "sha1": "f48613d48642",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 34561,
            "sha1": "e157e9ce164d",
            "format": "JPEG",
            "size": [
              1280,
              720
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
