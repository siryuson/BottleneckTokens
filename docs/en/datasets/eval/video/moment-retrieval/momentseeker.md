# MomentSeeker

> Task-oriented long-video moment retrieval benchmark supporting text, image, and video-conditioned queries.

| Property | Value |
|------|-----|
| Modality | video |
| Task type | `video_moment_retrieval` |
| Purpose | Evaluation |
| Provenance layers | 3 |

## Original Dataset

### Source Information
| Property | Value |
|------|-----|
| Original name | MomentSeeker |
| Paper | MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval (Yuan et al., 2025) ([OpenReview](https://openreview.net/forum?id=gKkA9Oc13m)) |
| HuggingFace | [siyrus/BToks-MomentSeeker](https://huggingface.co/datasets/siyrus/BToks-MomentSeeker) |
| License | See the original dataset. |

### Schema
- `query`: text query
- `input_frames`: query condition (may be a video segment, image, or empty)
- `positive_frames`: positive segment information
- `negative_frames`: negative segment information

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MomentSeeker](https://huggingface.co/datasets/siyrus/BToks-MomentSeeker) |
| Split | `test` |

### Schema
- `query`: text query
- `input_frames`: query-condition path
- `positive_frames` / `negative_frames`: candidate segment set

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/MomentSeeker/` |
| task_type | `video_moment_retrieval` |
| Evaluation mode | local |

### Lance Layout
- `queries.lance`: mixed query inputs (`id`, `text`, `images`) covering text-only, image-conditioned, and video-conditioned prompts
- `candidates.lance`: candidate video clips (`id`, `text`, `images`)
- `qrels.lance`: exhaustive relevance labels (`mode=exhaustive`)

### Baseline run and runtime notes

- `queries.lance` is mixed-modality on the query side: the same MomentSeeker subset contains pure-text queries, image-conditioned queries, and video-conditioned queries.
- `candidates.lance` always represents candidate video clips, so query-side and candidate-side encoding cost should be considered separately.
- For a clean single-subset baseline, run `scripts/eval.py` with `--log-file output/momentseeker.log eval.datasets=[MomentSeeker] eval.show_progress=false --verbose`.
- Under the block-cyclic protocol, DEBUG `retrieval_protocol` logs expose shard, gather, trim, and reorder state when deeper diagnosis is needed.
- After gather, query / candidate embeddings must be reordered back to the original global sample order using sample indices so qrels / ids stay strictly aligned. If ordering drifts, treat it as a protocol bug rather than acceptable metric noise.

### Samples

The examples below are generated from the current runtime-canonical output backed by `./data/converted`.

- The `text` field shows the current runtime-canonical model-visible text; the parser may reconstruct visual-token layout, option blocks, and trailing newlines from raw Lance fields.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_0`, `negative_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nFind the clip that corresponds to the given sentence and video segment: What happened to this window afterward?\n",
    "images": {
      "count": 8,
      "items": [
        {
          "index": 0,
          "bytes": 11240,
          "sha1": "36688d6f4493",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 1,
          "bytes": 16099,
          "sha1": "fce8e7c795b2",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 2,
          "bytes": 17521,
          "sha1": "19b21ab5d7cc",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 3,
          "bytes": 20722,
          "sha1": "94e1c0b72986",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 4,
          "bytes": 24712,
          "sha1": "279774cd01a4",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 5,
          "bytes": 18308,
          "sha1": "17458a2e2ef3",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 6,
          "bytes": 13775,
          "sha1": "159155d0d7e2",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 7,
          "bytes": 11614,
          "sha1": "b3843aadcd19",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 17764,
            "sha1": "da2743bc4e5e",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 23837,
            "sha1": "5d49baa1a50c",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 30725,
            "sha1": "0592e00cf29d",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 23733,
            "sha1": "8279cd30ecff",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 15708,
            "sha1": "cd98c0700b7f",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 16465,
            "sha1": "530e3921c9c8",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 29410,
            "sha1": "68834ffebd5b",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 25601,
            "sha1": "f0e016457a82",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 29798,
            "sha1": "31562ea14ce5",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 36062,
            "sha1": "d76cf2d98b3c",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 48848,
            "sha1": "9ef5de55919f",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 29880,
            "sha1": "d5a3f30a5b64",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 24289,
            "sha1": "0bb05bbb832a",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 17050,
            "sha1": "0626c5e1c8bb",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 15834,
            "sha1": "f153eabe4fcd",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 9626,
            "sha1": "0df4be726cb2",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_0"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```

#### Example 2
`query_id=q_533`, `positive_candidate_id=c_5425`, `negative_candidate_id=c_5426`

```json
{
  "queries.lance": {
    "id": "q_533",
    "text": "Find the clip that corresponds to the given text: When the man with the UNIQLO logo and the black woman in pink clothes took a selfie together, who held the phone?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_5425",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 63593,
            "sha1": "ba128193494c",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 77828,
            "sha1": "17be7b5918f4",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 92079,
            "sha1": "3bccb39af4e0",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 92242,
            "sha1": "a5800676f6d2",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 69607,
            "sha1": "3d846db5bddc",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 53883,
            "sha1": "94938ec722e1",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 47652,
            "sha1": "2f6df7993808",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 42285,
            "sha1": "8d76dcef28db",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_5426",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 72281,
            "sha1": "3e56d305a755",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 89074,
            "sha1": "91499c60429f",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 110251,
            "sha1": "d2992e6646ca",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 75369,
            "sha1": "e2fa02804def",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 65617,
            "sha1": "598f86adcfb0",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 47915,
            "sha1": "182b08068a5a",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 41091,
            "sha1": "f277bb994593",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 36082,
            "sha1": "b71faa8c572f",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_533",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_5425"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```

#### Example 3
`query_id=q_1200`, `positive_candidate_id=c_12139`, `negative_candidate_id=c_12140`

```json
{
  "queries.lance": {
    "id": "q_1200",
    "text": "<|image_pad|>\nSelect the video clip that aligns with the given text and image: What am I holding in my left hand in this picture?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 1252417,
          "sha1": "dc942030b194",
          "format": "JPEG",
          "size": [
            2560,
            1920
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_12139",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 316835,
            "sha1": "d5dcaf8eeafc",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 1,
            "bytes": 431783,
            "sha1": "f1e7a81be47f",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 2,
            "bytes": 238515,
            "sha1": "92a44368d5cf",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 3,
            "bytes": 285772,
            "sha1": "79190d4a3d9b",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 4,
            "bytes": 261027,
            "sha1": "4e713dbf5ff2",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 5,
            "bytes": 217957,
            "sha1": "6da790068293",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 6,
            "bytes": 190792,
            "sha1": "5f0fac34d2b2",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 7,
            "bytes": 196959,
            "sha1": "96751a51e23b",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_12140",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 294988,
            "sha1": "97810d4a1a0c",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 1,
            "bytes": 316136,
            "sha1": "07ff9202966b",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 2,
            "bytes": 388686,
            "sha1": "f80584505c17",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 3,
            "bytes": 345965,
            "sha1": "6304b353e04d",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 4,
            "bytes": 240051,
            "sha1": "aae29544d2d9",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 5,
            "bytes": 244562,
            "sha1": "5217139493f8",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 6,
            "bytes": 255804,
            "sha1": "ad712a083ea0",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 7,
            "bytes": 224098,
            "sha1": "f7251eaad8a7",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1200",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_12139"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```
