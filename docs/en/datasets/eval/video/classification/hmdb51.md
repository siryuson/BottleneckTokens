# HMDB51

> Classic video action recognition benchmark for complex scenes, covering 51 action classes.

| Property | Value |
|------|-----|
| Modality | video |
| Task type | `video_classification` |
| Purpose | Evaluation |
| Provenance layers | 3 |

## Original Dataset

### Source Information
| Property | Value |
|------|-----|
| Original name | HMDB51 |
| Paper | HMDB: A Large Video Database for Human Motion Recognition (Kuehne et al., 2011) ([Paper](https://openaccess.thecvf.com/content_ICCV_2011/papers/Kuehne_HMDB_A_Large_2011_ICCV_paper.pdf)) |
| HuggingFace | [siyrus/BToks-HMDB51](https://huggingface.co/datasets/siyrus/BToks-HMDB51) |
| License | See the original dataset. |

### Schema
- `video_id`: unique video identifier
- `pos_text`: positive action label text
- `neg_text` (optional): distractor label list

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-HMDB51](https://huggingface.co/datasets/siyrus/BToks-HMDB51) |
| Split | `test` |

### Schema
- `video_id`: video ID
- `pos_text`: correct class label
- `neg_text` (optional): candidate negative labels

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/HMDB51/` |
| task_type | `video_classification` |
| Evaluation mode | global |

### Lance Layout
- `queries.lance`: video-frame query (`id`, `text`, `images`)
- `candidates.lance`: global class candidates (`id`, `text`, `images=[]`)
- `qrels.lance`: sparse relevance annotations (`query_id`, `candidate_ids`, `candidate_scores`)

### Samples

The examples below are generated from the current runtime-canonical output backed by `./data/converted`.

- The `text` field shows the current runtime-canonical model-visible text; the parser may reconstruct visual-token layout, option blocks, and trailing newlines from raw Lance fields.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=pick`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nWhat actions or objects interactions are the person in the video doing?\n",
    "images": {
      "count": 16,
      "items": [
        {
          "index": 0,
          "bytes": 30573,
          "sha1": "d0cdde7cdb43",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 30542,
          "sha1": "fadf19fb1b36",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 31659,
          "sha1": "7e9d00e53012",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 32056,
          "sha1": "50d211839e3e",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 32283,
          "sha1": "f725775df558",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 31342,
          "sha1": "eee6158c2b90",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 32635,
          "sha1": "5a41da6a4c17",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 32369,
          "sha1": "d74ab95116b5",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 32832,
          "sha1": "d17472732ca4",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 32901,
          "sha1": "f1a7fe2a69f0",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 31815,
          "sha1": "a83caa0c0d36",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 30979,
          "sha1": "18162d00fb48",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 30313,
          "sha1": "c090f0dc53dd",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 31497,
          "sha1": "ef92fa348367",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 32180,
          "sha1": "c84875905f8f",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 31235,
          "sha1": "8bc2dcc59ec9",
          "format": "JPEG",
          "size": [
            432,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "pick",
      "text": "pick",
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
      "pick"
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
`query_id=q_666`, `positive_candidate_id=wave`

```json
{
  "queries.lance": {
    "id": "q_666",
    "text": "<|video_pad|>\nWhat actions or objects interactions are the person in the video doing?\n",
    "images": {
      "count": 10,
      "items": [
        {
          "index": 0,
          "bytes": 25412,
          "sha1": "cb2e9846db10",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 26587,
          "sha1": "4a549e94ba14",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 27418,
          "sha1": "1c52a40fee79",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 26554,
          "sha1": "a6ac35e704c3",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 24622,
          "sha1": "9ef82ce9f332",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 24643,
          "sha1": "fd603dbb9a7f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 24810,
          "sha1": "e2546bc91698",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 25738,
          "sha1": "35d6ded07308",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 24711,
          "sha1": "13764d49d568",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 25933,
          "sha1": "0db838402359",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "wave",
      "text": "wave",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_666",
    "mode": "sparse",
    "positive_candidate_ids": [
      "wave"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### Example 3
`query_id=q_749`, `positive_candidate_id=shoot_bow`

```json
{
  "queries.lance": {
    "id": "q_749",
    "text": "<|video_pad|>\nWhat actions or objects interactions are the person in the video doing?\n",
    "images": {
      "count": 40,
      "items": [
        {
          "index": 0,
          "bytes": 31405,
          "sha1": "bc8a841c1982",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 30386,
          "sha1": "49f3b9780ce7",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 30937,
          "sha1": "a9663f456056",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 29173,
          "sha1": "e9ec80d8e174",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 28999,
          "sha1": "d4e97a3a54f9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 27639,
          "sha1": "a16d1303d9c0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 26893,
          "sha1": "baaa98211d63",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 27458,
          "sha1": "c65429b99f9b",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 29075,
          "sha1": "e171492ff2c3",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 28068,
          "sha1": "0142449b348b",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 29308,
          "sha1": "ab6267843052",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 27597,
          "sha1": "548c321ed301",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 29184,
          "sha1": "d3b3c5862373",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 27688,
          "sha1": "436117b5d796",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 28222,
          "sha1": "b23b5ee481cf",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 27811,
          "sha1": "9e8553c5b366",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 28571,
          "sha1": "4e56a4f02ea2",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 28831,
          "sha1": "46cd074f61ed",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 29185,
          "sha1": "bf80bbf03fa0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 29098,
          "sha1": "34a899930ceb",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 29407,
          "sha1": "35bf228bfe6c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 29277,
          "sha1": "3a5fbf2cb89a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 29685,
          "sha1": "52b717d60d32",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 28665,
          "sha1": "eee6041bdd2a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 28995,
          "sha1": "4aef9b2e25a7",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 29080,
          "sha1": "3530ef0caf14",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 28919,
          "sha1": "d8f2e456dd21",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 29131,
          "sha1": "e535cb0f1d4f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 28693,
          "sha1": "e40ee8f15462",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 28751,
          "sha1": "7ac3896eb2bf",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 28295,
          "sha1": "d2216d3668f0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 28850,
          "sha1": "32bfd77d6722",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 29106,
          "sha1": "ced1cef80051",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 27272,
          "sha1": "ec84aa3bf7b2",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 27913,
          "sha1": "4e06d913dc2a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 35,
          "bytes": 27418,
          "sha1": "7376031f4ef1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 36,
          "bytes": 27713,
          "sha1": "ad5a6e6d7336",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 37,
          "bytes": 26793,
          "sha1": "40bcd7c05b6f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 38,
          "bytes": 26491,
          "sha1": "94040eaaa36c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 39,
          "bytes": 27196,
          "sha1": "27065ee92d1c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "shoot_bow",
      "text": "shoot_bow",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_749",
    "mode": "sparse",
    "positive_candidate_ids": [
      "shoot_bow"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
