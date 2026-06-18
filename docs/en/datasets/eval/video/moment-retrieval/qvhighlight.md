# QVHighlights

> Query-driven dataset for joint video moment and highlight localization.

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
| Original name | QVHighlights |
| Paper | QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries (Lei et al., 2021) ([arXiv](https://arxiv.org/abs/2107.09609)) |
| HuggingFace | [siyrus/BToks-QVHighlight](https://huggingface.co/datasets/siyrus/BToks-QVHighlight) |
| License | See the original dataset. |

### Schema
- `query`: query text
- `clips_dir_path`/`video_id`: candidate clip source
- `label_name`: positive clip identifier

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-QVHighlight](https://huggingface.co/datasets/siyrus/BToks-QVHighlight) |
| Split | `test` |

### Schema
- `query`: text query
- `clips_dir_path`: candidate clip directory for the query
- `label_name`: target moment label

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/QVHighlight/` |
| task_type | `video_moment_retrieval` |
| Evaluation mode | local |

### Lance Layout
- `queries.lance`: query video frames plus text (`id`, `text`, `images`)
- `candidates.lance`: candidate video clips (`id`, `text`, `images`)
- `qrels.lance`: exhaustive relevance annotations (`mode=exhaustive`)

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
    "text": "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: A girl and her mother cooked while talking with each other on facetime.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 14409,
          "sha1": "799a9d31d2f3",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 1,
          "bytes": 21381,
          "sha1": "eb73ad9635d9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 2,
          "bytes": 20650,
          "sha1": "29642ebcc8c3",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 3,
          "bytes": 21360,
          "sha1": "c26876b4be9e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 4,
          "bytes": 14743,
          "sha1": "0a5b0cc28c54",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 5,
          "bytes": 11287,
          "sha1": "c60ec2bd23d9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 6,
          "bytes": 8928,
          "sha1": "1ceece7d1e52",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 7,
          "bytes": 7735,
          "sha1": "3a90564ff5e8",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 8,
          "bytes": 7028,
          "sha1": "5a39b341bbb3",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 9,
          "bytes": 6238,
          "sha1": "5ad3d8671904",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 10,
          "bytes": 6429,
          "sha1": "78273097402b",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 11,
          "bytes": 5608,
          "sha1": "f1f90b3aa0a1",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 12,
          "bytes": 5487,
          "sha1": "17f638f8bfaf",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 13,
          "bytes": 5327,
          "sha1": "b12f57f50987",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 14,
          "bytes": 5706,
          "sha1": "5d55d071a54f",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 15,
          "bytes": 5600,
          "sha1": "6400e32f1a0a",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 16,
          "bytes": 5389,
          "sha1": "838e9aa812d0",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 17,
          "bytes": 5507,
          "sha1": "2d3aefba3ba9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 18,
          "bytes": 5488,
          "sha1": "0d992103e7e2",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 19,
          "bytes": 5772,
          "sha1": "b23ee7c1dfa1",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 20,
          "bytes": 5448,
          "sha1": "0326f43eae4d",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 21,
          "bytes": 5686,
          "sha1": "7f0d6a9466c5",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 22,
          "bytes": 5422,
          "sha1": "804191416648",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 23,
          "bytes": 5269,
          "sha1": "2e10fd180339",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 24,
          "bytes": 5558,
          "sha1": "9fe5f3a73384",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 25,
          "bytes": 5654,
          "sha1": "59fdeee5cdd5",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 26,
          "bytes": 5611,
          "sha1": "2fdf4270c3e1",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 27,
          "bytes": 5309,
          "sha1": "5e00678e8cf6",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 28,
          "bytes": 5616,
          "sha1": "24dda4903da7",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 29,
          "bytes": 5640,
          "sha1": "8aa39cf6babe",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 30,
          "bytes": 5723,
          "sha1": "50b8a527fff9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 31,
          "bytes": 6015,
          "sha1": "ab46740b98c5",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 32,
          "bytes": 5730,
          "sha1": "838a128c3b21",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 33,
          "bytes": 5749,
          "sha1": "d71aa72f7b21",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 34,
          "bytes": 5615,
          "sha1": "e3db6f451cc8",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 35,
          "bytes": 2606,
          "sha1": "ca83c50cf6c2",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 36,
          "bytes": 3788,
          "sha1": "441e858961b1",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 37,
          "bytes": 3822,
          "sha1": "fdd9eebd6c29",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 38,
          "bytes": 3996,
          "sha1": "1e662f0b534d",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 39,
          "bytes": 4983,
          "sha1": "a3ef9ab5d6cf",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 40,
          "bytes": 2396,
          "sha1": "f9a00e526868",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 41,
          "bytes": 5498,
          "sha1": "dc59bcbaa974",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 42,
          "bytes": 7500,
          "sha1": "cdae79692fcb",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 43,
          "bytes": 7143,
          "sha1": "ba533c3c3c77",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 44,
          "bytes": 6974,
          "sha1": "63250c8e1e96",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 45,
          "bytes": 6858,
          "sha1": "409d66252f21",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 46,
          "bytes": 7051,
          "sha1": "c7f2c5874480",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 47,
          "bytes": 7088,
          "sha1": "f35f253bf994",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 48,
          "bytes": 7097,
          "sha1": "99891d93da80",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 49,
          "bytes": 7021,
          "sha1": "243ad0faac88",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 50,
          "bytes": 7238,
          "sha1": "041dfe881fe9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 51,
          "bytes": 6649,
          "sha1": "48d697b43eb1",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 52,
          "bytes": 6977,
          "sha1": "f16cb59745c6",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 53,
          "bytes": 6504,
          "sha1": "42ad8dcc61db",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 54,
          "bytes": 7020,
          "sha1": "0ac4a3dd7f95",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 55,
          "bytes": 7226,
          "sha1": "be061f5df930",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 56,
          "bytes": 7065,
          "sha1": "5f9e091bcca4",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 57,
          "bytes": 7393,
          "sha1": "d0e743a6dc31",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 58,
          "bytes": 8247,
          "sha1": "d04592472761",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 59,
          "bytes": 8929,
          "sha1": "8f39154fe044",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 60,
          "bytes": 9184,
          "sha1": "de00137400d8",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 61,
          "bytes": 8971,
          "sha1": "46990a956a2f",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 62,
          "bytes": 8951,
          "sha1": "4a596dd8a5f6",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 63,
          "bytes": 10510,
          "sha1": "8e10530f6c50",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        }
      ]
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
            "bytes": 13822,
            "sha1": "61900287855f",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 1,
            "bytes": 21363,
            "sha1": "6298dffea1c0",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 2,
            "bytes": 21764,
            "sha1": "05cd39d5d825",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 3,
            "bytes": 21331,
            "sha1": "373c30ee1311",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 4,
            "bytes": 12709,
            "sha1": "c7ff9a4fe4d8",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 5,
            "bytes": 10507,
            "sha1": "8509221d0805",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 6,
            "bytes": 8717,
            "sha1": "5731b31a4c43",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 7,
            "bytes": 7952,
            "sha1": "71724c0e13bf",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 12440,
            "sha1": "875e8688a965",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 1,
            "bytes": 19624,
            "sha1": "8b461c6a8985",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 2,
            "bytes": 20277,
            "sha1": "1477ff5d3e62",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 3,
            "bytes": 19717,
            "sha1": "aaff559e1f0f",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 4,
            "bytes": 13258,
            "sha1": "6e4663883235",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 5,
            "bytes": 9718,
            "sha1": "32736666bab6",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 6,
            "bytes": 8045,
            "sha1": "767eec00c2be",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 7,
            "bytes": 7222,
            "sha1": "00feda3bb74f",
            "format": "JPEG",
            "size": [
              570,
              300
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
`query_id=q_1`, `positive_candidate_id=c_10`, `negative_candidate_id=c_11`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: A woman sitting in front of a desk wearing headphones and using her laptop\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 16899,
          "sha1": "a31f94f196cc",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 1,
          "bytes": 20822,
          "sha1": "f29c14d7fd34",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 2,
          "bytes": 26983,
          "sha1": "0122fc53b8d1",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 3,
          "bytes": 20636,
          "sha1": "739774be8605",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 4,
          "bytes": 18738,
          "sha1": "b4b2a8f84b88",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 5,
          "bytes": 13369,
          "sha1": "2befe5cd1c5f",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 6,
          "bytes": 10850,
          "sha1": "89c0176caedd",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 7,
          "bytes": 9746,
          "sha1": "f0202a41a027",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 8,
          "bytes": 9020,
          "sha1": "5d7d3110ad45",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 9,
          "bytes": 8282,
          "sha1": "30eb6415172e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 10,
          "bytes": 7414,
          "sha1": "c36a13d08027",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 11,
          "bytes": 7736,
          "sha1": "84e7311d6bcb",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 12,
          "bytes": 6713,
          "sha1": "2d529cd412b4",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 13,
          "bytes": 6811,
          "sha1": "a17a1c7de777",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 14,
          "bytes": 6891,
          "sha1": "789acc759aee",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 15,
          "bytes": 6927,
          "sha1": "29f1a3356b63",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 16,
          "bytes": 6960,
          "sha1": "f4cb58dd30db",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 17,
          "bytes": 6844,
          "sha1": "bb3ef6834459",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 18,
          "bytes": 7056,
          "sha1": "32b0ea4c2ac6",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 19,
          "bytes": 6085,
          "sha1": "b9712a223d6e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 20,
          "bytes": 6934,
          "sha1": "d4544e98d24b",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 21,
          "bytes": 6877,
          "sha1": "b761a5b17c17",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 22,
          "bytes": 7044,
          "sha1": "56a4e33fadf3",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 23,
          "bytes": 6946,
          "sha1": "2e01c7245dc6",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 24,
          "bytes": 7051,
          "sha1": "ec3b1447980a",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 25,
          "bytes": 6948,
          "sha1": "963b89c3ba33",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 26,
          "bytes": 6971,
          "sha1": "3b559b6b1e15",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 27,
          "bytes": 6850,
          "sha1": "2ddbf6216cb9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 28,
          "bytes": 6792,
          "sha1": "2422b5a48aaa",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 29,
          "bytes": 7009,
          "sha1": "fbe7a49abeb7",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 30,
          "bytes": 7070,
          "sha1": "d54526cac364",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 31,
          "bytes": 7053,
          "sha1": "09d915bf925f",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 32,
          "bytes": 6481,
          "sha1": "993c4dcf6ffc",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 33,
          "bytes": 6695,
          "sha1": "1c8477e0822e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 34,
          "bytes": 6843,
          "sha1": "fb8279f7fdb3",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 35,
          "bytes": 6835,
          "sha1": "52ca3a7bfe73",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 36,
          "bytes": 7047,
          "sha1": "c8f04a7537af",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 37,
          "bytes": 7089,
          "sha1": "9711672a52ae",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 38,
          "bytes": 6923,
          "sha1": "08cbd4d0acc9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 39,
          "bytes": 5976,
          "sha1": "b8f4e0f86525",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 40,
          "bytes": 6736,
          "sha1": "5fddddea808e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 41,
          "bytes": 6260,
          "sha1": "abb60ba7d986",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 42,
          "bytes": 7615,
          "sha1": "20c378b6e9f9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 43,
          "bytes": 5679,
          "sha1": "c950384d5d58",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 44,
          "bytes": 7481,
          "sha1": "fa8c67fec8e8",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 45,
          "bytes": 7761,
          "sha1": "c9b291a48ed7",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 46,
          "bytes": 6778,
          "sha1": "e852d3249129",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 47,
          "bytes": 7059,
          "sha1": "9f9722a8738c",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 48,
          "bytes": 7281,
          "sha1": "06b3a0658243",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 49,
          "bytes": 6546,
          "sha1": "031cbbf3144e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 50,
          "bytes": 8680,
          "sha1": "c3ffeb8a107e",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 51,
          "bytes": 8858,
          "sha1": "8978892db3b0",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 52,
          "bytes": 8625,
          "sha1": "14b787b672b9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 53,
          "bytes": 8135,
          "sha1": "415c1fabf9d9",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 54,
          "bytes": 7999,
          "sha1": "6f692f89272a",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 55,
          "bytes": 7860,
          "sha1": "5e87cfe993e4",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 56,
          "bytes": 8007,
          "sha1": "46208a59c83a",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 57,
          "bytes": 8244,
          "sha1": "316a4c821257",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 58,
          "bytes": 5749,
          "sha1": "126adccd4847",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 59,
          "bytes": 6029,
          "sha1": "ac104872f16a",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 60,
          "bytes": 6005,
          "sha1": "6a6ae0482beb",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 61,
          "bytes": 6399,
          "sha1": "ce402e805c08",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 62,
          "bytes": 6333,
          "sha1": "d773d3aa68f6",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        },
        {
          "index": 63,
          "bytes": 6490,
          "sha1": "d1353d921e5a",
          "format": "JPEG",
          "size": [
            570,
            300
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_10",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 17737,
            "sha1": "0660872f8586",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 1,
            "bytes": 21954,
            "sha1": "4464c0728bd9",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 2,
            "bytes": 31581,
            "sha1": "e63572d6452c",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 3,
            "bytes": 24899,
            "sha1": "372dd5be6456",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 4,
            "bytes": 20063,
            "sha1": "a79ea4255dfa",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 5,
            "bytes": 16408,
            "sha1": "b6ddd43cd007",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 6,
            "bytes": 12813,
            "sha1": "cc890200f822",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 7,
            "bytes": 10937,
            "sha1": "14cc43c739cc",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_11",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 11615,
            "sha1": "ffd0fe7199e4",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 1,
            "bytes": 15723,
            "sha1": "fcba27390928",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 2,
            "bytes": 20641,
            "sha1": "25aa4e165c0d",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 3,
            "bytes": 20262,
            "sha1": "a0559768713c",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 4,
            "bytes": 12172,
            "sha1": "2fe4c95f64ab",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 5,
            "bytes": 10445,
            "sha1": "49dececa06e9",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 6,
            "bytes": 8426,
            "sha1": "c12f3ba7426b",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          },
          {
            "index": 7,
            "bytes": 7417,
            "sha1": "6ceee63670f2",
            "format": "JPEG",
            "size": [
              570,
              300
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_10"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```
