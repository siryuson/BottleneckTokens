# Charades-STA

> Temporal activity localization benchmark based on natural-language descriptions, derived from Charades annotations.

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
| Original name | Charades-STA |
| Paper | TALL: Temporal Activity Localization via Language Query (Gao et al., 2017) ([arXiv](https://arxiv.org/abs/1705.02101)) |
| HuggingFace | [siyrus/BToks-Charades-STA](https://huggingface.co/datasets/siyrus/BToks-Charades-STA) |
| License | See the original dataset. |

### Schema
- `query`: query sentence
- `clips_dir_path`: candidate clip directory
- `label_name`: positive moment annotation

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-Charades-STA](https://huggingface.co/datasets/siyrus/BToks-Charades-STA) |
| Split | `test` |

### Schema
- `query`: text query
- `clips_dir_path`: candidate clip path
- `label_name`: positive candidate identifier

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/Charades-STA/` |
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
    "text": "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: person turn a light on.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 161039,
          "sha1": "f7ae3afbfc5e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 1,
          "bytes": 169921,
          "sha1": "53016ce8cbfa",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 2,
          "bytes": 173504,
          "sha1": "b7848d878633",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 3,
          "bytes": 174959,
          "sha1": "5168aea4dfbb",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 4,
          "bytes": 175674,
          "sha1": "d0cab5eb5826",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 5,
          "bytes": 176791,
          "sha1": "d82763ebf99a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 6,
          "bytes": 177512,
          "sha1": "66fe181adf84",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 7,
          "bytes": 180304,
          "sha1": "459f68c6478c",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 8,
          "bytes": 181625,
          "sha1": "52cbbca166f2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 9,
          "bytes": 181220,
          "sha1": "4931e8a4022f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 10,
          "bytes": 178513,
          "sha1": "a7ef0a57cb7f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 11,
          "bytes": 173407,
          "sha1": "06af38036aaf",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 12,
          "bytes": 176884,
          "sha1": "f883519b4510",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 13,
          "bytes": 177906,
          "sha1": "fbf0bb2d4d7d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 14,
          "bytes": 179851,
          "sha1": "062db00450e0",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 15,
          "bytes": 182205,
          "sha1": "99527438349e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 16,
          "bytes": 183164,
          "sha1": "828ea4d43616",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 17,
          "bytes": 184087,
          "sha1": "36980b390e30",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 18,
          "bytes": 173565,
          "sha1": "ffea68d60f3e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 19,
          "bytes": 177479,
          "sha1": "6d97b30d1254",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 20,
          "bytes": 179818,
          "sha1": "bbf7851b884d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 21,
          "bytes": 181103,
          "sha1": "f82adb7c8402",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 22,
          "bytes": 180829,
          "sha1": "2ea53eed6e28",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 23,
          "bytes": 181473,
          "sha1": "41cb2f9d9f62",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 24,
          "bytes": 182205,
          "sha1": "0670ee72c1af",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 25,
          "bytes": 181759,
          "sha1": "eee7d343bdb6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 26,
          "bytes": 181851,
          "sha1": "f3af94f29c40",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 27,
          "bytes": 182526,
          "sha1": "3794d1f14b29",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 28,
          "bytes": 182019,
          "sha1": "6efe8b7d98b6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 29,
          "bytes": 183029,
          "sha1": "168caa3f4db8",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 30,
          "bytes": 182848,
          "sha1": "da1c79611b89",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 31,
          "bytes": 183786,
          "sha1": "c45d38f7f10d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 32,
          "bytes": 183998,
          "sha1": "27dadaeac30f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 33,
          "bytes": 184261,
          "sha1": "1316b91eba16",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 34,
          "bytes": 183927,
          "sha1": "ffe830c46ef1",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 35,
          "bytes": 184381,
          "sha1": "049f01f8d1cd",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 36,
          "bytes": 177534,
          "sha1": "0c239332dcf9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 37,
          "bytes": 180748,
          "sha1": "1e6c773c27ec",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 38,
          "bytes": 181543,
          "sha1": "183d3b5c670e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 39,
          "bytes": 182100,
          "sha1": "9e80cb0ab504",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 40,
          "bytes": 182634,
          "sha1": "00801997fd4d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 41,
          "bytes": 183272,
          "sha1": "052bf1123a6b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 42,
          "bytes": 183700,
          "sha1": "dca1a828c9b6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 43,
          "bytes": 183922,
          "sha1": "c7de586fd5ce",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 44,
          "bytes": 184115,
          "sha1": "d421612ce80a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 45,
          "bytes": 183064,
          "sha1": "9cebc937521f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 46,
          "bytes": 182061,
          "sha1": "140e9aec3f91",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 47,
          "bytes": 183210,
          "sha1": "d0236ed6d892",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 48,
          "bytes": 184552,
          "sha1": "9d10f8a848ad",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 49,
          "bytes": 184193,
          "sha1": "0c4a930db1b6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 50,
          "bytes": 184008,
          "sha1": "ea0215f861f9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 51,
          "bytes": 182928,
          "sha1": "db8699f18807",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 52,
          "bytes": 180656,
          "sha1": "63a5b7bf1698",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 53,
          "bytes": 175119,
          "sha1": "b0ab7ea4721b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 54,
          "bytes": 166870,
          "sha1": "3dc79e82b517",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 55,
          "bytes": 166954,
          "sha1": "cc8a4bdda6de",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 56,
          "bytes": 170640,
          "sha1": "34ffd90bdc0d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 57,
          "bytes": 171668,
          "sha1": "7a07828c7439",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 58,
          "bytes": 176211,
          "sha1": "7d195e2372b1",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 59,
          "bytes": 177157,
          "sha1": "3485e725cc58",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 60,
          "bytes": 177098,
          "sha1": "58020b8f1d65",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 61,
          "bytes": 176251,
          "sha1": "4b7a6ee19bc5",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 62,
          "bytes": 176988,
          "sha1": "d1731b522797",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 63,
          "bytes": 176064,
          "sha1": "13e4596d6e3f",
          "format": "JPEG",
          "size": [
            1280,
            720
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
            "bytes": 159736,
            "sha1": "8479ecfbcea8",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 163341,
            "sha1": "6f0ddbffe275",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 163381,
            "sha1": "f8b148f3524f",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 164047,
            "sha1": "1a49b7724fee",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 164234,
            "sha1": "fda097ae4946",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 162908,
            "sha1": "e6059e2aad27",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 164548,
            "sha1": "1874fdbf18de",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 163027,
            "sha1": "fda68d91e7a4",
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
      "id": "c_1",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 160521,
            "sha1": "0b8659f88f4d",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 162076,
            "sha1": "8ae931a9bd48",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 163168,
            "sha1": "6e3149eea9b6",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 163562,
            "sha1": "9d45e1b474c7",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 164040,
            "sha1": "e11f38051ac6",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 164547,
            "sha1": "a1d94ce3fc90",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 164823,
            "sha1": "d104401e8d07",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 164832,
            "sha1": "c86a30483b3d",
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
    "text": "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: the person puts down the bag.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 158169,
          "sha1": "8ad5fc5786fd",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 1,
          "bytes": 163196,
          "sha1": "955e1653522b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 2,
          "bytes": 158954,
          "sha1": "72c2f27f7a30",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 3,
          "bytes": 155981,
          "sha1": "ecc7f051510d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 4,
          "bytes": 155456,
          "sha1": "fa5e613ec532",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 5,
          "bytes": 157395,
          "sha1": "ab7a7803015a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 6,
          "bytes": 156425,
          "sha1": "564002b7d15e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 7,
          "bytes": 155867,
          "sha1": "caa3411b0de8",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 8,
          "bytes": 155827,
          "sha1": "3339fb26d772",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 9,
          "bytes": 161391,
          "sha1": "417867890b67",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 10,
          "bytes": 160760,
          "sha1": "8bf99a95e7f6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 11,
          "bytes": 158657,
          "sha1": "2d40d7457c42",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 12,
          "bytes": 158563,
          "sha1": "4f0447a0bedd",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 13,
          "bytes": 159015,
          "sha1": "c24f89d959b6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 14,
          "bytes": 155292,
          "sha1": "aff0952eca4f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 15,
          "bytes": 155198,
          "sha1": "f18cacb88009",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 16,
          "bytes": 157651,
          "sha1": "02858e4f678e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 17,
          "bytes": 163973,
          "sha1": "aa6391915506",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 18,
          "bytes": 166532,
          "sha1": "4672d7d36763",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 19,
          "bytes": 164005,
          "sha1": "a63dc0d3b18e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 20,
          "bytes": 165335,
          "sha1": "b0ff64c0de94",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 21,
          "bytes": 163934,
          "sha1": "a35cacdb9cbb",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 22,
          "bytes": 165476,
          "sha1": "83bd914eeca1",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 23,
          "bytes": 163313,
          "sha1": "9d0106754c1f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 24,
          "bytes": 163528,
          "sha1": "55967c94f5e3",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 25,
          "bytes": 164563,
          "sha1": "aa61bbcadd35",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 26,
          "bytes": 165784,
          "sha1": "62c03a4842a9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 27,
          "bytes": 164267,
          "sha1": "48320db0afd1",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 28,
          "bytes": 163703,
          "sha1": "8a4bd39de56a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 29,
          "bytes": 163924,
          "sha1": "95652194f76a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 30,
          "bytes": 167745,
          "sha1": "f6e3b9903310",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 31,
          "bytes": 164352,
          "sha1": "c6e833028813",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 32,
          "bytes": 166551,
          "sha1": "c24a0e4d7afa",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 33,
          "bytes": 165577,
          "sha1": "a740739bd977",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 34,
          "bytes": 165296,
          "sha1": "928c1e1bf076",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 35,
          "bytes": 153177,
          "sha1": "55c52a514e36",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 36,
          "bytes": 162992,
          "sha1": "c6916237444b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 37,
          "bytes": 163311,
          "sha1": "9e4c65b5b7aa",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 38,
          "bytes": 163803,
          "sha1": "14301f53444b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 39,
          "bytes": 163581,
          "sha1": "fd5abcb16e8d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 40,
          "bytes": 163956,
          "sha1": "f6e3481b26ea",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 41,
          "bytes": 165761,
          "sha1": "f89b11983a1d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 42,
          "bytes": 165163,
          "sha1": "85532eb37e0d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 43,
          "bytes": 164032,
          "sha1": "3284b87f3e8d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 44,
          "bytes": 163794,
          "sha1": "3ee18cfde0ad",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 45,
          "bytes": 164479,
          "sha1": "a1a1d83248b3",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 46,
          "bytes": 165218,
          "sha1": "edb65000c5c5",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 47,
          "bytes": 163719,
          "sha1": "7dbc8e4154de",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 48,
          "bytes": 162011,
          "sha1": "b0889240c322",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 49,
          "bytes": 163753,
          "sha1": "9305be101da6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 50,
          "bytes": 165156,
          "sha1": "a30dbaf9056b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 51,
          "bytes": 165590,
          "sha1": "97d4892d2f87",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 52,
          "bytes": 165105,
          "sha1": "65ba245ffc86",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 53,
          "bytes": 159693,
          "sha1": "8ebdc951fb4f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 54,
          "bytes": 163655,
          "sha1": "40f9e46ac163",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 55,
          "bytes": 164211,
          "sha1": "8fbe0af24f4f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 56,
          "bytes": 164707,
          "sha1": "488f9105f111",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 57,
          "bytes": 164963,
          "sha1": "697b45556bcc",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 58,
          "bytes": 164637,
          "sha1": "35f5836ba760",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 59,
          "bytes": 165221,
          "sha1": "1a233822f257",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 60,
          "bytes": 165294,
          "sha1": "9f1d1eee1d6f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 61,
          "bytes": 164779,
          "sha1": "eb8be7e0e532",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 62,
          "bytes": 165780,
          "sha1": "511310bf7afa",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 63,
          "bytes": 164489,
          "sha1": "c7c3668035ea",
          "format": "JPEG",
          "size": [
            1280,
            720
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
            "bytes": 136634,
            "sha1": "70878a853a1e",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 146113,
            "sha1": "b2dc063fa437",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 148767,
            "sha1": "7c2a9cac4055",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 148811,
            "sha1": "dce846141c41",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 146694,
            "sha1": "f11206d2bd61",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 148320,
            "sha1": "443f58de863a",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 149614,
            "sha1": "861b2fd0cf8c",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 147520,
            "sha1": "ac714d1c3dcb",
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
      "id": "c_11",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 140294,
            "sha1": "4f507f06647c",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 147138,
            "sha1": "bb7920128053",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 148651,
            "sha1": "4e2eea62d3ef",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 149765,
            "sha1": "f57152643dc1",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 150139,
            "sha1": "073a2a299685",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 151415,
            "sha1": "f13a47f34330",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 152160,
            "sha1": "cd74dfdd5d1e",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 153218,
            "sha1": "6f555094c75f",
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
