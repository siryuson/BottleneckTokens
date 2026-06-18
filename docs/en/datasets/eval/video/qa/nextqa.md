# NExTQA

> Video QA benchmark focused on causal and temporal reasoning for deeper video understanding.

| Property | Value |
|------|-----|
| Modality | video |
| Task type | `video_question_answer` |
| Purpose | Evaluation |
| Provenance layers | 3 |

## Original Dataset

### Source Information
| Property | Value |
|------|-----|
| Original name | NExT-QA |
| Paper | NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions (Xiao et al., 2021) ([arXiv](https://arxiv.org/abs/2105.08276)) |
| HuggingFace | [siyrus/BToks-NExTQA](https://huggingface.co/datasets/siyrus/BToks-NExTQA) |
| License | See the original dataset. |

### Schema
- `video`: video ID
- `question`: question
- `a0`~`a4`(or `options`): candidate answers
- `answer`: correct answer

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-NExTQA](https://huggingface.co/datasets/siyrus/BToks-NExTQA) |
| Split | `test` (subset: `MC`) |

### Schema
- `video`: video identifier
- `question`: question
- `options`: multiple-choice candidates
- `answer`: correct option

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/NExTQA/` |
| task_type | `video_question_answer` |
| Evaluation mode | local |

### Lance Layout
- `queries.lance`: video frames plus question query (`id`, `text`, `images`)
- `candidates.lance`: text answer candidates (`id`, `text`, `images=[]`)
- `qrels.lance`: local exhaustive annotations (`mode=exhaustive`)

### Samples

The examples below are generated from the current runtime-canonical output backed by `./data/converted`.

- The `text` field shows the current runtime-canonical model-visible text; the parser may reconstruct visual-token layout, option blocks, and trailing newlines from raw Lance fields.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_2`, `negative_candidate_id=c_0`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  what did the baby do after throwing the green cup away while on the floor near the end\nOptions:\n(A) clap proudly\n(B) the lady sitting down\n(C) lay on floor\n(D) just picked it up\n(E) crawl\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 23532,
          "sha1": "eedca380e0aa",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 23957,
          "sha1": "c7ac39d1c4d7",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 23267,
          "sha1": "4420367b2ebb",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 22461,
          "sha1": "3739e7b79f6c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 23278,
          "sha1": "d7ddeb70d70d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 21755,
          "sha1": "f42c113865fa",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 20740,
          "sha1": "08dee267eaab",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 22332,
          "sha1": "334e8c08cf3e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 22305,
          "sha1": "30428e37950e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 23072,
          "sha1": "b32efd9a8bcc",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 22233,
          "sha1": "2c6f0988e7f6",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 22529,
          "sha1": "c9865c9ba91c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 23905,
          "sha1": "b574ad45917b",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 22437,
          "sha1": "3f2de128d2e4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 21858,
          "sha1": "254696fed794",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 20091,
          "sha1": "6164c8499bb9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 19577,
          "sha1": "91a6903a81d9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 21400,
          "sha1": "cbfb279fabd5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 20901,
          "sha1": "1248beba1ed0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 21058,
          "sha1": "96908c6ffe20",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 21003,
          "sha1": "14689884b408",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 19007,
          "sha1": "f3955ebb3ed5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 20221,
          "sha1": "b4e2262945c2",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 17900,
          "sha1": "07ad29f5ee79",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 17468,
          "sha1": "3fc4dbbf1592",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 16425,
          "sha1": "f30810240cb1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 19184,
          "sha1": "ce1716ada1da",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 19848,
          "sha1": "0008a8ba851a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 19632,
          "sha1": "98c37201367e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 19471,
          "sha1": "d9b74fa93126",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 19761,
          "sha1": "a461b4f0d003",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 18341,
          "sha1": "c50c34a7d6e5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 17777,
          "sha1": "a675dd2831eb",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 16677,
          "sha1": "c29d5771371a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 19947,
          "sha1": "9ec2a40f6b86",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 35,
          "bytes": 19875,
          "sha1": "6377f22a6843",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 36,
          "bytes": 20230,
          "sha1": "1f2b3f15ecde",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 37,
          "bytes": 19506,
          "sha1": "b9bf06e700f9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 38,
          "bytes": 18974,
          "sha1": "9309058b313a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 39,
          "bytes": 19306,
          "sha1": "3ded48883aad",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 40,
          "bytes": 19409,
          "sha1": "12adfdd7136a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 41,
          "bytes": 19377,
          "sha1": "6adaa98ad5b4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 42,
          "bytes": 20021,
          "sha1": "24bd13bb4177",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 43,
          "bytes": 20197,
          "sha1": "c25e72dd39ab",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 44,
          "bytes": 19169,
          "sha1": "53cc86314e9f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 45,
          "bytes": 19338,
          "sha1": "2f24203e50db",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 46,
          "bytes": 18946,
          "sha1": "49d71cfa91fb",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 47,
          "bytes": 18059,
          "sha1": "bc30b1ffc15a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 48,
          "bytes": 18639,
          "sha1": "11f6f7fa3bc8",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 49,
          "bytes": 19781,
          "sha1": "5c4bc6da1ede",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 50,
          "bytes": 19832,
          "sha1": "9f360025cdb7",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 51,
          "bytes": 19963,
          "sha1": "36ad5d8dc63f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 52,
          "bytes": 19269,
          "sha1": "2ceb43ba0644",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 53,
          "bytes": 19176,
          "sha1": "27772b89f047",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 54,
          "bytes": 19173,
          "sha1": "b09b2a75f902",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 55,
          "bytes": 18818,
          "sha1": "dec155848589",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 56,
          "bytes": 21037,
          "sha1": "d6e8e3c4c3dd",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 57,
          "bytes": 23175,
          "sha1": "6efcb16e905a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 58,
          "bytes": 22631,
          "sha1": "d65dfc2fd00c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 59,
          "bytes": 24626,
          "sha1": "ec2b425a0901",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 60,
          "bytes": 22603,
          "sha1": "a5e0afa09964",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 61,
          "bytes": 23846,
          "sha1": "4a3cd6cdb61f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 62,
          "bytes": 25573,
          "sha1": "688c75d381fc",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 63,
          "bytes": 25371,
          "sha1": "639a9eb152fe",
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
      "id": "c_2",
      "text": "lay on floor",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_0",
      "text": "clap proudly",
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
      "c_2"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 5,
    "negative_candidate_count": 4
  }
}
```

#### Example 2
`query_id=q_1`, `positive_candidate_id=c_6`, `negative_candidate_id=c_5`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  where is this place\nOptions:\n(A) mall\n(B) river\n(C) swimming pool\n(D) living room\n(E) mountain\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 1,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 2,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 3,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 4,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 5,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 6,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 7,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 8,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 9,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 10,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 11,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 12,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 13,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 14,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 15,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 16,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 17,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 18,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 19,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 20,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 21,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 22,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 23,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 24,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 25,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 26,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 27,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 28,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 29,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 30,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 31,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 32,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 33,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 34,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 35,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 36,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 37,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 38,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 39,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 40,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 41,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 42,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 43,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 44,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 45,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 46,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 47,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 48,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 49,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 50,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 51,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 52,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 53,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 54,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 55,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 56,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 57,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 58,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 59,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 60,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 61,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 62,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        },
        {
          "index": 63,
          "bytes": 221697,
          "sha1": "2bd7980fe6d2",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_6",
      "text": "river",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_5",
      "text": "mall",
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
      "c_6"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 5,
    "negative_candidate_count": 4
  }
}
```
