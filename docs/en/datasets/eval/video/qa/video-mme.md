# Video-MME

> Comprehensive multimodal QA benchmark for evaluating multiple dimensions of video understanding.

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
| Original name | Video-MME |
| Paper | Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis (Fu et al., 2024) ([arXiv](https://arxiv.org/abs/2405.21075)) |
| HuggingFace | [siyrus/BToks-Video-MME](https://huggingface.co/datasets/siyrus/BToks-Video-MME) |
| License | See the original dataset. |

### Schema
- `videoID`: video ID
- `question`: question text
- `options`: option list
- `answer`: correct option marker

### Samples
> Sample screenshot pending.

---

## VLM2Vec HF Format

### Source Information
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-Video-MME](https://huggingface.co/datasets/siyrus/BToks-Video-MME) |
| Split | `test` |

### Schema
- `videoID`: video identifier
- `question`: question
- `options`: candidate answers
- `answer`: correct answer index/marker

### Samples
> Sample screenshot pending.

---

## BToks Lance Format

### Source Information
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/video-tasks/Video-MME/` |
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
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?\nA. Apples.\nB. Candles.\nC. Berries.\nD. The three kinds are of the same number.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 63159,
          "sha1": "031760fd3dcc",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 1,
          "bytes": 61400,
          "sha1": "34b3e4675a01",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 2,
          "bytes": 60778,
          "sha1": "07514eabf8e8",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 3,
          "bytes": 187006,
          "sha1": "498b7333d852",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 4,
          "bytes": 270448,
          "sha1": "c505705a3918",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 5,
          "bytes": 275609,
          "sha1": "d7419190bf3e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 6,
          "bytes": 274686,
          "sha1": "145f90cf7e96",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 7,
          "bytes": 274447,
          "sha1": "ced8ff91d260",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 8,
          "bytes": 273770,
          "sha1": "5b8c52e8ee42",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 9,
          "bytes": 266743,
          "sha1": "ab6f68be70d9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 10,
          "bytes": 266424,
          "sha1": "9e04d1b30394",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 11,
          "bytes": 265620,
          "sha1": "0d337206a884",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 12,
          "bytes": 264497,
          "sha1": "41c9f12808e2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 13,
          "bytes": 266077,
          "sha1": "d484bffb68d9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 14,
          "bytes": 323905,
          "sha1": "b006d2db421d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 15,
          "bytes": 309871,
          "sha1": "3eb3bd117a15",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 16,
          "bytes": 321426,
          "sha1": "006c67ce2c7e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 17,
          "bytes": 161286,
          "sha1": "4674f9e3e213",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 18,
          "bytes": 175171,
          "sha1": "d15378bdea9f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 19,
          "bytes": 178213,
          "sha1": "7f4d6e556c7d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 20,
          "bytes": 178173,
          "sha1": "e057b8297138",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 21,
          "bytes": 87672,
          "sha1": "33becef5a0ce",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 22,
          "bytes": 109986,
          "sha1": "fb595602dbe6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 23,
          "bytes": 109992,
          "sha1": "d048f4b0b723",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 24,
          "bytes": 109992,
          "sha1": "25eca68039f8",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 25,
          "bytes": 445600,
          "sha1": "bbb8db949da4",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 26,
          "bytes": 438234,
          "sha1": "190f277c5db9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 27,
          "bytes": 196226,
          "sha1": "a549419a18ca",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 28,
          "bytes": 401906,
          "sha1": "0037209e50cf",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 29,
          "bytes": 103026,
          "sha1": "a6b38d32b91d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 30,
          "bytes": 132834,
          "sha1": "86b13cc22278",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 31,
          "bytes": 245898,
          "sha1": "3498fddc46e7",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 32,
          "bytes": 238955,
          "sha1": "5111e663302d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 33,
          "bytes": 249033,
          "sha1": "7118744c56b2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 34,
          "bytes": 307708,
          "sha1": "9aeee1107042",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 35,
          "bytes": 149728,
          "sha1": "f1c304d2bdd9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 36,
          "bytes": 149847,
          "sha1": "0f5e2a264d81",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 37,
          "bytes": 150079,
          "sha1": "dab78a2e881b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 38,
          "bytes": 250486,
          "sha1": "53a840028950",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 39,
          "bytes": 267279,
          "sha1": "75a51c8ab20a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 40,
          "bytes": 286844,
          "sha1": "620549419bf5",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 41,
          "bytes": 293107,
          "sha1": "bad707dcaf68",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 42,
          "bytes": 297073,
          "sha1": "1ad93d888b1c",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 43,
          "bytes": 452999,
          "sha1": "b2f173f6284b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 44,
          "bytes": 424475,
          "sha1": "b8d3cac65724",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 45,
          "bytes": 406530,
          "sha1": "c08fbde0f966",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 46,
          "bytes": 531923,
          "sha1": "01cdadd2cf5f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 47,
          "bytes": 451344,
          "sha1": "e44cbedf3404",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 48,
          "bytes": 150158,
          "sha1": "9e499771706a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 49,
          "bytes": 141577,
          "sha1": "c7acfb2fd7bd",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 50,
          "bytes": 45358,
          "sha1": "5b81935d99c7",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 51,
          "bytes": 46147,
          "sha1": "c2fed78aa67f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 52,
          "bytes": 26327,
          "sha1": "46dae3c1559a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 53,
          "bytes": 268916,
          "sha1": "ee94c55a7de7",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 54,
          "bytes": 269027,
          "sha1": "a7a89a2d0176",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 55,
          "bytes": 265970,
          "sha1": "5907ee916a98",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 56,
          "bytes": 329744,
          "sha1": "2616979783be",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 57,
          "bytes": 335877,
          "sha1": "317f81d41aed",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 58,
          "bytes": 334738,
          "sha1": "29c392a9094c",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 59,
          "bytes": 341702,
          "sha1": "057863f530b2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 60,
          "bytes": 285759,
          "sha1": "086e1db8d7d6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 61,
          "bytes": 318161,
          "sha1": "18628005ddaf",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 62,
          "bytes": 319931,
          "sha1": "5bce274d815b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 63,
          "bytes": 334645,
          "sha1": "cd783f7bea2b",
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
      "id": "c_2",
      "text": "C. Berries.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_0",
      "text": "A. Apples.",
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
    "total_candidate_ids": 4,
    "negative_candidate_count": 3
  }
}
```

#### Example 2
`query_id=q_1`, `positive_candidate_id=c_4`, `negative_candidate_id=c_5`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  What is the genre of this video?\nA. It is a news report that introduces the history behind Christmas decorations.\nB. It is a documentary on the evolution of Christmas holiday recipes.\nC. It is a travel vlog exploring Christmas markets around the world.\nD. It is a tutorial on DIY Christmas ornament crafting.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 63159,
          "sha1": "031760fd3dcc",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 1,
          "bytes": 61400,
          "sha1": "34b3e4675a01",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 2,
          "bytes": 60778,
          "sha1": "07514eabf8e8",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 3,
          "bytes": 187006,
          "sha1": "498b7333d852",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 4,
          "bytes": 270448,
          "sha1": "c505705a3918",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 5,
          "bytes": 275609,
          "sha1": "d7419190bf3e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 6,
          "bytes": 274686,
          "sha1": "145f90cf7e96",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 7,
          "bytes": 274447,
          "sha1": "ced8ff91d260",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 8,
          "bytes": 273770,
          "sha1": "5b8c52e8ee42",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 9,
          "bytes": 266743,
          "sha1": "ab6f68be70d9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 10,
          "bytes": 266424,
          "sha1": "9e04d1b30394",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 11,
          "bytes": 265620,
          "sha1": "0d337206a884",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 12,
          "bytes": 264497,
          "sha1": "41c9f12808e2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 13,
          "bytes": 266077,
          "sha1": "d484bffb68d9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 14,
          "bytes": 323905,
          "sha1": "b006d2db421d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 15,
          "bytes": 309871,
          "sha1": "3eb3bd117a15",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 16,
          "bytes": 321426,
          "sha1": "006c67ce2c7e",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 17,
          "bytes": 161286,
          "sha1": "4674f9e3e213",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 18,
          "bytes": 175171,
          "sha1": "d15378bdea9f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 19,
          "bytes": 178213,
          "sha1": "7f4d6e556c7d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 20,
          "bytes": 178173,
          "sha1": "e057b8297138",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 21,
          "bytes": 87672,
          "sha1": "33becef5a0ce",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 22,
          "bytes": 109986,
          "sha1": "fb595602dbe6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 23,
          "bytes": 109992,
          "sha1": "d048f4b0b723",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 24,
          "bytes": 109992,
          "sha1": "25eca68039f8",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 25,
          "bytes": 445600,
          "sha1": "bbb8db949da4",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 26,
          "bytes": 438234,
          "sha1": "190f277c5db9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 27,
          "bytes": 196226,
          "sha1": "a549419a18ca",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 28,
          "bytes": 401906,
          "sha1": "0037209e50cf",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 29,
          "bytes": 103026,
          "sha1": "a6b38d32b91d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 30,
          "bytes": 132834,
          "sha1": "86b13cc22278",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 31,
          "bytes": 245898,
          "sha1": "3498fddc46e7",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 32,
          "bytes": 238955,
          "sha1": "5111e663302d",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 33,
          "bytes": 249033,
          "sha1": "7118744c56b2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 34,
          "bytes": 307708,
          "sha1": "9aeee1107042",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 35,
          "bytes": 149728,
          "sha1": "f1c304d2bdd9",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 36,
          "bytes": 149847,
          "sha1": "0f5e2a264d81",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 37,
          "bytes": 150079,
          "sha1": "dab78a2e881b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 38,
          "bytes": 250486,
          "sha1": "53a840028950",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 39,
          "bytes": 267279,
          "sha1": "75a51c8ab20a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 40,
          "bytes": 286844,
          "sha1": "620549419bf5",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 41,
          "bytes": 293107,
          "sha1": "bad707dcaf68",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 42,
          "bytes": 297073,
          "sha1": "1ad93d888b1c",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 43,
          "bytes": 452999,
          "sha1": "b2f173f6284b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 44,
          "bytes": 424475,
          "sha1": "b8d3cac65724",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 45,
          "bytes": 406530,
          "sha1": "c08fbde0f966",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 46,
          "bytes": 531923,
          "sha1": "01cdadd2cf5f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 47,
          "bytes": 451344,
          "sha1": "e44cbedf3404",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 48,
          "bytes": 150158,
          "sha1": "9e499771706a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 49,
          "bytes": 141577,
          "sha1": "c7acfb2fd7bd",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 50,
          "bytes": 45358,
          "sha1": "5b81935d99c7",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 51,
          "bytes": 46147,
          "sha1": "c2fed78aa67f",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 52,
          "bytes": 26327,
          "sha1": "46dae3c1559a",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 53,
          "bytes": 268916,
          "sha1": "ee94c55a7de7",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 54,
          "bytes": 269027,
          "sha1": "a7a89a2d0176",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 55,
          "bytes": 265970,
          "sha1": "5907ee916a98",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 56,
          "bytes": 329744,
          "sha1": "2616979783be",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 57,
          "bytes": 335877,
          "sha1": "317f81d41aed",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 58,
          "bytes": 334738,
          "sha1": "29c392a9094c",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 59,
          "bytes": 341702,
          "sha1": "057863f530b2",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 60,
          "bytes": 285759,
          "sha1": "086e1db8d7d6",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 61,
          "bytes": 318161,
          "sha1": "18628005ddaf",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 62,
          "bytes": 319931,
          "sha1": "5bce274d815b",
          "format": "JPEG",
          "size": [
            1280,
            720
          ]
        },
        {
          "index": 63,
          "bytes": 334645,
          "sha1": "cd783f7bea2b",
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
      "id": "c_4",
      "text": "A. It is a news report that introduces the history behind Christmas decorations.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_5",
      "text": "B. It is a documentary on the evolution of Christmas holiday recipes.",
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
      "c_4"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 4,
    "negative_candidate_count": 3
  }
}
```
