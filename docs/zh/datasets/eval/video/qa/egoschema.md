# EgoSchema

> 超长第一视角视频问答基准，评估长时序理解与推理能力。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_question_answer` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | EgoSchema |
| 论文 | EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding（Mangalam et al., 2023）([arXiv](https://arxiv.org/abs/2308.09126)) |
| HuggingFace | [siyrus/BToks-EgoSchema](https://huggingface.co/datasets/siyrus/BToks-EgoSchema) |
| 许可证 | 参见原始数据集 |

### Schema
- `video_id`: 视频标识
- `question`: 问题文本
- `options`: 多选答案列表
- `answer`: 正确答案索引

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-EgoSchema](https://huggingface.co/datasets/siyrus/BToks-EgoSchema) |
| 分割 | `test`（subset: `Subset`） |

### Schema
- `video_id`: 视频 ID
- `question`: 问题
- `options`: 候选答案
- `answer`: 正确选项

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/EgoSchema/` |
| task_type | `video_question_answer` |
| 评测模式 | local |

### Lance 布局
- `queries.lance`: 视频帧+问题查询（`id`, `text`, `images`）
- `candidates.lance`: 文本候选答案（`id`, `text`, `images=[]`）
- `qrels.lance`: 局部穷举标注（`mode=exhaustive`）

### 样本

以下示例基于当前 runtime canonical 输出生成，数据来源为 `./data/converted`.

- `text` 展示当前 runtime canonical 的模型可见文本；parser 可能在 Lance 原始字段基础上重组 visual token 布局、选项块和尾部换行。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_3`，`negative_candidate_id=c_0`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  Taking into account all the actions performed by c, what can you deduce about the primary objective and focus within the video content?A. C is cooking. B. C is doing laundry. C. C is cleaning the kitchen. D. C is cleaning dishes. E. C is cleaning the bathroom.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 66293,
          "sha1": "537d4cb650c0",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 1,
          "bytes": 51022,
          "sha1": "ce409532d3b6",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 2,
          "bytes": 57703,
          "sha1": "d083044b5a5b",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 3,
          "bytes": 56081,
          "sha1": "3de83d512521",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 4,
          "bytes": 62477,
          "sha1": "3e4767e8a8e0",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 5,
          "bytes": 53288,
          "sha1": "58cd94275e7e",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 6,
          "bytes": 56479,
          "sha1": "9bca3edd93bc",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 7,
          "bytes": 60210,
          "sha1": "ab552f1f56fa",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 8,
          "bytes": 56081,
          "sha1": "2a57c93a3246",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 9,
          "bytes": 57946,
          "sha1": "2a18b2dd1ae5",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 10,
          "bytes": 56295,
          "sha1": "acbdd9c072be",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 11,
          "bytes": 57596,
          "sha1": "8092b4d7e257",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 12,
          "bytes": 52045,
          "sha1": "e44f22b88218",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 13,
          "bytes": 52352,
          "sha1": "d6b76bfef7b1",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 14,
          "bytes": 37669,
          "sha1": "76df685aceaa",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 15,
          "bytes": 42152,
          "sha1": "a6ac715e0d56",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 16,
          "bytes": 56494,
          "sha1": "4f46e4d7c480",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 17,
          "bytes": 52655,
          "sha1": "2490bfdeef3d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 18,
          "bytes": 49090,
          "sha1": "1e9275451681",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 19,
          "bytes": 58000,
          "sha1": "766f9c15327d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 20,
          "bytes": 52415,
          "sha1": "fe90c7378fd9",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 21,
          "bytes": 53208,
          "sha1": "205bd97ff710",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 22,
          "bytes": 43437,
          "sha1": "4fdf0cbae00d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 23,
          "bytes": 31150,
          "sha1": "675fbf2aa0e6",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 24,
          "bytes": 48013,
          "sha1": "1bce04a06dd9",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 25,
          "bytes": 54747,
          "sha1": "9c938b57af4e",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 26,
          "bytes": 41655,
          "sha1": "d55015841d00",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 27,
          "bytes": 40681,
          "sha1": "15359c97c9fe",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 28,
          "bytes": 53760,
          "sha1": "e45af6e792dc",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 29,
          "bytes": 56832,
          "sha1": "b4cb841a1ec3",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 30,
          "bytes": 48126,
          "sha1": "8c4af93c93eb",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 31,
          "bytes": 52620,
          "sha1": "aabae72777ef",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 32,
          "bytes": 46411,
          "sha1": "97be65a9e68c",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 33,
          "bytes": 40772,
          "sha1": "7f28bf42c5ce",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 34,
          "bytes": 47733,
          "sha1": "44d2a0882ce9",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 35,
          "bytes": 52336,
          "sha1": "ad104776889d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 36,
          "bytes": 40127,
          "sha1": "15b6973d584d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 37,
          "bytes": 49146,
          "sha1": "4ccb1703a898",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 38,
          "bytes": 49975,
          "sha1": "2e5e995f8ddf",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 39,
          "bytes": 48336,
          "sha1": "cc6d8ab49859",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 40,
          "bytes": 47942,
          "sha1": "ec1ad8f8d78e",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 41,
          "bytes": 48260,
          "sha1": "5a74c3bd73d4",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 42,
          "bytes": 31672,
          "sha1": "4790264ef1ec",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 43,
          "bytes": 38795,
          "sha1": "0e751b2287f1",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 44,
          "bytes": 48359,
          "sha1": "e191ed2bd448",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 45,
          "bytes": 43018,
          "sha1": "fef6f62b25c7",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 46,
          "bytes": 45834,
          "sha1": "e6199289a4ca",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 47,
          "bytes": 45735,
          "sha1": "9e6742814c7d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 48,
          "bytes": 42654,
          "sha1": "417c58c788cc",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 49,
          "bytes": 43293,
          "sha1": "e1ec180e83ae",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 50,
          "bytes": 38952,
          "sha1": "3afab2a2c95c",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 51,
          "bytes": 42032,
          "sha1": "f76dde7b04af",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 52,
          "bytes": 44395,
          "sha1": "2e9dce19771a",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 53,
          "bytes": 45971,
          "sha1": "30f1672f280d",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 54,
          "bytes": 43668,
          "sha1": "5f92fcb7d3c1",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 55,
          "bytes": 52522,
          "sha1": "2f39efcd8cfc",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 56,
          "bytes": 37464,
          "sha1": "d50e7c9484bd",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 57,
          "bytes": 39013,
          "sha1": "d1c425877470",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 58,
          "bytes": 34499,
          "sha1": "cb9fcf1b2700",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 59,
          "bytes": 42666,
          "sha1": "e3c2c4663d22",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 60,
          "bytes": 44299,
          "sha1": "aeac7d6c49f4",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 61,
          "bytes": 41687,
          "sha1": "36270baeb39e",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 62,
          "bytes": 31589,
          "sha1": "bed8e6c71296",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        },
        {
          "index": 63,
          "bytes": 34981,
          "sha1": "ba336546a25e",
          "format": "JPEG",
          "size": [
            480,
            360
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_3",
      "text": "D. C is cleaning dishes.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_0",
      "text": "A. C is cooking.",
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
      "c_3"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 5,
    "negative_candidate_count": 4
  }
}
```

#### 示例 2
`query_id=q_1`，`positive_candidate_id=c_9`，`negative_candidate_id=c_5`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?A. To provide a source of water for the paintbrush. B. To provide a place to store the paintbrush. C. To provide a place to dispose of the paintbrush. D. To provide a place to rest the paintbrush. E. To clean the paintbrush.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 72189,
          "sha1": "1c5cf38cee38",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 1,
          "bytes": 73131,
          "sha1": "16133e0ad820",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 2,
          "bytes": 72147,
          "sha1": "3a2f57a9a9a7",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 3,
          "bytes": 53465,
          "sha1": "765c8ad6feca",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 4,
          "bytes": 75405,
          "sha1": "7c4f5f97e074",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 5,
          "bytes": 68208,
          "sha1": "19bcdd03f8c4",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 6,
          "bytes": 65746,
          "sha1": "d257b9637592",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 7,
          "bytes": 60270,
          "sha1": "c7ce581fe043",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 8,
          "bytes": 59213,
          "sha1": "899c66a49984",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 9,
          "bytes": 60121,
          "sha1": "483d7d7a4c73",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 10,
          "bytes": 60486,
          "sha1": "6201df4259cb",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 11,
          "bytes": 58909,
          "sha1": "95ba7e9c023f",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 12,
          "bytes": 67523,
          "sha1": "32e96b9e5a96",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 13,
          "bytes": 70864,
          "sha1": "0db7088835bc",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 14,
          "bytes": 70394,
          "sha1": "daf217bf0b0e",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 15,
          "bytes": 70047,
          "sha1": "e79eb08dca96",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 16,
          "bytes": 70287,
          "sha1": "b7ba3541f23f",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 17,
          "bytes": 67769,
          "sha1": "59efd56fd5fd",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 18,
          "bytes": 60815,
          "sha1": "e7579c750652",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 19,
          "bytes": 62426,
          "sha1": "270967db314f",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 20,
          "bytes": 58851,
          "sha1": "5e5d178152a8",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 21,
          "bytes": 69148,
          "sha1": "c24dedb677b5",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 22,
          "bytes": 73411,
          "sha1": "358cb2fefadc",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 23,
          "bytes": 72898,
          "sha1": "7b2b52c41055",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 24,
          "bytes": 73320,
          "sha1": "f48480a69586",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 25,
          "bytes": 74053,
          "sha1": "37e39cc373fc",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 26,
          "bytes": 72811,
          "sha1": "433dbb4691d7",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 27,
          "bytes": 58603,
          "sha1": "6bb79982bee4",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 28,
          "bytes": 54356,
          "sha1": "eb026f6b327c",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 29,
          "bytes": 66296,
          "sha1": "11785164b58c",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 30,
          "bytes": 70294,
          "sha1": "8dd2153363f1",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 31,
          "bytes": 65662,
          "sha1": "0b3cdea5e9d2",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 32,
          "bytes": 71439,
          "sha1": "841fe6ce2f38",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 33,
          "bytes": 65295,
          "sha1": "37a36a397b11",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 34,
          "bytes": 74257,
          "sha1": "537dbb3a0c37",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 35,
          "bytes": 63032,
          "sha1": "f87c4680f122",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 36,
          "bytes": 75601,
          "sha1": "d08155f1a3f8",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 37,
          "bytes": 53822,
          "sha1": "a9364427bad6",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 38,
          "bytes": 74782,
          "sha1": "9288c009fb70",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 39,
          "bytes": 69027,
          "sha1": "614312d4de25",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 40,
          "bytes": 66587,
          "sha1": "6a6139bb6ccf",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 41,
          "bytes": 73174,
          "sha1": "d0ecfcb6f8c7",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 42,
          "bytes": 67298,
          "sha1": "9e4f4d674cfc",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 43,
          "bytes": 73482,
          "sha1": "75e27a5e8506",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 44,
          "bytes": 72580,
          "sha1": "1f842af7b199",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 45,
          "bytes": 66913,
          "sha1": "76fcf072f9f7",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 46,
          "bytes": 63012,
          "sha1": "c64a9468b613",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 47,
          "bytes": 69647,
          "sha1": "32a62f8a1116",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 48,
          "bytes": 70963,
          "sha1": "2825b01f0d2f",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 49,
          "bytes": 67089,
          "sha1": "526c74c6dccc",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 50,
          "bytes": 66524,
          "sha1": "149964a7c238",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 51,
          "bytes": 68671,
          "sha1": "7eb2edd4d7f9",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 52,
          "bytes": 69567,
          "sha1": "d1a5a8ab532c",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 53,
          "bytes": 69179,
          "sha1": "cc34090aab5b",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 54,
          "bytes": 69585,
          "sha1": "d6ab4f0e29e0",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 55,
          "bytes": 57402,
          "sha1": "d31f982df22c",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 56,
          "bytes": 69727,
          "sha1": "f29cf0dd9850",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 57,
          "bytes": 71817,
          "sha1": "f7fbd4b26aa7",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 58,
          "bytes": 71626,
          "sha1": "d155aa627061",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 59,
          "bytes": 64624,
          "sha1": "50e1e9df6e48",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 60,
          "bytes": 69682,
          "sha1": "752cd5580fe1",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 61,
          "bytes": 64067,
          "sha1": "7e240782713b",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 62,
          "bytes": 68579,
          "sha1": "a87cdc1bd54c",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        },
        {
          "index": 63,
          "bytes": 57232,
          "sha1": "88b1e7397426",
          "format": "JPEG",
          "size": [
            640,
            360
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_9",
      "text": "E. To clean the paintbrush.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_5",
      "text": "A. To provide a source of water for the paintbrush.",
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
      "c_9"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 5,
    "negative_candidate_count": 4
  }
}
```
