# ActivityNet-QA

> 基于复杂网络视频的大规模人工标注视频问答数据集。

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
| 原始名称 | ActivityNet-QA |
| 论文 | ActivityNet-QA: A Dataset for Understanding Complex Web Videos via Question Answering（Yu et al., 2019）([arXiv](https://arxiv.org/abs/1906.02467)) |
| HuggingFace | [siyrus/BToks-ActivityNetQA](https://huggingface.co/datasets/siyrus/BToks-ActivityNetQA) |
| 许可证 | 参见原始数据集 |

### Schema
- `video_name`/`video_id`: 视频标识
- `question`: 问题
- `answer`: 参考答案

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-ActivityNetQA](https://huggingface.co/datasets/siyrus/BToks-ActivityNetQA) |
| 分割 | `test` |

### Schema
- `video_name`/`video_id`: 视频 ID
- `question`: 问题文本
- `answer`: 正确答案文本

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/ActivityNetQA/` |
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
`query_id=q_0`，`positive_candidate_id=c_0`，`negative_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  does the girl in black clothes have long hair? (A) yes; (B) no.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 18087,
          "sha1": "3d795907288b",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 1,
          "bytes": 17475,
          "sha1": "c13d51b38256",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 2,
          "bytes": 17585,
          "sha1": "df3a7e0f1681",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 3,
          "bytes": 17990,
          "sha1": "3906a018fc00",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 4,
          "bytes": 18096,
          "sha1": "42d20c6b74e2",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 5,
          "bytes": 17792,
          "sha1": "ce2f0e567054",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 6,
          "bytes": 17125,
          "sha1": "4d6e5d05bd14",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 7,
          "bytes": 17552,
          "sha1": "16b8778dee59",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 8,
          "bytes": 17692,
          "sha1": "28096c3058d3",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 9,
          "bytes": 17682,
          "sha1": "12f9212cefb0",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 10,
          "bytes": 17327,
          "sha1": "46963eaa1f03",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 11,
          "bytes": 16907,
          "sha1": "15c928e5729c",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 12,
          "bytes": 17299,
          "sha1": "a75efe8bfda6",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 13,
          "bytes": 17059,
          "sha1": "a293d3a88d01",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 14,
          "bytes": 16915,
          "sha1": "997d62a64f57",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 15,
          "bytes": 16026,
          "sha1": "922909d16934",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 16,
          "bytes": 15941,
          "sha1": "a20102899260",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 17,
          "bytes": 16212,
          "sha1": "6558b4515664",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 18,
          "bytes": 17118,
          "sha1": "731001cc47d5",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 19,
          "bytes": 16578,
          "sha1": "6b69042babc9",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 20,
          "bytes": 16549,
          "sha1": "7b862d19bf38",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 21,
          "bytes": 16396,
          "sha1": "017059fdfe90",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 22,
          "bytes": 16713,
          "sha1": "2fd89153f3fc",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 23,
          "bytes": 17037,
          "sha1": "93c0e91d5bbb",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 24,
          "bytes": 17044,
          "sha1": "3816f296dea9",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 25,
          "bytes": 16775,
          "sha1": "dbb7e379595b",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 26,
          "bytes": 16639,
          "sha1": "d1823822521e",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 27,
          "bytes": 16954,
          "sha1": "b2cb7a82a601",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 28,
          "bytes": 17143,
          "sha1": "ddd05dfbf853",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 29,
          "bytes": 17182,
          "sha1": "177afdcd7b38",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 30,
          "bytes": 17238,
          "sha1": "469de30f6268",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 31,
          "bytes": 17214,
          "sha1": "ce420c571af1",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 32,
          "bytes": 16903,
          "sha1": "8900a50d3366",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 33,
          "bytes": 16964,
          "sha1": "a94d6ab264d6",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 34,
          "bytes": 16654,
          "sha1": "a8d5c7ce7a24",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 35,
          "bytes": 16442,
          "sha1": "06e9ec2cf26e",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 36,
          "bytes": 16980,
          "sha1": "51d7a71bfe26",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 37,
          "bytes": 17000,
          "sha1": "165233f02364",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 38,
          "bytes": 17322,
          "sha1": "94e89c9ffd7d",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 39,
          "bytes": 16846,
          "sha1": "90c6a04bdd35",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 40,
          "bytes": 16832,
          "sha1": "6521a672a02b",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 41,
          "bytes": 16773,
          "sha1": "27e5930ca967",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 42,
          "bytes": 16856,
          "sha1": "9ec7d9cd78cd",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 43,
          "bytes": 16657,
          "sha1": "d4de7157c6f4",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 44,
          "bytes": 17010,
          "sha1": "56f638c39360",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 45,
          "bytes": 16363,
          "sha1": "63764c1fb09d",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 46,
          "bytes": 16285,
          "sha1": "0017f4002a13",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 47,
          "bytes": 16980,
          "sha1": "cb41015be510",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 48,
          "bytes": 16463,
          "sha1": "df3713164eda",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 49,
          "bytes": 16432,
          "sha1": "b1afa07abc08",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 50,
          "bytes": 16771,
          "sha1": "589e8e52ef63",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 51,
          "bytes": 17104,
          "sha1": "374ec4dc0ebf",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 52,
          "bytes": 17495,
          "sha1": "91b578cdb380",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 53,
          "bytes": 17497,
          "sha1": "a7f9b9db2ec5",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 54,
          "bytes": 17179,
          "sha1": "b6bc31d265f9",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 55,
          "bytes": 15660,
          "sha1": "13dff50ef5a6",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 56,
          "bytes": 15073,
          "sha1": "ffcf773d55ca",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 57,
          "bytes": 14742,
          "sha1": "99f418718b58",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 58,
          "bytes": 14890,
          "sha1": "4e16881ba3ee",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 59,
          "bytes": 15856,
          "sha1": "ea764a9138d8",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 60,
          "bytes": 15765,
          "sha1": "f1b891eb0c96",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 61,
          "bytes": 14891,
          "sha1": "9fe32ed8f742",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 62,
          "bytes": 14716,
          "sha1": "93679ab58e42",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        },
        {
          "index": 63,
          "bytes": 14112,
          "sha1": "ed0e2fa41f51",
          "format": "JPEG",
          "size": [
            270,
            360
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "yes",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1",
      "text": "no",
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
      "c_0"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 2,
    "negative_candidate_count": 1
  }
}
```

#### 示例 2
`query_id=q_1`，`positive_candidate_id=c_3`，`negative_candidate_id=c_2`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:  is the man in black outdoors? (A) yes; (B) no.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 3820,
          "sha1": "9823f6ba333c",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 1,
          "bytes": 35512,
          "sha1": "f05f48851257",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 2,
          "bytes": 7072,
          "sha1": "daff992db99c",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 3,
          "bytes": 7737,
          "sha1": "6b4e45c9a1a7",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 4,
          "bytes": 74739,
          "sha1": "3eb6ddb999eb",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 5,
          "bytes": 76032,
          "sha1": "e4c9f6257ca1",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 6,
          "bytes": 73092,
          "sha1": "9c782a8cfc93",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 7,
          "bytes": 61432,
          "sha1": "7217c7d25aee",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 8,
          "bytes": 11081,
          "sha1": "93ac3db3712e",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 9,
          "bytes": 70721,
          "sha1": "30e6915ae972",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 10,
          "bytes": 74387,
          "sha1": "ddcfed3795cd",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 11,
          "bytes": 73269,
          "sha1": "3b3a89e1a593",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 12,
          "bytes": 76723,
          "sha1": "ed11ebaa0582",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 13,
          "bytes": 69567,
          "sha1": "22febf2f4a82",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 14,
          "bytes": 66193,
          "sha1": "e0bb2d8e1d6a",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 15,
          "bytes": 64499,
          "sha1": "432eef96a6f1",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 16,
          "bytes": 66959,
          "sha1": "a7372461a3e1",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 17,
          "bytes": 65085,
          "sha1": "460cb76e52e5",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 18,
          "bytes": 74328,
          "sha1": "f5de4a58ecea",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 19,
          "bytes": 72615,
          "sha1": "5f38d312759c",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 20,
          "bytes": 76496,
          "sha1": "7eeb99926461",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 21,
          "bytes": 78332,
          "sha1": "2995c8ea9ca1",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 22,
          "bytes": 71205,
          "sha1": "7f8633b37cac",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 23,
          "bytes": 75536,
          "sha1": "674ac73432a2",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 24,
          "bytes": 76673,
          "sha1": "5c16e46672e4",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 25,
          "bytes": 70086,
          "sha1": "7ec229f097dc",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 26,
          "bytes": 73424,
          "sha1": "fc770ca8c4e3",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 27,
          "bytes": 71995,
          "sha1": "ecad8612e20c",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 28,
          "bytes": 68164,
          "sha1": "25b5649286f7",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 29,
          "bytes": 68406,
          "sha1": "88fd0c5bcc71",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 30,
          "bytes": 70370,
          "sha1": "956224170a97",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 31,
          "bytes": 68029,
          "sha1": "e78e87af431b",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 32,
          "bytes": 71448,
          "sha1": "a4ed89ae2cab",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 33,
          "bytes": 71198,
          "sha1": "073e238cb4d1",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 34,
          "bytes": 72966,
          "sha1": "b7256ed521db",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 35,
          "bytes": 77562,
          "sha1": "2070681d8cf5",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 36,
          "bytes": 76222,
          "sha1": "ba58117f8ed4",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 37,
          "bytes": 74233,
          "sha1": "cf72da2eb0f9",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 38,
          "bytes": 70538,
          "sha1": "5f528d3e4c51",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 39,
          "bytes": 58394,
          "sha1": "ccfd9ef1b2f8",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 40,
          "bytes": 55344,
          "sha1": "ed249b264c0d",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 41,
          "bytes": 68639,
          "sha1": "f7265da194df",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 42,
          "bytes": 67485,
          "sha1": "8de874dc541e",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 43,
          "bytes": 62507,
          "sha1": "4721dc1d89d8",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 44,
          "bytes": 66165,
          "sha1": "266652e9b590",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 45,
          "bytes": 68612,
          "sha1": "351bb2bd8f4d",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 46,
          "bytes": 41484,
          "sha1": "567a81b81881",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 47,
          "bytes": 49092,
          "sha1": "71141d1bf760",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 48,
          "bytes": 51486,
          "sha1": "63f3c8a15337",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 49,
          "bytes": 59116,
          "sha1": "e511f5772357",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 50,
          "bytes": 69906,
          "sha1": "609503b2c2ec",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 51,
          "bytes": 69602,
          "sha1": "e104aa85462e",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 52,
          "bytes": 71934,
          "sha1": "9ea5fd4fa916",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 53,
          "bytes": 69806,
          "sha1": "1fd8d4a9b5f1",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 54,
          "bytes": 13604,
          "sha1": "1443b6216543",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 55,
          "bytes": 39784,
          "sha1": "4a0df7858b02",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 56,
          "bytes": 52736,
          "sha1": "ffc9c5c07636",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 57,
          "bytes": 55319,
          "sha1": "883cc591e614",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 58,
          "bytes": 55617,
          "sha1": "d9c6d7d4acd4",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 59,
          "bytes": 51802,
          "sha1": "64bc283642e8",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 60,
          "bytes": 53328,
          "sha1": "c7dd1655776b",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 61,
          "bytes": 54390,
          "sha1": "38bbb188dbb7",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 62,
          "bytes": 51058,
          "sha1": "3d7b1237e1e0",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        },
        {
          "index": 63,
          "bytes": 51411,
          "sha1": "497e68b5fbf4",
          "format": "JPEG",
          "size": [
            544,
            360
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_3",
      "text": "no",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2",
      "text": "yes",
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
      "c_3"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 2,
    "negative_candidate_count": 1
  }
}
```
