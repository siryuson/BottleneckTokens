# Something-Something V2

> 聚焦细粒度人-物交互与时序关系理解的视频动作识别数据集。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_classification` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Something-Something V2 |
| 论文 | The "Something Something" Video Database for Learning and Evaluating Visual Common Sense（Goyal et al., 2017）([arXiv](https://arxiv.org/abs/1706.04261)) |
| HuggingFace | [siyrus/BToks-SmthSmthV2](https://huggingface.co/datasets/siyrus/BToks-SmthSmthV2) |
| 许可证 | 参见原始数据集 |

### Schema
- `video_id`: 视频唯一标识
- `pos_text`: 正样本动作标签文本
- `neg_text`: 与当前样本相关的负标签集合（用于局部穷举评测）

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-SmthSmthV2](https://huggingface.co/datasets/siyrus/BToks-SmthSmthV2) |
| 分割 | `test` |

### Schema
- `video_id`: 视频 ID
- `pos_text`: 正确类别标签
- `neg_text`: 每条查询对应的候选负标签

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/SmthSmthV2/` |
| task_type | `video_classification` |
| 评测模式 | local |

### Lance 布局
- `queries.lance`: 视频帧查询（`id`, `text`, `images`）
- `candidates.lance`: 全量类别候选（`id`, `text`, `images=[]`）
- `qrels.lance`: 穷举相关性标注（`mode=exhaustive`，含正负候选分数）

### 样本

以下示例基于当前 runtime canonical 输出生成，数据来源为 `./data/converted`.

- `text` 展示当前 runtime canonical 的模型可见文本；parser 可能在 Lance 原始字段基础上重组 visual token 布局、选项块和尾部换行。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=holding pen`，`negative_candidate_id=approaching adaptor with your camera`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nWhat actions or object interactions are being performed by the person in the video?\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 21721,
          "sha1": "66c37d798d84",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 20290,
          "sha1": "2577638a574a",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 19781,
          "sha1": "1fe5c47600ed",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 19420,
          "sha1": "06fb82ed5381",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 19559,
          "sha1": "0944b39b3be4",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 19809,
          "sha1": "6b644a369836",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 20042,
          "sha1": "a97da6632484",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 22497,
          "sha1": "3a38e9a98a56",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 19810,
          "sha1": "721ba2c4c28c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 19533,
          "sha1": "0612616697f8",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 19488,
          "sha1": "85d0b9417e73",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 19512,
          "sha1": "a38b5c535446",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 19749,
          "sha1": "57f8f8b4cb97",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 20171,
          "sha1": "4f276769b29c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 22650,
          "sha1": "6580e45a1b6e",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 20912,
          "sha1": "48e871f0d65c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 20429,
          "sha1": "c1896e510a4b",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 20392,
          "sha1": "29b28f66f13d",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 20592,
          "sha1": "696ac24cfadc",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 20705,
          "sha1": "131524d03e08",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 23663,
          "sha1": "097623df6606",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 21214,
          "sha1": "0f0338760852",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 20827,
          "sha1": "04eee21ef865",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 20941,
          "sha1": "4902fde4aee5",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 20625,
          "sha1": "1b9e822dc5bc",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 20871,
          "sha1": "77d94622ad39",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 23343,
          "sha1": "aa207ef6d256",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 20877,
          "sha1": "e4be5dea9850",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 20673,
          "sha1": "7c05c7536f88",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 20635,
          "sha1": "e4674d37a785",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 20506,
          "sha1": "c076af0f62f4",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 20898,
          "sha1": "ce5c6052d15f",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 23581,
          "sha1": "11538ca999b8",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 21210,
          "sha1": "b4983794d4f5",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 20887,
          "sha1": "9126c35547d7",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 35,
          "bytes": 20870,
          "sha1": "b332b9c77947",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 36,
          "bytes": 20837,
          "sha1": "c2060b5bb991",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 37,
          "bytes": 21191,
          "sha1": "b4e008ae0b7c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 38,
          "bytes": 23454,
          "sha1": "1b65fa38754f",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 39,
          "bytes": 21233,
          "sha1": "9afbf125abf0",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 40,
          "bytes": 20989,
          "sha1": "ab7b7a289c2e",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 41,
          "bytes": 20666,
          "sha1": "00888828dce0",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 42,
          "bytes": 20592,
          "sha1": "133c0607f56b",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 43,
          "bytes": 20504,
          "sha1": "93194d992ac8",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 44,
          "bytes": 23487,
          "sha1": "fdeb84de59b0",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 45,
          "bytes": 21180,
          "sha1": "61fdd6eec1d2",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 46,
          "bytes": 20983,
          "sha1": "18e4eef6c417",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 47,
          "bytes": 20597,
          "sha1": "6e77c36c9267",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 48,
          "bytes": 20593,
          "sha1": "a7d0c385ac1c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 49,
          "bytes": 20730,
          "sha1": "a19de5566a3f",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 50,
          "bytes": 23429,
          "sha1": "28567a1e2aca",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 51,
          "bytes": 20755,
          "sha1": "b7b7d1d837c8",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 52,
          "bytes": 20443,
          "sha1": "2e88b4fb7f7d",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 53,
          "bytes": 20189,
          "sha1": "c42d28549dff",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 54,
          "bytes": 20381,
          "sha1": "51024a038e71",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 55,
          "bytes": 20555,
          "sha1": "d3543638e723",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 56,
          "bytes": 23411,
          "sha1": "a70f48851081",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 57,
          "bytes": 20806,
          "sha1": "61dce6c5a060",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 58,
          "bytes": 20493,
          "sha1": "e39bc30c4e01",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 59,
          "bytes": 20407,
          "sha1": "3bffa270e2dc",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 60,
          "bytes": 20304,
          "sha1": "b2633af4773b",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 61,
          "bytes": 20440,
          "sha1": "4099980311fa",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 62,
          "bytes": 23257,
          "sha1": "bcc6100760a0",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 63,
          "bytes": 20556,
          "sha1": "5e39509f4057",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "holding pen",
      "text": "holding pen",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "approaching adaptor with your camera",
      "text": "approaching adaptor with your camera",
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
      "holding pen"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 174,
    "negative_candidate_count": 173
  }
}
```

#### 示例 2
`query_id=q_333`，`positive_candidate_id=putting 2 bananas onto box`，`negative_candidate_id=approaching wardrobe with your camera`

```json
{
  "queries.lance": {
    "id": "q_333",
    "text": "<|video_pad|>\nWhat actions or object interactions are being performed by the person in the video?\n",
    "images": {
      "count": 35,
      "items": [
        {
          "index": 0,
          "bytes": 22817,
          "sha1": "73a7dd0387e2",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 22393,
          "sha1": "57dae092a69e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 22264,
          "sha1": "15d228a6981e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 22328,
          "sha1": "7d8e8f04ec9b",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 21717,
          "sha1": "8263d5e6897e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 21624,
          "sha1": "a298dddfcc97",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 21291,
          "sha1": "11594c32f673",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 22622,
          "sha1": "13f4e1917bc6",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 22834,
          "sha1": "a996576faafe",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 22925,
          "sha1": "1ed6779c21c4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 23575,
          "sha1": "ff2810dedd0a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 23896,
          "sha1": "326537f59e25",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 24357,
          "sha1": "440386cfae91",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 26783,
          "sha1": "11ffca881130",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 26610,
          "sha1": "342c45a552e5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 26826,
          "sha1": "d2d09bf063e4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 26943,
          "sha1": "7a7516dd6d7c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 26612,
          "sha1": "cc5b36d70f37",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 26229,
          "sha1": "7db628f109e0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 25744,
          "sha1": "8669b36e6592",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 24047,
          "sha1": "60a78e15309f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 22279,
          "sha1": "307ee2f1bd63",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 22462,
          "sha1": "60714d22f49a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 23164,
          "sha1": "31427d2e03ce",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 23473,
          "sha1": "44b7bb3dca34",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 24177,
          "sha1": "d491bfa8f991",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 23484,
          "sha1": "bb6d43ccf1b9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 22423,
          "sha1": "c2ac2fc19deb",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 25120,
          "sha1": "1372c1d6be9a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 26015,
          "sha1": "dcb012149632",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 25982,
          "sha1": "41ee70b103bb",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 26113,
          "sha1": "d01417cdbe4d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 26577,
          "sha1": "fd6c783430cf",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 26665,
          "sha1": "57de3fc8ed35",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 25707,
          "sha1": "0c6bef6acaca",
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
      "id": "putting 2 bananas onto box",
      "text": "putting 2 bananas onto box",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "approaching wardrobe with your camera",
      "text": "approaching wardrobe with your camera",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_333",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "putting 2 bananas onto box"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 174,
    "negative_candidate_count": 173
  }
}
```

#### 示例 3
`query_id=q_666`，`positive_candidate_id=holding pc mouse`，`negative_candidate_id=approaching calculator with your camera`

```json
{
  "queries.lance": {
    "id": "q_666",
    "text": "<|video_pad|>\nWhat actions or object interactions are being performed by the person in the video?\n",
    "images": {
      "count": 57,
      "items": [
        {
          "index": 0,
          "bytes": 25439,
          "sha1": "1579d6d59fb2",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 25821,
          "sha1": "a5e89190dcb4",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 25805,
          "sha1": "8bee848e2fc9",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 25546,
          "sha1": "f9f81d410be0",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 25563,
          "sha1": "4759b8932796",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 25496,
          "sha1": "07c15081b6a9",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 26023,
          "sha1": "21c2f834a024",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 26764,
          "sha1": "c9eab311dc4c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 26701,
          "sha1": "5101e05fefaf",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 26316,
          "sha1": "9a80bd8b43f1",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 24637,
          "sha1": "bb5c3422c418",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 24250,
          "sha1": "d819b916d030",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 24099,
          "sha1": "4ba358ccacc2",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 24020,
          "sha1": "c03169de1c7e",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 23894,
          "sha1": "88d76b081c40",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 24569,
          "sha1": "d2908c789d02",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 24644,
          "sha1": "69a0425f7373",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 24219,
          "sha1": "50e51f5c39ef",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 23754,
          "sha1": "85f773770648",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 22808,
          "sha1": "a79938411b48",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 23626,
          "sha1": "89c11a7d8ae6",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 23735,
          "sha1": "0bed025964f1",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 23682,
          "sha1": "f67380d91b5d",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 23481,
          "sha1": "6099252331d8",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 23273,
          "sha1": "413273c79218",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 23788,
          "sha1": "0370b6a5e4fc",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 23929,
          "sha1": "92f304aed7c4",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 23761,
          "sha1": "c43f441ca6bc",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 22989,
          "sha1": "42d81aaa2fed",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 22862,
          "sha1": "3620ba25a5f7",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 23097,
          "sha1": "ade7a504003a",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 23480,
          "sha1": "cd567a622696",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 23657,
          "sha1": "8f356b0b4b8f",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 23808,
          "sha1": "beeb958e9970",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 23420,
          "sha1": "aaac8abcad90",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 35,
          "bytes": 23321,
          "sha1": "e0797e401c00",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 36,
          "bytes": 23589,
          "sha1": "2a7ebdd53524",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 37,
          "bytes": 23719,
          "sha1": "6c43edbae582",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 38,
          "bytes": 23883,
          "sha1": "e0e7654f7d50",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 39,
          "bytes": 24063,
          "sha1": "3c706d9bb1cf",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 40,
          "bytes": 24256,
          "sha1": "9d52ca97a181",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 41,
          "bytes": 24705,
          "sha1": "01ce02d3a126",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 42,
          "bytes": 24585,
          "sha1": "ab618b496780",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 43,
          "bytes": 24625,
          "sha1": "a4c96122e49c",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 44,
          "bytes": 25821,
          "sha1": "25761db7f0fa",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 45,
          "bytes": 25491,
          "sha1": "8f679cfac4cb",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 46,
          "bytes": 25476,
          "sha1": "d9db285071e3",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 47,
          "bytes": 25143,
          "sha1": "01cb8ed40f05",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 48,
          "bytes": 25170,
          "sha1": "ff746c77c36b",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 49,
          "bytes": 25058,
          "sha1": "c04d4a9fab64",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 50,
          "bytes": 26088,
          "sha1": "aa15dd248e2b",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 51,
          "bytes": 25500,
          "sha1": "22845d9e355e",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 52,
          "bytes": 25826,
          "sha1": "ff6152260ba8",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 53,
          "bytes": 25227,
          "sha1": "6fc199d3c0f0",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 54,
          "bytes": 24867,
          "sha1": "60b68fa3742f",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 55,
          "bytes": 24529,
          "sha1": "520cc2e4961b",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        },
        {
          "index": 56,
          "bytes": 25852,
          "sha1": "96302f9c4288",
          "format": "JPEG",
          "size": [
            427,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "holding pc mouse",
      "text": "holding pc mouse",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "approaching calculator with your camera",
      "text": "approaching calculator with your camera",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_666",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "holding pc mouse"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 174,
    "negative_candidate_count": 173
  }
}
```
