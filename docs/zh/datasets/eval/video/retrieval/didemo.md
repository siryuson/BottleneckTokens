# DiDeMo

> 面向自然语言时刻定位与视频检索的细粒度描述数据集。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_retrieval` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | DiDeMo |
| 论文 | Localizing Moments in Video with Natural Language（Hendricks et al., 2017）([arXiv](https://arxiv.org/abs/1708.01641)) |
| HuggingFace | [siyrus/BToks-DiDeMo](https://huggingface.co/datasets/siyrus/BToks-DiDeMo) |
| 许可证 | 参见原始数据集 |

### Schema
- `video`/`video_id`: 视频标识
- `caption`: 查询描述文本

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-DiDeMo](https://huggingface.co/datasets/siyrus/BToks-DiDeMo) |
| 分割 | `test` |

### Schema
- `video`/`video_id`: 视频 ID
- `caption`: 文本查询

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/DiDeMo/` |
| task_type | `video_retrieval` |
| 评测模式 | global |

### Lance 布局
- `queries.lance`: 文本查询（`id`, `text`, `images=[]`）
- `candidates.lance`: 视频帧候选（`id`, `text`, `images`）
- `qrels.lance`: 稀疏相关性标注（单正样本为主）

### 样本

以下示例基于当前 runtime canonical 输出生成，数据来源为 `./data/converted`.

- `text` 展示当前 runtime canonical 的模型可见文本；parser 可能在 Lance 原始字段基础上重组 visual token 布局、选项块和尾部换行。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_0`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find a video that includes the following described scenes: we pan right off the mural and on to a building. the view changes from artwork to a building in this shot. the screen moves to something dark the camera pans from the end of the mural to another structure and then a building.\n",
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
            "bytes": 33910,
            "sha1": "81da46599371",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 17108,
            "sha1": "5c5428d9154c",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 15906,
            "sha1": "7f415fb148e6",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 7679,
            "sha1": "f38ac474f70e",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 9994,
            "sha1": "9c056fc80f06",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 4142,
            "sha1": "37be8c16cba8",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 7883,
            "sha1": "fc30e40b5131",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 4048,
            "sha1": "28206e905e4b",
            "format": "JPEG",
            "size": [
              640,
              480
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

#### 示例 2
`query_id=q_1`，`positive_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find a video that includes the following described scenes: the sunset first comes into view. nothing but ocean in these clips first time sun glares on water first time.\n",
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
            "bytes": 23975,
            "sha1": "7de99ba4b5a1",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 8643,
            "sha1": "f3367c726be6",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 6991,
            "sha1": "c56362886634",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 7573,
            "sha1": "09025a0b4d47",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 7139,
            "sha1": "9e2078239104",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 8037,
            "sha1": "8b8c730f03ca",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 9635,
            "sha1": "8609d5301c4e",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 11791,
            "sha1": "e6b70a583563",
            "format": "JPEG",
            "size": [
              640,
              480
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
