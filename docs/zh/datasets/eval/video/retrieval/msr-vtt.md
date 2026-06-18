# MSR-VTT

> 大规模开放域视频-文本检索基准，覆盖丰富场景与描述。

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
| 原始名称 | MSR-VTT |
| 论文 | MSR-VTT: A Large Video Description Dataset for Bridging Video and Language（Xu et al., 2016）([CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html)) |
| HuggingFace | [siyrus/BToks-MSR-VTT](https://huggingface.co/datasets/siyrus/BToks-MSR-VTT) |
| 许可证 | 参见原始数据集 |

### Schema
- `video_id`: 视频标识
- `caption`: 查询描述文本

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MSR-VTT](https://huggingface.co/datasets/siyrus/BToks-MSR-VTT) |
| 分割 | `test`（subset: `test_1k`） |

### Schema
- `video_id`: 视频 ID
- `caption`: 文本查询

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/MSR-VTT/` |
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
    "text": "Find a video that contains the following visual content: a woman creating a fondant baby and flower\n",
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
            "bytes": 12126,
            "sha1": "29da827bf5c8",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 1,
            "bytes": 15466,
            "sha1": "615bd8a88625",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 2,
            "bytes": 17522,
            "sha1": "f1b9c19bb4a7",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 3,
            "bytes": 11633,
            "sha1": "1a7afff0dbfd",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 4,
            "bytes": 9139,
            "sha1": "12e490bb9474",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 5,
            "bytes": 10201,
            "sha1": "dbfc60c73355",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 6,
            "bytes": 10398,
            "sha1": "be2e806d7925",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 7,
            "bytes": 9185,
            "sha1": "b5450ecc0e11",
            "format": "JPEG",
            "size": [
              298,
              224
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
    "text": "Find a video that contains the following visual content: baseball player hits ball\n",
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
            "bytes": 11502,
            "sha1": "c27da12ca45f",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 1,
            "bytes": 21016,
            "sha1": "433c4255cf54",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 2,
            "bytes": 6725,
            "sha1": "680defc550b8",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 3,
            "bytes": 13143,
            "sha1": "2d8a5a253696",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 4,
            "bytes": 12462,
            "sha1": "df98477abfd7",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 5,
            "bytes": 10881,
            "sha1": "d78a3239b244",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 6,
            "bytes": 14361,
            "sha1": "e70a12598135",
            "format": "JPEG",
            "size": [
              298,
              224
            ]
          },
          {
            "index": 7,
            "bytes": 11026,
            "sha1": "a22757d2eb0a",
            "format": "JPEG",
            "size": [
              298,
              224
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
