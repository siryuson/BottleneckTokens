# YouCook2

> 大规模烹饪步骤视频数据集，支持过程理解与视频检索。

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
| 原始名称 | YouCook2 |
| 论文 | Towards Automatic Learning of Procedures from Web Instructional Videos（Zhou et al., 2018）([arXiv](https://arxiv.org/abs/1703.09788)) |
| HuggingFace | [lmms-lab/YouCook2](https://huggingface.co/datasets/lmms-lab/YouCook2) |
| 许可证 | 参见原始数据集 |

### Schema
- `id`: 视频片段 ID
- `sentence`: 查询描述文本

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [lmms-lab/YouCook2](https://huggingface.co/datasets/lmms-lab/YouCook2) |
| 分割 | `val` |

### Schema
- `id`: 视频 ID
- `sentence`: 文本查询

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/YouCook2/` |
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
    "text": "Find a video that demonstrates the following action while making a recipe: pick the ends off the verdalago\n",
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
            "bytes": 12854,
            "sha1": "cf0a998989ca",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 1,
            "bytes": 6289,
            "sha1": "87984ac67bd7",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 2,
            "bytes": 5536,
            "sha1": "c30d39f30a0b",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 3,
            "bytes": 5529,
            "sha1": "f58264d9ef99",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 4,
            "bytes": 6196,
            "sha1": "a8cd7fdde05c",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 5,
            "bytes": 5818,
            "sha1": "6c550a496e00",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 6,
            "bytes": 5262,
            "sha1": "5b63adfde988",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 7,
            "bytes": 4750,
            "sha1": "296bc0d5e934",
            "format": "JPEG",
            "size": [
              640,
              360
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
    "text": "Find a video that demonstrates the following action while making a recipe: combine lemon juice sumac garlic salt and oil in a bowl\n",
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
            "bytes": 19055,
            "sha1": "73a444282e3a",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 1,
            "bytes": 6016,
            "sha1": "f2a46df1e640",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 2,
            "bytes": 6867,
            "sha1": "4c60b990b5a3",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 3,
            "bytes": 7251,
            "sha1": "5a096cc81007",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 4,
            "bytes": 7029,
            "sha1": "cd7e17d7d07b",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 5,
            "bytes": 6357,
            "sha1": "2446d1bc4418",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 6,
            "bytes": 6784,
            "sha1": "25fba3a20536",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 7,
            "bytes": 6857,
            "sha1": "0b7c385ad32a",
            "format": "JPEG",
            "size": [
              640,
              360
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
