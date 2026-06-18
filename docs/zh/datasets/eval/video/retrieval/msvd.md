# MSVD

> 早期经典视频描述与检索数据集，基于多描述并行标注构建。

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
| 原始名称 | MSVD（Microsoft Video Description Corpus） |
| 论文 | Collecting Highly Parallel Data for Paraphrase Evaluation（Chen & Dolan, 2011）([ACL Anthology](https://aclanthology.org/P11-1020/)) |
| HuggingFace | [siyrus/BToks-MSVD](https://huggingface.co/datasets/siyrus/BToks-MSVD) |
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
| HuggingFace | [siyrus/BToks-MSVD](https://huggingface.co/datasets/siyrus/BToks-MSVD) |
| 分割 | `test` |

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
| 目录 | `MMEB-V2/video-tasks/MSVD/` |
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
    "text": "Find the video snippet that corresponds to the given summary: two young men are playing table tennis\n",
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
            "bytes": 14503,
            "sha1": "5fe3ea8a459a",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 8562,
            "sha1": "5b7bc686ad4d",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 7795,
            "sha1": "5b8f98ba1a69",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 8014,
            "sha1": "70fe6028dba1",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 8070,
            "sha1": "e0ccb1d936ec",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 7932,
            "sha1": "c81173ebac9d",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 8011,
            "sha1": "503a09304f44",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 6852,
            "sha1": "64009cc0eb07",
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
    "text": "Find the video snippet that corresponds to the given summary: a man rides his horse in the dessert\n",
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
            "bytes": 22042,
            "sha1": "d82484bcfa2c",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 9181,
            "sha1": "4441cf9c6211",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 7925,
            "sha1": "7826eead9a53",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 8413,
            "sha1": "9ceff6ed8b17",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 7702,
            "sha1": "a102816c0dd2",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 8104,
            "sha1": "11446652c067",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 7615,
            "sha1": "008b5d5e7175",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 8532,
            "sha1": "02789bd1d01c",
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
