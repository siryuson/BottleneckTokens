# VisRAG ChartQA

> VisRAG 图表问答检索子集，强调图表视觉证据检索能力。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `visdoc_visrag` |
| 用途 | 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [openbmb/VisRAG-Ret-Test-ChartQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ChartQA) |
| 分割 | `test` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（Yu et al., 2024）([arXiv](https://arxiv.org/abs/2410.10594)) |

### Schema
- `queries(query-id, query)`
- `corpus(corpus-id, image)`
- `qrels(query-id, corpus-id, score)`

### 样本

**query.text**: `Find a document image that matches the given query: How many more people felt inspired frequently th...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|> Understand the content of the provided document image....`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/VisRAG_ChartQA/` |
| task_type | `visdoc_visrag` |
| 聚合指标 | `ndcg@5` |

### Lance 布局
- `queries.lance`
- `candidates.lance`
- `qrels.lance`

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=3960.png-2`，`positive_candidate_id=3960.png`

```json
{
  "queries.lance": {
    "id": "3960.png-2",
    "text": "Find a document image that matches the given query: How many more people felt inspired frequently than depressed frequently?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "3960.png",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 154758,
            "sha1": "04c791c73c55",
            "format": "PNG",
            "size": [
              840,
              788
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "3960.png-2",
    "mode": "sparse",
    "positive_candidate_ids": [
      "3960.png"
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
`query_id=13750.png-1`，`positive_candidate_id=13750.png`

```json
{
  "queries.lance": {
    "id": "13750.png-1",
    "text": "Find a document image that matches the given query: What's the percentage of U.S adults who refused?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "13750.png",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 21012,
            "sha1": "814b24eb74db",
            "format": "PNG",
            "size": [
              460,
              310
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "13750.png-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "13750.png"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
