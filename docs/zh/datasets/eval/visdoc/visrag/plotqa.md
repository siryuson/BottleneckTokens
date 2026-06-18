# VisRAG PlotQA

> 面向统计图与曲线图问答的视觉检索子集。

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
| HuggingFace | [openbmb/VisRAG-Ret-Test-PlotQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-PlotQA) |
| 分割 | `test` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（Yu et al., 2024）([arXiv](https://arxiv.org/abs/2410.10594)) |

### Schema
- `queries(query-id, query)`
- `corpus(corpus-id, image)`
- `qrels(query-id, corpus-id, score)`

### 样本

**query.text**: `Find a document image that matches the given query: Across all years, what is the minimum percentage...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|> Understand the content of the provided document image....`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/VisRAG_PlotQA/` |
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
`query_id=8111.png-1`，`positive_candidate_id=8111.png`

```json
{
  "queries.lance": {
    "id": "8111.png-1",
    "text": "Find a document image that matches the given query: Across all years, what is the minimum percentage of unemployed female labor force in Europe(all income levels) ?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "8111.png",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 30510,
            "sha1": "91e69b3f3181",
            "format": "PNG",
            "size": [
              1083,
              700
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "8111.png-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "8111.png"
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
`query_id=21444.png-1`，`positive_candidate_id=21444.png`

```json
{
  "queries.lance": {
    "id": "21444.png-1",
    "text": "Find a document image that matches the given query: Is the depth of food deficit in 2003 less than that in 2005 ?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "21444.png",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 35000,
            "sha1": "1c79d36d4624",
            "format": "PNG",
            "size": [
              983,
              650
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "21444.png-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "21444.png"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
