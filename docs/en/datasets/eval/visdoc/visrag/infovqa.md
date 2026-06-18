# VisRAG InfoVQA

> 信息图问答检索子集，评测复杂视觉布局信息抽取前的检索阶段。

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
| HuggingFace | [openbmb/VisRAG-Ret-Test-InfoVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-InfoVQA) |
| 分割 | `test` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（Yu et al., 2024）([arXiv](https://arxiv.org/abs/2410.10594)) |

### Schema
- 标准 BEIR 结构：`queries` / `corpus` / `qrels`
- `score` 表示相关性强度

### 样本
> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/VisRAG_InfoVQA/` |
| task_type | `visdoc_visrag` |
| 聚合指标 | `ndcg@5` |

### Lance 布局
- `queries.lance`
- `candidates.lance`
- `qrels.lance`

### 样本

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=36966.jpeg-1`, `positive_candidate_id=36966.jpeg`

```json
{
  "queries.lance": {
    "id": "36966.jpeg-1",
    "text": "Find a document image that matches the given query: Which social media site was invented in 2002?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "36966.jpeg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 858641,
            "sha1": "93e1a9e60b38",
            "format": "JPEG",
            "size": [
              1200,
              7237
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "36966.jpeg-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "36966.jpeg"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### Example 2
`query_id=36966.jpeg-2`, `positive_candidate_id=36966.jpeg`

```json
{
  "queries.lance": {
    "id": "36966.jpeg-2",
    "text": "Find a document image that matches the given query: Which year was Facebook founded?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "36966.jpeg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 858641,
            "sha1": "93e1a9e60b38",
            "format": "JPEG",
            "size": [
              1200,
              7237
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "36966.jpeg-2",
    "mode": "sparse",
    "positive_candidate_ids": [
      "36966.jpeg"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
