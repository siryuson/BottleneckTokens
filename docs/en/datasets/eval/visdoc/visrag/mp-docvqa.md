# VisRAG MP-DocVQA

> 多页面文档问答检索子集，评测跨页证据召回能力。

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
| HuggingFace | [openbmb/VisRAG-Ret-Test-MP-DocVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-MP-DocVQA) |
| 分割 | `test` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（Yu et al., 2024）([arXiv](https://arxiv.org/abs/2410.10594)) |

### Schema
- BEIR 格式：`queries`、`corpus`、`qrels`
- 页面图像与查询文本通过 `query-id` / `corpus-id` 对齐

### 样本
> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/VisRAG_MP-DocVQA/` |
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
`query_id=txpp0227_p9.jpg-1`, `positive_candidate_id=txpp0227_p9.jpg`

```json
{
  "queries.lance": {
    "id": "txpp0227_p9.jpg-1",
    "text": "Find a document image that matches the given query: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "txpp0227_p9.jpg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 158929,
            "sha1": "6f26810c0216",
            "format": "JPEG",
            "size": [
              850,
              1644
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "txpp0227_p9.jpg-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "txpp0227_p9.jpg"
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
`query_id=snbx0223_p43.jpg-1`, `positive_candidate_id=snbx0223_p43.jpg`

```json
{
  "queries.lance": {
    "id": "snbx0223_p43.jpg-1",
    "text": "Find a document image that matches the given query: How many nomination committee meetings has Y. C. Deveshwar attended?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "snbx0223_p43.jpg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 397264,
            "sha1": "7660e95ba822",
            "format": "JPEG",
            "size": [
              1653,
              2339
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "snbx0223_p43.jpg-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "snbx0223_p43.jpg"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
