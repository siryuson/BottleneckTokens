# VisRAG ArxivQA

> VisRAG 框架下的 ArxivQA 文档检索评测子集。

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
| HuggingFace | [openbmb/VisRAG-Ret-Test-ArxivQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ArxivQA) |
| 分割 | `test` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（Yu et al., 2024）([arXiv](https://arxiv.org/abs/2410.10594)) |

### Schema
- BEIR 风格：`queries`, `corpus`, `qrels`
- `corpus.image` 为文档页图像，`qrels.score` 为相关性

### 样本
> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/VisRAG_ArxivQA/` |
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
`query_id=1508.06771_0.jpg-1`, `positive_candidate_id=1508.06771_0.jpg`

```json
{
  "queries.lance": {
    "id": "1508.06771_0.jpg-1",
    "text": "Find a document image that matches the given query: Which statement best describes the relationship between the components labeled 'Myosin' and 'Actin filament'?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1508.06771_0.jpg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 72710,
            "sha1": "a28a40472caf",
            "format": "JPEG",
            "size": [
              819,
              602
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "1508.06771_0.jpg-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1508.06771_0.jpg"
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
`query_id=2201.06587_2.jpg-1`, `positive_candidate_id=2201.06587_2.jpg`

```json
{
  "queries.lance": {
    "id": "2201.06587_2.jpg-1",
    "text": "Find a document image that matches the given query: What is the approximate value of \\( E_{\\alpha}(k) \\) for \\( k = 0.5 \\)?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "2201.06587_2.jpg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 218330,
            "sha1": "fc3190cc391b",
            "format": "JPEG",
            "size": [
              1947,
              2016
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "2201.06587_2.jpg-1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "2201.06587_2.jpg"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
