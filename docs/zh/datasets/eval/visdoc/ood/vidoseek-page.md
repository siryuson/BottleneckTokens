# ViDoSeek-page

> ViDoSeek 页级 OOD 检索子集，评测视觉文档检索的域外泛化能力。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `visdoc_ood` |
| 用途 | 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-ViDoSeek-page-fixed](https://huggingface.co/datasets/siyrus/BToks-ViDoSeek-page-fixed) |
| 分割 | `test` |
| 论文 | ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents（Wang et al., 2025）([arXiv](https://arxiv.org/abs/2502.18017)) |

### Schema
- 采用 BEIR 风格子集组织：`queries`、`corpus`、`qrels`
- 页级检索目标以 `corpus-id` 对应文档页图像

### 样本

**query.text**: `Apply for Nordic Swan Ecolabel license, what is recommended as a web browser according to the Nordic...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoSeek-page/` |
| task_type | `visdoc_ood` |
| 聚合指标 | `ndcg@5` |

### Lance 布局
- `queries.lance`
- `candidates.lance`
- `qrels.lance`

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。
- `metadata` 继续保留 runtime 侧的辅助字段，例如 `corpus_range` 和候选页 `path`。

#### 示例 1
`query_id=0`，`positive_candidate_id=3`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: Apply for Nordic Swan Ecolabel license, what is recommended as a web browser according to the Nordic Ecolabelling Portal instructions?\n",
    "images": {
      "count": 0,
      "items": []
    },
    "metadata": {
      "corpus_range": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "3",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "metadata": {
        "path": "3.png"
      },
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 131888,
            "sha1": "cfd30c6ec079",
            "format": "PNG",
            "size": [
              960,
              540
            ]
          }
        ]
      },
      "score": 3.0
    }
  },
  "qrels.lance": {
    "query_id": "0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "3"
    ],
    "positive_candidate_scores": [
      3.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=1`，`positive_candidate_id=2`

```json
{
  "queries.lance": {
    "id": "1",
    "text": "Find a document image that matches the given query: What is the first step in the Nordic Swan Ecolabel license application process according to the Application Guide?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "2",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 52271,
            "sha1": "8691e910e2bf",
            "format": "PNG",
            "size": [
              960,
              540
            ]
          }
        ]
      },
      "score": 3.0
    }
  },
  "qrels.lance": {
    "query_id": "1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "2"
    ],
    "positive_candidate_scores": [
      3.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
