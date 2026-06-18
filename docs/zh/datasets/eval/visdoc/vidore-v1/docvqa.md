# ViDoRe DocVQA

> 面向文档问答场景的视觉文档检索子集，强调版面感知与证据定位。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `visdoc_vidore_v1` |
| 用途 | 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [vidore/docvqa_test_subsampled_beir](https://huggingface.co/datasets/vidore/docvqa_test_subsampled_beir) |
| 分割 | `test` |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models（Faysse et al., 2024）([arXiv](https://arxiv.org/abs/2407.01449)) |

### Schema
- BEIR 风格：`queries` / `corpus` / `qrels`
- 关键字段：`query-id`, `query`, `corpus-id`, `image`, `score`

### 样本

**query.text**: `What is the dividend payout in 2012?...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoRe_docvqa/` |
| task_type | `visdoc_vidore_v1` |
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
`query_id=0`，`positive_candidate_id=0`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: What is the dividend payout in 2012?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "0",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 123486,
            "sha1": "d41f7cc9cfe2",
            "format": "PNG",
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
    "query_id": "0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "0"
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
`query_id=75`，`positive_candidate_id=77`

```json
{
  "queries.lance": {
    "id": "75",
    "text": "Find a document image that matches the given query: What is the figure number?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "77",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 1075076,
            "sha1": "4a5b796bca80",
            "format": "PNG",
            "size": [
              2208,
              1692
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "75",
    "mode": "sparse",
    "positive_candidate_ids": [
      "77",
      "450"
    ],
    "positive_candidate_scores": [
      1.0,
      1.0
    ],
    "total_candidate_ids": 2,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=3`，`positive_candidate_id=3`

```json
{
  "queries.lance": {
    "id": "3",
    "text": "Find a document image that matches the given query: What is the table number?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "3",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 1119732,
            "sha1": "006e33b103cb",
            "format": "PNG",
            "size": [
              2286,
              1790
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "3",
    "mode": "sparse",
    "positive_candidate_ids": [
      "3",
      "23",
      "87",
      "427",
      "493"
    ],
    "positive_candidate_scores": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "total_candidate_ids": 5,
    "negative_candidate_count": 0
  }
}
```
