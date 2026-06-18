# ViDoRe ESG Reports Human Labeled V2

> 人工标注 ESG 报告检索子集，强调真实企业报告场景的文档检索。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `visdoc_vidore_v2` |
| 用途 | 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [vidore/esg_reports_human_labeled_v2](https://huggingface.co/datasets/vidore/esg_reports_human_labeled_v2) |
| 分割 | `test` |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models（Faysse et al., 2024）([arXiv](https://arxiv.org/abs/2407.01449)) |

### Schema
- `queries`: 查询文本（`query-id`, `query`）
- `corpus`: 文档页图像（`corpus-id`, `image`）
- `qrels`: 相关性映射（`query-id`, `corpus-id`, `score`）

### 样本

**query.text**: `Who is responsible for integrating climate consideration into Jack in the box governance?...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoRe_esg_reports_human_labeled_v2/` |
| task_type | `visdoc_vidore_v2` |
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
`query_id=0`，`positive_candidate_id=742`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: Who is responsible for integrating climate consideration into Jack in the box governance?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "742",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 739900,
            "sha1": "d077a70f8b91",
            "format": "JPEG",
            "size": [
              3400,
              2200
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
      "742",
      "743",
      "752"
    ],
    "positive_candidate_scores": [
      1.0,
      1.0,
      1.0
    ],
    "total_candidate_ids": 3,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=17`，`positive_candidate_id=1261`

```json
{
  "queries.lance": {
    "id": "17",
    "text": "Find a document image that matches the given query: How many new Shake Shack restaurants opened between 2021 and 2022?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1261",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 589919,
            "sha1": "474d04d0a615",
            "format": "JPEG",
            "size": [
              4973,
              3223
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "17",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1261"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=34`，`positive_candidate_id=1177`

```json
{
  "queries.lance": {
    "id": "34",
    "text": "Find a document image that matches the given query: What was the impact of RBI charity campaigns?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1177",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 479768,
            "sha1": "d5c7ffd79934",
            "format": "JPEG",
            "size": [
              2800,
              1800
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "34",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1177",
      "1161",
      "1162",
      "1163"
    ],
    "positive_candidate_scores": [
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "total_candidate_ids": 4,
    "negative_candidate_count": 0
  }
}
```
