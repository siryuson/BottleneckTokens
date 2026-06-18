# ViDoRe ESG Reports V2 Multilingual

> ESG 报告多语言检索子集，覆盖跨语言与跨版面检索需求。

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
| HuggingFace | [vidore/esg_reports_v2](https://huggingface.co/datasets/vidore/esg_reports_v2) |
| 分割 | `test` |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models（Faysse et al., 2024）([arXiv](https://arxiv.org/abs/2407.01449)) |

### Schema
- 标准 BEIR 结构：`queries` / `corpus` / `qrels`
- 页面级候选以图像存储，使用 `score` 表示相关性

### 样本

**query.text**: `Quelles sont les marques de RBI mentionnées dans le rapport ?...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoRe_esg_reports_v2_multilingual/` |
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
`query_id=0`，`positive_candidate_id=2`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: Quelles sont les marques de RBI mentionnées dans le rapport ?\n",
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
            "bytes": 507688,
            "sha1": "0f64fa9f6f39",
            "format": "JPEG",
            "size": [
              2800,
              1800
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "2",
      "3",
      "4",
      "7",
      "8",
      "9",
      "10",
      "12",
      "14",
      "15",
      "16",
      "17",
      "18",
      "19",
      "20",
      "21",
      "25",
      "26",
      "27",
      "33",
      "34",
      "35",
      "36",
      "49"
    ],
    "positive_candidate_scores": [
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0
    ],
    "total_candidate_ids": 24,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=75`，`positive_candidate_id=477`

```json
{
  "queries.lance": {
    "id": "75",
    "text": "Find a document image that matches the given query: What are HelloFresh's sustainability ambitions?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "477",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 494704,
            "sha1": "8cbc2c995470",
            "format": "JPEG",
            "size": [
              1654,
              2338
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "75",
    "mode": "sparse",
    "positive_candidate_ids": [
      "477",
      "478",
      "482",
      "483",
      "489",
      "490",
      "491",
      "495",
      "496",
      "498"
    ],
    "positive_candidate_scores": [
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=151`，`positive_candidate_id=1170`

```json
{
  "queries.lance": {
    "id": "151",
    "text": "Find a document image that matches the given query: ¿Cómo se compara la gestión de energía y agua con la gestión de residuos alimentarios y envases en Jack in the Box?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1170",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 550278,
            "sha1": "fa1707bc1099",
            "format": "JPEG",
            "size": [
              3400,
              2200
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "151",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1170"
    ],
    "positive_candidate_scores": [
      2.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
