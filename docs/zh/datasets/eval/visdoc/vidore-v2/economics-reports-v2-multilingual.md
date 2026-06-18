# ViDoRe Economics Reports V2 Multilingual

> 经济报告多语言检索子集。

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
| HuggingFace | [vidore/economics_reports_v2](https://huggingface.co/datasets/vidore/economics_reports_v2) |
| 分割 | `test` |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models（Faysse et al., 2024）([arXiv](https://arxiv.org/abs/2407.01449)) |

### Schema
- `queries(query-id, query)`
- `corpus(corpus-id, image)`
- `qrels(query-id, corpus-id, score)`

### 样本

**query.text**: `How do the fiscal challenges in small states compare to those in larger economies?...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoRe_economics_reports_v2_multilingual/` |
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
`query_id=0`，`positive_candidate_id=63`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: How do the fiscal challenges in small states compare to those in larger economies?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "63",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 194518,
            "sha1": "3384e2b55cdc",
            "format": "JPEG",
            "size": [
              850,
              1100
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
      "63"
    ],
    "positive_candidate_scores": [
      2.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=77`，`positive_candidate_id=17`

```json
{
  "queries.lance": {
    "id": "77",
    "text": "Find a document image that matches the given query: Quels sont les défis dans la gestion des risques financiers liés au climat dans les EMDEs?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "17",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 126724,
            "sha1": "462a58a989ab",
            "format": "JPEG",
            "size": [
              850,
              1100
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "77",
    "mode": "sparse",
    "positive_candidate_ids": [
      "17",
      "19",
      "29",
      "63",
      "115",
      "117",
      "118",
      "121",
      "122",
      "144",
      "178",
      "179",
      "180",
      "182",
      "188",
      "189",
      "195"
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
      2.0
    ],
    "total_candidate_ids": 17,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=154`，`positive_candidate_id=18`

```json
{
  "queries.lance": {
    "id": "154",
    "text": "Find a document image that matches the given query: ¿Cómo se compara el clima de negocios en América Latina con el de Medio Oriente?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "18",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 191197,
            "sha1": "e290c477908b",
            "format": "JPEG",
            "size": [
              850,
              1100
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "154",
    "mode": "sparse",
    "positive_candidate_ids": [
      "18",
      "25",
      "129",
      "130",
      "184",
      "219",
      "296",
      "299",
      "342",
      "344"
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
