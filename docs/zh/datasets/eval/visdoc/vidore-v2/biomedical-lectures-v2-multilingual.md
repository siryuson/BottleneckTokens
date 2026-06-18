# ViDoRe Biomedical Lectures V2 Multilingual

> 生物医学讲义多语言检索子集，评测跨语言文档证据检索。

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
| HuggingFace | [vidore/biomedical_lectures_v2](https://huggingface.co/datasets/vidore/biomedical_lectures_v2) |
| 分割 | `test` |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models（Faysse et al., 2024）([arXiv](https://arxiv.org/abs/2407.01449)) |

### Schema
- BEIR 风格：`queries`、`corpus`、`qrels`
- 多语言查询统一在 `query` 字段，页面图像在 `image` 字段

### 样本

**query.text**: `What are the specific outcomes of using autologous chondrocyte implantation in canine studies?...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoRe_biomedical_lectures_v2_multilingual/` |
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
`query_id=0`，`positive_candidate_id=27`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: What are the specific outcomes of using autologous chondrocyte implantation in canine studies?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "27",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 150460,
            "sha1": "c3238aeb1c1e",
            "format": "JPEG",
            "size": [
              1000,
              750
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
      "27",
      "28",
      "29",
      "36"
    ],
    "positive_candidate_scores": [
      2.0,
      2.0,
      2.0,
      2.0
    ],
    "total_candidate_ids": 4,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=213`，`positive_candidate_id=331`

```json
{
  "queries.lance": {
    "id": "213",
    "text": "Find a document image that matches the given query: Comment le document décrit-il la transition du comportement cellulaire non coopératif au comportement coopératif?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "331",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 96043,
            "sha1": "ebeb70c5c7c6",
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
    "query_id": "213",
    "mode": "sparse",
    "positive_candidate_ids": [
      "331",
      "332",
      "333",
      "334",
      "335"
    ],
    "positive_candidate_scores": [
      2.0,
      2.0,
      2.0,
      2.0,
      2.0
    ],
    "total_candidate_ids": 5,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=426`，`positive_candidate_id=696`

```json
{
  "queries.lance": {
    "id": "426",
    "text": "Find a document image that matches the given query: ¿Cuáles son las posibles implicaciones de alterar las estructuras terciaria y cuaternaria del colágeno en sus funciones biológicas?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "696",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 126080,
            "sha1": "0ddfc90794c4",
            "format": "JPEG",
            "size": [
              1100,
              850
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "426",
    "mode": "sparse",
    "positive_candidate_ids": [
      "696"
    ],
    "positive_candidate_scores": [
      2.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
