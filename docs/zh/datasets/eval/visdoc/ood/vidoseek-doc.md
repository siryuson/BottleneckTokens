# ViDoSeek-doc

> ViDoSeek 文档级 OOD 检索子集，强调复杂文档场景下的检索鲁棒性。

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
| HuggingFace | [siyrus/BToks-ViDoSeek](https://huggingface.co/datasets/siyrus/BToks-ViDoSeek) |
| 分割 | `test` |
| 论文 | ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents（Wang et al., 2025）([arXiv](https://arxiv.org/abs/2502.18017)) |

### Schema
- `queries(query-id, query)`
- `corpus(corpus-id, image)`
- `qrels(query-id, corpus-id, score)`

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
| 目录 | `MMEB-V2/visdoc-tasks/ViDoSeek-doc/` |
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
- `metadata` 继续保留 runtime 侧的辅助字段，例如 `corpus_range`。

#### 示例 1
`query_id=0`，`positive_candidate_id=0`

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
      "id": "0",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 32904,
            "sha1": "c8184c764b8a",
            "format": "JPEG",
            "size": [
              960,
              540
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
      "0",
      "1",
      "2",
      "3",
      "4",
      "5",
      "6",
      "7",
      "8",
      "9",
      "10",
      "11",
      "12",
      "13",
      "14",
      "15",
      "16",
      "17",
      "18",
      "19",
      "20",
      "21",
      "22",
      "23",
      "4"
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
      2.0,
      3.0
    ],
    "total_candidate_ids": 25,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=380`，`positive_candidate_id=1959`

```json
{
  "queries.lance": {
    "id": "380",
    "text": "Find a document image that matches the given query: What are the three components of the transitioning HPC I/O Storage Stack mentioned in the Seagate presentation on Accelerating Spectrum Scale with an Intelligent IO Manager?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1959",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 39177,
            "sha1": "777bd7250ed4",
            "format": "JPEG",
            "size": [
              720,
              405
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "380",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1959",
      "1960",
      "1961",
      "1962",
      "1963",
      "1964",
      "1965",
      "1966",
      "1967",
      "1968",
      "1969",
      "1970",
      "1971",
      "1972",
      "1973",
      "1974",
      "1975",
      "1976",
      "1977",
      "1962"
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
      3.0
    ],
    "total_candidate_ids": 20,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=760`，`positive_candidate_id=3750`

```json
{
  "queries.lance": {
    "id": "760",
    "text": "Find a document image that matches the given query: What are the two primary components represented in the theoretical model discussed in the Italian Geosciences Congresso SGI-SIMP 2014 presentation?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "3750",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 76642,
            "sha1": "86c84b8a2fe9",
            "format": "JPEG",
            "size": [
              720,
              540
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "760",
    "mode": "sparse",
    "positive_candidate_ids": [
      "3750",
      "3751",
      "3752",
      "3753",
      "3754",
      "3755",
      "3756",
      "3757",
      "3758",
      "3759",
      "3760",
      "3761",
      "3762",
      "3763",
      "3764",
      "3753"
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
      3.0
    ],
    "total_candidate_ids": 16,
    "negative_candidate_count": 0
  }
}
```
