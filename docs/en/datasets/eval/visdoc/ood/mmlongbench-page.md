# MMLongBench-page

> MMLongBench 页级 OOD 检索子集，关注长文档多页视觉检索能力。

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
| HuggingFace | [siyrus/BToks-MMLongBench-page-fixed](https://huggingface.co/datasets/siyrus/BToks-MMLongBench-page-fixed) |
| 分割 | `test` |
| 论文 | MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations（Ma et al., 2024）([arXiv](https://arxiv.org/abs/2407.01523)) |

### Schema
- BEIR 三元结构：`queries`、`corpus`、`qrels`
- 支持跨页检索评估，相关性标注位于 `qrels.score`

### 样本
> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/MMLongBench-page/` |
| task_type | `visdoc_ood` |
| 聚合指标 | `ndcg@5` |

### Lance 布局
- `queries.lance`
- `candidates.lance`
- `qrels.lance`

### 样本

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.
- Runtime `metadata` is preserved for the query-side page range and candidate page path.

#### Example 1
`query_id=0`, `positive_candidate_id=4`

```json
{
  "queries.lance": {
    "id": "0",
    "text": "Find a document image that matches the given query: According to the report, how do 5% of the Latinos see economic upward mobility for their children?\n",
    "images": {
      "count": 0,
      "items": []
    },
    "metadata": {
      "corpus_range": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "4",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "metadata": {
        "path": "4.png"
      },
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 118043,
            "sha1": "2e3d4c2fa4b5",
            "format": "PNG",
            "size": [
              612,
              792
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
      "4"
    ],
    "positive_candidate_scores": [
      3.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### Example 2
`query_id=744`, `positive_candidate_id=4133`

```json
{
  "queries.lance": {
    "id": "744",
    "text": "Find a document image that matches the given query: In which system the throttle valve is placed beneath the fuel injector?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "4133",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 172158,
            "sha1": "19d2ad5eeb6d",
            "format": "PNG",
            "size": [
              768,
              576
            ]
          }
        ]
      },
      "score": 3.0
    }
  },
  "qrels.lance": {
    "query_id": "744",
    "mode": "sparse",
    "positive_candidate_ids": [
      "4133",
      "4134"
    ],
    "positive_candidate_scores": [
      3.0,
      3.0
    ],
    "total_candidate_ids": 2,
    "negative_candidate_count": 0
  }
}
```

#### Example 3
`query_id=936`, `positive_candidate_id=4500`

```json
{
  "queries.lance": {
    "id": "936",
    "text": "Find a document image that matches the given query: Mention Tablet Capture Devices used by some hospitals?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "4500",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 148915,
            "sha1": "c4a5df409105",
            "format": "PNG",
            "size": [
              720,
              540
            ]
          }
        ]
      },
      "score": 3.0
    }
  },
  "qrels.lance": {
    "query_id": "936",
    "mode": "sparse",
    "positive_candidate_ids": [
      "4500",
      "4500",
      "4500",
      "4500",
      "4500",
      "4500"
    ],
    "positive_candidate_scores": [
      3.0,
      3.0,
      3.0,
      3.0,
      3.0,
      3.0
    ],
    "total_candidate_ids": 6,
    "negative_candidate_count": 0
  }
}
```
