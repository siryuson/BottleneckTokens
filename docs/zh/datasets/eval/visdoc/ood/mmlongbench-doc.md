# MMLongBench-doc

> MMLongBench 文档级 OOD 检索子集，覆盖长上下文文档理解场景。

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
| HuggingFace | [siyrus/BToks-MMLongBench-doc](https://huggingface.co/datasets/siyrus/BToks-MMLongBench-doc) |
| 分割 | `test` |
| 论文 | MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations（Ma et al., 2024）([arXiv](https://arxiv.org/abs/2407.01523)) |

### Schema
- `queries`: 问题/检索查询
- `corpus`: 文档页图像候选
- `qrels`: 查询与候选页相关性标注

### 样本

**query.text**: `According to the report, how do 5% of the Latinos see economic upward mobility for their children?...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/MMLongBench-doc/` |
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
`query_id=0`，`positive_candidate_id=0`

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
      "id": "0",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "metadata": {
        "path": "0.png"
      },
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 86259,
            "sha1": "5441c0122320",
            "format": "PNG",
            "size": [
              612,
              792
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
      "22"
    ],
    "positive_candidate_scores": [
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      3.0,
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
    "total_candidate_ids": 23,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=372`，`positive_candidate_id=1797`

```json
{
  "queries.lance": {
    "id": "372",
    "text": "Find a document image that matches the given query: Who are the non-executive and independent directors of GODFREY PHILLIPS INDIA LIMITED? Enumerate all of them in a list.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1797",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 104712,
            "sha1": "11f7b47d7011",
            "format": "PNG",
            "size": [
              593,
              840
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "372",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1797",
      "1798",
      "1799",
      "1800",
      "1801",
      "1802",
      "1803",
      "1804",
      "1805",
      "1806",
      "1807",
      "1808",
      "1809",
      "1810",
      "1811",
      "1812",
      "1813",
      "1814",
      "1815",
      "1816"
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
      3.0,
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
    "total_candidate_ids": 20,
    "negative_candidate_count": 0
  }
}
```

#### 示例 3
`query_id=744`，`positive_candidate_id=4127`

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
      "id": "4127",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 55735,
            "sha1": "95d9c292f599",
            "format": "PNG",
            "size": [
              768,
              576
            ]
          }
        ]
      },
      "score": 2.0
    }
  },
  "qrels.lance": {
    "query_id": "744",
    "mode": "sparse",
    "positive_candidate_ids": [
      "4127",
      "4128",
      "4129",
      "4130",
      "4131",
      "4132",
      "4133",
      "4134",
      "4135",
      "4136",
      "4137",
      "4138",
      "4139",
      "4140",
      "4141",
      "4142",
      "4143",
      "4144",
      "4145",
      "4146",
      "4147",
      "4148",
      "4149",
      "4150",
      "4151",
      "4152",
      "4153",
      "4154",
      "4155",
      "4156",
      "4157",
      "4158"
    ],
    "positive_candidate_scores": [
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      2.0,
      3.0,
      3.0,
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
    "total_candidate_ids": 32,
    "negative_candidate_count": 0
  }
}
```
