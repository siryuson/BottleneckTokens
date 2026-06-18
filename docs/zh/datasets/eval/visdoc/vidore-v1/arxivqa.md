# ViDoRe ArxivQA

> 来自学术论文场景的视觉文档检索子集，评测跨版面证据匹配能力。

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
| HuggingFace | [vidore/arxivqa_test_subsampled_beir](https://huggingface.co/datasets/vidore/arxivqa_test_subsampled_beir) |
| 分割 | `test` |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models（Faysse et al., 2024）([arXiv](https://arxiv.org/abs/2407.01449)) |

### Schema
- 采用 BEIR 风格三元结构：
  - `queries`: `query-id`, `query`
  - `corpus`: `corpus-id`, `image`
  - `qrels`: `query-id`, `corpus-id`, `score`

### 样本

**query.text**: `Based on the graph, what is the impact of correcting for fspec not equal to 1 on the surface density...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/ViDoRe_arxivqa/` |
| task_type | `visdoc_vidore_v1` |
| 聚合指标 | `ndcg@5` |

### Lance 布局
- `queries.lance`: 文本查询（`id`, `text`, `images=[]`）
- `candidates.lance`: 文档页图像候选（`id`, `text`, `images`）
- `qrels.lance`: 稀疏相关性标注（`query_id`, `candidate_ids`, `candidate_scores`）

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
    "text": "Find a document image that matches the given query: Based on the graph, what is the impact of correcting for fspec not equal to 1 on the surface density trend?\n",
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
            "bytes": 139626,
            "sha1": "adc52adafc98",
            "format": "JPEG",
            "size": [
              1621,
              1191
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
`query_id=1`，`positive_candidate_id=1`

```json
{
  "queries.lance": {
    "id": "1",
    "text": "Find a document image that matches the given query: Based on the progression from JUL10 to FEB11Q, what trend can be observed in the thread participation?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "1",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 226757,
            "sha1": "b38181cfcc6e",
            "format": "JPEG",
            "size": [
              2016,
              899
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "1"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
