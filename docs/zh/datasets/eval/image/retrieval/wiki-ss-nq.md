# Wiki-SS-NQ

> 基于维基百科截图语料的文本到图像检索子集。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_retrieval` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集
### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Wiki-SS-NQ |
| 论文 | Unifying Multimodal Retrieval via Document Screenshot Embedding ([arXiv](https://arxiv.org/abs/2406.11251)) |
| 主要作者 | Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, Jimmy Lin |
| HuggingFace | [Tevatron/wiki-ss-nq](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) |
| 许可证 | 参见原始数据集 |

### Schema
自然语言问题、页面截图候选与相关性标注。

### 样本

**query.text**: `Find the document image that can answer the given query: who needs to know about the jet stream...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `Wiki-SS-NQ` |
| 分割 | `test` |
| 评测解析器 | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `Find the document image that can answer the given query: who needs to know about the jet stream...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/Wiki-SS-NQ/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1482`，`negative_candidate_id=c_2775`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find the document image that can answer the given query: who needs to know about the jet stream\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1482",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 186874,
            "sha1": "256d45696c03",
            "format": "JPEG",
            "size": [
              980,
              980
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2775",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 81037,
            "sha1": "197e031f464b",
            "format": "JPEG",
            "size": [
              980,
              980
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_1482"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```

#### 示例 2
`query_id=q_1`，`positive_candidate_id=c_7031`，`negative_candidate_id=c_929`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find the document image that can answer the given query: who founded the red cross to relieve the suffering of the war wounded\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_7031",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 156902,
            "sha1": "87a79e84422d",
            "format": "JPEG",
            "size": [
              980,
              980
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_929",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 217585,
            "sha1": "acc47505b96d",
            "format": "JPEG",
            "size": [
              980,
              980
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_7031"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
