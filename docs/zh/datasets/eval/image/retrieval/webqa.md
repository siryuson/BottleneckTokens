# WebQA

> 多跳多模态问答衍生的文本到图像检索子集。

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
| 原始名称 | WebQA |
| 论文 | WebQA: Multihop and Multimodal QA ([arXiv](https://arxiv.org/abs/2109.00590)) |
| 主要作者 | Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, Yonatan Bisk |
| HuggingFace | [Stanford/web_questions](https://huggingface.co/datasets/Stanford/web_questions) |
| 许可证 | 参见原始数据集 |

### Schema
问题、证据文本与图像候选集合。

### 样本

**query.text**: `Find a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `WebQA` |
| 分割 | `test` |
| 评测解析器 | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `Find a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/WebQA/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_477`，`negative_candidate_id=c_814`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals in a cup shape?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_477",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: 2020-05-08 15 17 05 Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 19200,
            "sha1": "a1b41cccf4f8",
            "format": "JPEG",
            "size": [
              341,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_814",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Louvre - French sculptures - Room 25 - 03.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 15435,
            "sha1": "eebff019af25",
            "format": "JPEG",
            "size": [
              386,
              256
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
      "c_477"
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
`query_id=q_1`，`positive_candidate_id=c_2136`，`negative_candidate_id=c_1188`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find a Wikipedia image that answers this question: What water-related object is sitting in front of the Torre del Reloj?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2136",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Torre del Reloj de la Plaza Colón de Antofagasta (1).\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 18024,
            "sha1": "b7dcd3976d9e",
            "format": "JPEG",
            "size": [
              256,
              301
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1188",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Isaac White Stanford University Cardinals versus University of San Diego Torendos , Mens Basketball, Chase Center, San Francisco, 2019 Al Attles Classic, Basketball Hall of Fame , CA,.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 15761,
            "sha1": "1e65c6a0cd0c",
            "format": "JPEG",
            "size": [
              256,
              313
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
      "c_2136"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
