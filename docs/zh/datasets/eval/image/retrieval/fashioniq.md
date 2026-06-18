# FashionIQ

> 基于自然语言反馈的时尚图像检索数据集。

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
| 原始名称 | Fashion IQ |
| 论文 | Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback ([arXiv](https://arxiv.org/abs/1905.12794)) |
| 主要作者 | Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, Rogerio Feris |
| HuggingFace | [McAuley-Lab/Amazon-Customer-Reviews-Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Customer-Reviews-Dataset) |
| 许可证 | 参见原始数据集 |

### Schema
参考图像、文本反馈、目标图像。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `FashionIQ` |
| 分割 | `test` |
| 评测解析器 | `image_i2i_vg` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/FashionIQ/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_2930`，`negative_candidate_id=c_3283`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nFind an image to match the fashion image and style note: Is shiny and silver with shorter sleeves and fit and flare.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 9165,
          "sha1": "9c48d6effccd",
          "format": "JPEG",
          "size": [
            256,
            332
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2930",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 7212,
            "sha1": "01a3eb6f3c3d",
            "format": "JPEG",
            "size": [
              256,
              332
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3283",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 8242,
            "sha1": "a2a5b55d1d97",
            "format": "JPEG",
            "size": [
              256,
              356
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
      "c_2930"
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
`query_id=q_1`，`positive_candidate_id=c_4212`，`negative_candidate_id=c_2044`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nFind an image to match the fashion image and style note: Is grey with black design and is a light printed short dress.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 8965,
          "sha1": "faa293c84050",
          "format": "JPEG",
          "size": [
            256,
            306
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4212",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 8777,
            "sha1": "4126fa19f31a",
            "format": "JPEG",
            "size": [
              256,
              320
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2044",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9211,
            "sha1": "fbb11f14fad3",
            "format": "JPEG",
            "size": [
              256,
              332
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
      "c_4212"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
