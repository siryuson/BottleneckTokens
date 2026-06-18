# ImageNet-A

> ImageNet 对抗/困难自然分布子集，用于测试鲁棒性泛化。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_classification` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集
### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | ImageNet-A |
| 论文 | Natural Adversarial Examples ([arXiv](https://arxiv.org/abs/1907.07174)) |
| 主要作者 | Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, Dawn Song |
| HuggingFace | [Maysee/tiny-imagenet-a](https://huggingface.co/datasets/Maysee/tiny-imagenet-a) |
| 许可证 | 参见原始数据集 |

### Schema
图像 + ImageNet 类别标签（困难样本）。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `Afghan hound, Afghan...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `ImageNet-A` |
| 分割 | `test` |
| 评测解析器 | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `Afghan hound, Afghan...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/ImageNet-A/` |
| task_type | `image_classification` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_90`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image for classification\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 15440,
          "sha1": "11a2813d1749",
          "format": "JPEG",
          "size": [
            236,
            314
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_90",
      "text": "Rottweiler",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_90"
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
`query_id=q_1`，`positive_candidate_id=c_90`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image for classification\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 7255,
          "sha1": "c0de3dc2b653",
          "format": "JPEG",
          "size": [
            300,
            300
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_90",
      "text": "Rottweiler",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_90"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
