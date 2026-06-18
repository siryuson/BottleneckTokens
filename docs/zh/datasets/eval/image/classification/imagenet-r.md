# ImageNet-R

> ImageNet 领域外渲染风格子集（绘画/卡通等），用于评估分布外鲁棒性。

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
| 原始名称 | ImageNet-R |
| 论文 | Are we done with ImageNet? ([arXiv](https://arxiv.org/abs/2006.07159)) |
| 主要作者 | Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Austin Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, Justin Gilmer |
| HuggingFace | [yerevann/imagenet-r](https://huggingface.co/datasets/yerevann/imagenet-r) |
| 许可证 | 参见原始数据集 |

### Schema
图像 + ImageNet 类别标签（渲染风格迁移样本）。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `African_chameleon...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `ImageNet-R` |
| 分割 | `test` |
| 评测解析器 | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `African_chameleon...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/ImageNet-R/` |
| task_type | `image_classification` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_67`

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
          "bytes": 16148,
          "sha1": "6b287ddef517",
          "format": "JPEG",
          "size": [
            450,
            470
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_67",
      "text": "flute",
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
      "c_67"
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
`query_id=q_1`，`positive_candidate_id=c_67`

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
          "bytes": 4072,
          "sha1": "a199cbd4c74b",
          "format": "JPEG",
          "size": [
            257,
            194
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_67",
      "text": "flute",
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
      "c_67"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
