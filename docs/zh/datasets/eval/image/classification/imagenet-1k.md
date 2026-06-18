# ImageNet-1K

> 经典大规模图像分类基准，用于评估通用视觉语义区分能力。

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
| 原始名称 | ImageNet (ILSVRC 2012) |
| 论文 | ImageNet Large Scale Visual Recognition Challenge ([arXiv](https://arxiv.org/abs/1409.0575)) |
| 主要作者 | Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, Li Fei-Fei |
| HuggingFace | [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| 许可证 | 参见原始数据集 |

### Schema
常见分类样本结构为图像与类别标签；在 MMEB 流程中统一映射为指令化 query 与候选标签集合。

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
| 子集名 | `ImageNet-1K` |
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
| 目录 | `MMEB-V2/image-tasks/ImageNet-1K/` |
| task_type | `image_classification` |
| 评测模式 | local |

### Lance 布局

**queries.lance**: `id`, `text`, `images`  
**candidates.lance**: `id`, `text`, `images`  
**qrels.lance**: `query_id`, `mode`, `candidate_ids`, `candidate_scores`

**metadata.json**:
```json
{
  "name": "ImageNet-1K",
  "task_type": "image_classification",
  "num_queries": "...",
  "num_candidates": "..."
}
```

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_317`

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
          "bytes": 39025,
          "sha1": "7ddac4291320",
          "format": "JPEG",
          "size": [
            408,
            500
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_317",
      "text": "coucal",
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
      "c_317"
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
`query_id=q_1`，`positive_candidate_id=c_65`

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
          "bytes": 34385,
          "sha1": "a86566606b54",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_65",
      "text": "Italian greyhound",
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
      "c_65"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
