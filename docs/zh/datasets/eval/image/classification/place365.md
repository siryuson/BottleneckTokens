# Place365

> 场景识别基准，用于评估模型对地理与场景语义的区分能力。

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
| 原始名称 | Places365 |
| 论文 | Places: A 10 million Image Database for Scene Recognition ([arXiv](https://arxiv.org/abs/1610.02055)) |
| 主要作者 | Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, Antonio Torralba |
| HuggingFace | [Eth1et/places365](https://huggingface.co/datasets/Eth1et/places365) |
| 许可证 | 参见原始数据集 |

### Schema
图像 + 场景标签。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `airfield...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `Place365` |
| 分割 | `test` |
| 评测解析器 | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `airfield...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/Place365/` |
| task_type | `image_classification` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_55`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nIdentify the scene shown in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 12954,
          "sha1": "9f46d14d531a",
          "format": "JPEG",
          "size": [
            256,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_55",
      "text": "berth",
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
      "c_55"
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
`query_id=q_1`，`positive_candidate_id=c_262`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nIdentify the scene shown in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 16390,
          "sha1": "cfad8b3516e1",
          "format": "JPEG",
          "size": [
            256,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_262",
      "text": "pharmacy",
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
      "c_262"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
