# MSCOCO I2T

> 图像到文本检索评测子集（image-to-text）。

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
| 原始名称 | MS COCO Captions |
| 论文 | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| 主要作者 | Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| 许可证 | 参见原始数据集 |

### Schema
图像与多条描述文本，检索任务以图搜文。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `2 Zebras standing next to each other in plaines....`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `MSCOCO_i2t` |
| 分割 | `test` |
| 评测解析器 | `image_i2t` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `2 Zebras standing next to each other in plaines....`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/MSCOCO_i2t/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_2183`，`negative_candidate_id=c_2081`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nFind an image caption describing the given everyday image.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 26975,
          "sha1": "1bbf892ae698",
          "format": "JPEG",
          "size": [
            455,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2183",
      "text": "A man with a red helmet on a small moped on a dirt road.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2081",
      "text": "A man standing in front of an old truck.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_2183"
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
`query_id=q_1`，`positive_candidate_id=c_3646`，`negative_candidate_id=c_2705`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nFind an image caption describing the given everyday image.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 13565,
          "sha1": "bda536a3ac8c",
          "format": "JPEG",
          "size": [
            383,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_3646",
      "text": "A young girl inhales with the intent of blowing out a candle.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2705",
      "text": "A red fire hydrant is in the snow near a mountain.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_3646"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
