# MSCOCO (Visual Grounding)

> 基于 COCO 的视觉定位/匹配评测子集。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_visual_grounding` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集
### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | MS COCO |
| 论文 | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| 主要作者 | Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| 许可证 | 参见原始数据集 |

### Schema
图像与描述文本，配合候选图像形成视觉定位式检索。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 1 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `MSCOCO` |
| 分割 | `test` |
| 评测解析器 | `image_i2i_vg` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 1 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/MSCOCO/` |
| task_type | `image_visual_grounding` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1609`，`negative_candidate_id=c_2343`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nCrop the image to to isolate the object labeled as \"dog\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 170295,
          "sha1": "714a7089e714",
          "format": "JPEG",
          "size": [
            529,
            640
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1609",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 1520,
            "sha1": "545be211d39a",
            "format": "JPEG",
            "size": [
              38,
              29
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2343",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 11951,
            "sha1": "c75eceacf713",
            "format": "JPEG",
            "size": [
              238,
              186
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
      "c_1609"
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
`query_id=q_1`，`positive_candidate_id=c_4033`，`negative_candidate_id=c_1763`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nCrop the image to to isolate the object labeled as \"dog\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 89601,
          "sha1": "739294454342",
          "format": "JPEG",
          "size": [
            640,
            480
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4033",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 17549,
            "sha1": "4d031657f81b",
            "format": "JPEG",
            "size": [
              152,
              280
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1763",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 1267,
            "sha1": "4ea6d84fd2b3",
            "format": "JPEG",
            "size": [
              22,
              45
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
      "c_4033"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
