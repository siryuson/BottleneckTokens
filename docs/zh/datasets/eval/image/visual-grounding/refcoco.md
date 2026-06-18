# RefCOCO

> 指代表达理解数据集，用于目标定位与图文匹配评测。

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
| 原始名称 | RefCOCO |
| 论文 | Generation and Comprehension of Unambiguous Object Descriptions ([arXiv](https://arxiv.org/abs/1511.02283)) |
| 主要作者 | Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, Kevin Murphy |
| HuggingFace | [jxu124/refcoco-benchmark](https://huggingface.co/datasets/jxu124/refcoco-benchmark) |
| 许可证 | 参见原始数据集 |

### Schema
图像、指代表达、目标区域标注。

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
| 子集名 | `RefCOCO` |
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
| 目录 | `MMEB-V2/image-tasks/RefCOCO/` |
| task_type | `image_visual_grounding` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_2033`，`negative_candidate_id=c_4986`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions. \"bowl behind the others can only see part\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 43142,
          "sha1": "6f679b125e46",
          "format": "JPEG",
          "size": [
            640,
            428
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2033",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 2963,
            "sha1": "5b9ac77aa5f4",
            "format": "JPEG",
            "size": [
              172,
              116
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_4986",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 2782,
            "sha1": "416a76bf5bb8",
            "format": "JPEG",
            "size": [
              119,
              123
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
      "c_2033"
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
`query_id=q_1`，`positive_candidate_id=c_8220`，`negative_candidate_id=c_4937`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions. \"front bowl w/carrots in it\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 43142,
          "sha1": "6f679b125e46",
          "format": "JPEG",
          "size": [
            640,
            428
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_8220",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 26362,
            "sha1": "837cdcee2d8e",
            "format": "JPEG",
            "size": [
              455,
              284
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_4937",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 29450,
            "sha1": "09484421a47b",
            "format": "JPEG",
            "size": [
              235,
              295
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
      "c_8220"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
