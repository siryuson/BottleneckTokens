# RefCOCO-Matching

> RefCOCO 匹配式子集，目标为在候选对象中匹配指代表达。

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
| 原始名称 | RefCOCO-Matching |
| 论文 | Generation and Comprehension of Unambiguous Object Descriptions ([arXiv](https://arxiv.org/abs/1511.02283)) |
| 主要作者 | Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, Kevin Murphy |
| HuggingFace | [jxu124/refcoco-benchmark](https://huggingface.co/datasets/jxu124/refcoco-benchmark) |
| 许可证 | 参见原始数据集 |

### Schema
图像、指代表达、候选目标（同图多目标）匹配。

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
| 子集名 | `RefCOCO-Matching` |
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
| 目录 | `MMEB-V2/image-tasks/RefCOCO-Matching/` |
| task_type | `image_visual_grounding` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_561`，`negative_candidate_id=c_562`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"bowl behind the others can only see part\"\n",
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
      "id": "c_561",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"Dish in top right corner\"\n",
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
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_562",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"front bowl w/carrots in it\"\n",
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
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_561"
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
`query_id=q_1`，`positive_candidate_id=c_1892`，`negative_candidate_id=c_1891`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"little girl\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 67273,
          "sha1": "91db79adf516",
          "format": "JPEG",
          "size": [
            640,
            481
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1892",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"pink\"\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 67273,
            "sha1": "91db79adf516",
            "format": "JPEG",
            "size": [
              640,
              481
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1891",
      "text": "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"green woman\"\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 67273,
            "sha1": "91db79adf516",
            "format": "JPEG",
            "size": [
              640,
              481
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
      "c_1892"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
