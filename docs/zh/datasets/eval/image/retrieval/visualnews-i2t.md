# VisualNews I2T

> 新闻图文检索子集（image-to-text）。

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
| 原始名称 | VisualNews |
| 论文 | Visual News: Benchmark and Challenges in News Image Captioning ([arXiv](https://arxiv.org/abs/2010.03743)) |
| 主要作者 | Fuxiao Liu, Yinghan Wang, Tianlu Wang, Vicente Ordonez |
| HuggingFace | [FuxiaoLiu/visualnews](https://huggingface.co/datasets/FuxiaoLiu/visualnews) |
| 许可证 | 参见原始数据集 |

### Schema
新闻图像、新闻标题/描述、元数据。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `08102015 Lucrative farms in the Ica Valley produce asparagus and other crops such as grapes and tang...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `VisualNews_i2t` |
| 分割 | `test` |
| 评测解析器 | `image_i2t` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `08102015 Lucrative farms in the Ica Valley produce asparagus and other crops such as grapes and tang...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/VisualNews_i2t/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_8351`，`negative_candidate_id=c_15164`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nFind a caption for the news in the given photo.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 11525,
          "sha1": "a9c09e155019",
          "format": "JPEG",
          "size": [
            384,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_8351",
      "text": "Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_15164",
      "text": "Some of the 112 portraits already completed.",
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
      "c_8351"
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
`query_id=q_1`，`positive_candidate_id=c_310`，`negative_candidate_id=c_15694`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nFind a caption for the news in the given photo.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 20929,
          "sha1": "3ae38f6edd64",
          "format": "JPEG",
          "size": [
            398,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_310",
      "text": "A Maryland State Police cruiser sits at a blocked southbound entrance on the BaltimoreWashington Parkway that accesses Fort Meade.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_15694",
      "text": "Syracuse Orange head coach Jim Boeheim looks on during the first half against the Georgetown Hoyas at Verizon Center.",
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
      "c_310"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
