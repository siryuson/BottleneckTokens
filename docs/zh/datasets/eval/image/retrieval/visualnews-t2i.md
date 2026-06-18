# VisualNews T2I

> 新闻文本到图像检索子集（text-to-image）。

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
新闻图文对与附加元数据。

### 样本

**query.text**: `Retrieve an image of this news caption. Indian National Congress Vice President Rahul Gandhi address...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `VisualNews_t2i` |
| 分割 | `test` |
| 评测解析器 | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `Retrieve an image of this news caption. Indian National Congress Vice President Rahul Gandhi address...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/VisualNews_t2i/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_2279`，`negative_candidate_id=c_4508`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Retrieve an image of this news caption. Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2279",
      "text": "<|image_pad|>\nRepresent the given image.\n",
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
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_4508",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 21553,
            "sha1": "bd63dc998d72",
            "format": "JPEG",
            "size": [
              256,
              378
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
      "c_2279"
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
`query_id=q_1`，`positive_candidate_id=c_2870`，`negative_candidate_id=c_442`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Retrieve an image of this news caption. A Maryland State Police cruiser sits at a blocked southbound entrance on the BaltimoreWashington Parkway that accesses Fort Meade.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2870",
      "text": "<|image_pad|>\nRepresent the given image.\n",
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
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_442",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 25184,
            "sha1": "e029063f652b",
            "format": "JPEG",
            "size": [
              384,
              256
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
      "c_2870"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
