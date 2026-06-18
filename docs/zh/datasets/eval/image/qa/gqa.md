# GQA

> 强调组合推理与场景图语义的视觉问答数据集。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_question_answer` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集
### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | GQA |
| 论文 | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering ([arXiv](https://arxiv.org/abs/1902.09506)) |
| 主要作者 | Drew A. Hudson, Christopher D. Manning |
| HuggingFace | [Graphcore/gqa](https://huggingface.co/datasets/Graphcore/gqa) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、答案及结构化语义信息。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `It is a bathroom....`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `GQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `It is a bathroom....`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/GQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_5372`，`negative_candidate_id=c_6519`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is this bird called?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 49455,
          "sha1": "a21c0a1610b2",
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
      "id": "c_5372",
      "text": "This is a parrot.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_6519",
      "text": "Yes, there is a clock that is gold.",
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
      "c_5372"
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
`query_id=q_1`，`positive_candidate_id=c_3343`，`negative_candidate_id=c_3782`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What color is the helmet in the middle of the image?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 51950,
          "sha1": "fa6868314a30",
          "format": "JPEG",
          "size": [
            500,
            333
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_3343",
      "text": "The helmet is light blue.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3782",
      "text": "The man is to the right of the American flag.",
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
      "c_3343"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
