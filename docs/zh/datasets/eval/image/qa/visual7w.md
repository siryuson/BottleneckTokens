# Visual7W

> 基于图像与多种问答形式的视觉问答数据集。

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
| 原始名称 | Visual7W |
| 论文 | Visual7W: Grounded Question Answering in Images ([arXiv](https://arxiv.org/abs/1511.07332)) |
| 主要作者 | Yuke Zhu, Oliver Groth, Michael Bernstein, Li Fei-Fei |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、候选答案与标注答案。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `"RESTAURANT"....`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `Visual7W` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `"RESTAURANT"....`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/Visual7W/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1344`，`negative_candidate_id=c_3163`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What color is the sidewalk?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 106591,
          "sha1": "cc861c687eff",
          "format": "JPEG",
          "size": [
            800,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1344",
      "text": "Gray.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3163",
      "text": "To receive orders.",
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
      "c_1344"
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
`query_id=q_1`，`positive_candidate_id=c_639`，`negative_candidate_id=c_2189`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Where is the man sitting?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 60190,
          "sha1": "e698932ba877",
          "format": "JPEG",
          "size": [
            800,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_639",
      "text": "At the computer.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2189",
      "text": "On the street.",
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
      "c_639"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
