# DocVQA

> 文档图像问答基准，关注页面文字与版式理解。

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
| 原始名称 | DocVQA (Task 1: VQA on Document Images) |
| 论文 | ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction ([arXiv](https://arxiv.org/abs/1911.12482)) |
| 主要作者 | Minesh Mathew, Dimosthenis Karatzas, C. V. Jawahar |
| HuggingFace | [cmarkea/docvqa](https://huggingface.co/datasets/cmarkea/docvqa) |
| 许可证 | 参见原始数据集 |

### Schema
文档图像、问题文本、答案列表。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `"Effects on health of alcohol consumption"...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `DocVQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `"Effects on health of alcohol consumption"...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/DocVQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_269`，`negative_candidate_id=c_721`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the ‘actual’ value per 1000, during the year 1975?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 276397,
          "sha1": "9ee7cdea0973",
          "format": "JPEG",
          "size": [
            2257,
            1764
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_269",
      "text": "0.28",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_721",
      "text": "1992",
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
      "c_269"
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
`query_id=q_1`，`positive_candidate_id=c_4388`，`negative_candidate_id=c_3025`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is name of university?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 333318,
          "sha1": "e1daaf1bed29",
          "format": "JPEG",
          "size": [
            808,
            1077
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4388",
      "text": "university of california",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3025",
      "text": "Postcards",
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
      "c_4388"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
