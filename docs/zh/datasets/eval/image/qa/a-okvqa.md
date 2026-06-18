# A-OKVQA

> 结合常识推理与视觉证据的开放式视觉问答数据集。

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
| 原始名称 | A-OKVQA |
| 论文 | A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge ([arXiv](https://arxiv.org/abs/2206.01718)) |
| 主要作者 | Dustin Schwenk, Jack Hessel, Ari Holtzman, Ronan Le Bras, Chandra Bhagavatula, Yejin Choi |
| HuggingFace | [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、直接答案与理由等字段。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `1000...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `A-OKVQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `1000...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/A-OKVQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_140`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is in the motorcyclist's mouth?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 54735,
          "sha1": "a6288593a2ab",
          "format": "JPEG",
          "size": [
            640,
            569
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_140",
      "text": "cigarette",
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
      "c_140"
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
`query_id=q_1`，`positive_candidate_id=c_774`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which number birthday is probably being celebrated?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 70271,
          "sha1": "3d8607931911",
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
      "id": "c_774",
      "text": "thirty",
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
      "c_774"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
