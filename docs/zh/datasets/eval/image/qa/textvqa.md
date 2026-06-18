# TextVQA

> 面向图像文本读取与推理的视觉问答数据集。

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
| 原始名称 | TextVQA |
| 论文 | Towards VQA Models That Can Read ([arXiv](https://arxiv.org/abs/1904.08920)) |
| 主要作者 | Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach |
| HuggingFace | [lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、OCR 文本与答案。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `#...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `TextVQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `#...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/TextVQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1209`，`negative_candidate_id=c_2583`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: what is the brand of this camera?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 47139,
          "sha1": "5e72db6ed822",
          "format": "JPEG",
          "size": [
            1024,
            664
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1209",
      "text": "dakota",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2583",
      "text": "rose",
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
      "c_1209"
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
`query_id=q_1`，`positive_candidate_id=c_1148`，`negative_candidate_id=c_1263`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: what does the small white text spell?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 80334,
          "sha1": "aaaea1a3a7e2",
          "format": "JPEG",
          "size": [
            1024,
            683
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1148",
      "text": "copenhagen",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1263",
      "text": "dino buzzati",
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
      "c_1148"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
