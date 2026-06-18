# OK-VQA

> 需要外部知识的视觉问答数据集。

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
| 原始名称 | OK-VQA |
| 论文 | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge ([arXiv](https://arxiv.org/abs/1906.00067)) |
| 主要作者 | Kenneth Marino, Mohammad Rastegari, Ali Farhadi, Roozbeh Mottaghi |
| HuggingFace | [HuggingFaceM4/OK-VQA](https://huggingface.co/datasets/HuggingFaceM4/OK-VQA) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、答案集合。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `OK-VQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/OK-VQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1830`，`negative_candidate_id=c_319`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What sport can you use this for?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 67015,
          "sha1": "f4094f7a48a7",
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
      "id": "c_1830",
      "text": "race",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_319",
      "text": "bake it",
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
      "c_1830"
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
`query_id=q_1`，`positive_candidate_id=c_2429`，`negative_candidate_id=c_1620`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Name the type of plant this is?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 50950,
          "sha1": "0f7fe258734f",
          "format": "JPEG",
          "size": [
            640,
            437
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2429",
      "text": "vine",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1620",
      "text": "octogon",
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
      "c_2429"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
