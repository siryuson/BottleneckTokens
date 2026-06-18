# ChartQA

> 图表视觉问答数据集，强调数值与逻辑推理。

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
| 原始名称 | ChartQA |
| 论文 | ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning ([arXiv](https://arxiv.org/abs/2203.10244)) |
| 主要作者 | Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, Enamul Hoque |
| HuggingFace | [ahmed-masry/ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) |
| 许可证 | 参见原始数据集 |

### Schema
图表图像、问题、答案。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `-1.6...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `ChartQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `-1.6...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/ChartQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_227`，`negative_candidate_id=c_150`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: How many food item is shown in the bar graph?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 62667,
          "sha1": "ca30ebb41d6a",
          "format": "JPEG",
          "size": [
            850,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_227",
      "text": "14",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_150",
      "text": "11.8",
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
      "c_227"
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
`query_id=q_1`，`positive_candidate_id=c_40`，`negative_candidate_id=c_1541`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the difference in value between Lamb and Corn?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 62667,
          "sha1": "ca30ebb41d6a",
          "format": "JPEG",
          "size": [
            850,
            600
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_40",
      "text": "0.57",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1541",
      "text": "Too little",
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
      "c_40"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
