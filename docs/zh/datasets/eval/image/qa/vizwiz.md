# VizWiz

> 由视障用户拍摄图像构成的真实场景视觉问答数据集。

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
| 原始名称 | VizWiz-VQA |
| 论文 | VizWiz Grand Challenge: Answering Visual Questions from Blind People ([arXiv](https://arxiv.org/abs/1802.08218)) |
| 主要作者 | Danna Gurari, Qing Li, Abigale Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, Jeffrey P. Bigham |
| HuggingFace | [lmms-lab/VizWiz-VQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、答案标注与可回答性信息。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `$150 gift card...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `VizWiz` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `$150 gift card...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/VizWiz/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_883`，`negative_candidate_id=c_1276`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Can you tell me what this medicine is please?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 4314,
          "sha1": "3cf0301aced8",
          "format": "JPEG",
          "size": [
            121,
            162
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_883",
      "text": "night time",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1276",
      "text": "tonic wine",
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
      "c_883"
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
`query_id=q_1`，`positive_candidate_id=c_471`，`negative_candidate_id=c_67`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the title of this book?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 29759,
          "sha1": "8ed3f2e024f8",
          "format": "JPEG",
          "size": [
            484,
            648
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_471",
      "text": "dog years",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_67",
      "text": "advanced boot options",
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
      "c_471"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
