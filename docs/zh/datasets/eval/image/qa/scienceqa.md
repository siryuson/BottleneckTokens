# ScienceQA

> 面向科学教育场景的多模态问答数据集。

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
| 原始名称 | ScienceQA |
| 论文 | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering ([arXiv](https://arxiv.org/abs/2209.09513)) |
| 主要作者 | Pan Lu, Swaroop Mishra, Tang Yao, Daniel Q. Fu, John Tstiatsios, Michihiro Yasunaga, Xinyun Chen, Silvio Savarese, Hannaneh Hajishirzi, Luke Zettlemoyer, Wen-tau Yih |
| HuggingFace | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| 许可证 | 参见原始数据集 |

### Schema
题目、可选图像、选项与答案。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `-22°C...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `ScienceQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `-22°C...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/ScienceQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_544`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which of the following could Gordon's test show?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 13169,
          "sha1": "a7a4b9a345b3",
          "format": "JPEG",
          "size": [
            302,
            232
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_544",
      "text": "how steady a parachute with a 1 m vent was at 200 km per hour",
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
      "c_544"
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
`query_id=q_1`，`positive_candidate_id=c_296`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is the name of the colony shown?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 19696,
          "sha1": "e11723e214c6",
          "format": "JPEG",
          "size": [
            452,
            595
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_296",
      "text": "New Hampshire",
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
      "c_296"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
