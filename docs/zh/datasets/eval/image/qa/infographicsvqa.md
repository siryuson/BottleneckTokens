# InfographicsVQA

> 面向信息图场景的视觉问答数据集，强调图文混合与布局理解。

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
| 原始名称 | InfographicsVQA |
| 论文 | InfographicVQA ([arXiv](https://arxiv.org/abs/2104.12756)) |
| 主要作者 | Furkan Biten, Lluis Gomez, Marçal Rusinol, Dimosthenis Karatzas, C. V. Jawahar |
| HuggingFace | [vidore/infovqa_test_subsampled_beir](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir) |
| 许可证 | 参见原始数据集 |

### Schema
信息图图像、问题、答案集合。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `"beast of kandahar"...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `InfographicsVQA` |
| 分割 | `test` |
| 评测解析器 | `image_qa` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `"beast of kandahar"...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/InfographicsVQA/` |
| task_type | `image_question_answer` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1446`，`negative_candidate_id=c_1507`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which social platform has heavy female audience?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 255643,
          "sha1": "c6815eb609f2",
          "format": "JPEG",
          "size": [
            969,
            2221
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1446",
      "text": "pinterest",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1507",
      "text": "runny nose, cough, sore throat",
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
      "c_1446"
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
`query_id=q_1`，`positive_candidate_id=c_1488`，`negative_candidate_id=c_1015`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which three business types is Pinterest good for?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 255643,
          "sha1": "c6815eb609f2",
          "format": "JPEG",
          "size": [
            969,
            2221
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1488",
      "text": "restaurants, interior design, wedding venues",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1015",
      "text": "doers, artists",
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
      "c_1488"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
