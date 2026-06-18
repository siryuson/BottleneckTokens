# N24News

> 多模态新闻分类数据集，覆盖 24 个新闻类别。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_classification` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | N24News |
| 论文 | N24News: A New Dataset for Multimodal News Classification ([arXiv](https://arxiv.org/abs/2108.13327)) |
| 主要作者 | Zhen Wang, Xu Shan, Xiangxie Zhang, Jie Yang |
| HuggingFace | [linxy/Multimodal-N24News](https://huggingface.co/datasets/linxy/Multimodal-N24News) |
| 许可证 | 参见原始数据集 |

### Schema
图像 + 新闻文本/标题 + 类别标签；在 MMEB 中统一为图像查询与候选标签。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `Art & Design...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `N24News` |
| 分割 | `test` |
| 评测解析器 | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `Art & Design...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/N24News/` |
| task_type | `image_classification` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`（统一三表结构）。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_21`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given news image with the following caption for domain classification: Anthony Pidgeon\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 189554,
          "sha1": "1c213c84dcf6",
          "format": "JPEG",
          "size": [
            1050,
            549
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_21",
      "text": "Travel",
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
      "c_21"
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
`query_id=q_1`，`positive_candidate_id=c_17`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given news image with the following caption for domain classification: Ms. Goodman styled Amber Valletta with wings for a 1993 shoot by Peter Lindbergh for Harper's Bazaar.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 138959,
          "sha1": "004cebd3131c",
          "format": "JPEG",
          "size": [
            1050,
            550
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_17",
      "text": "Style",
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
      "c_17"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
