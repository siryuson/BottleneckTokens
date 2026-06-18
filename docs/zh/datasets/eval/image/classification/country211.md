# Country211

> 国家地理定位分类子集，用于评估视觉地理语义识别能力。

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
| 原始名称 | Country211 |
| 论文 | Learning Transferable Visual Models From Natural Language Supervision（附录给出 Country211）([arXiv](https://arxiv.org/abs/2103.00020)) |
| 主要作者 | Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever |
| HuggingFace | [osv5m/country211](https://huggingface.co/datasets/osv5m/country211) |
| 许可证 | 参见原始数据集 |

### Schema
图像 + ISO 国家标签。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `Afghanistan...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `Country211` |
| 分割 | `test` |
| 评测解析器 | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `Afghanistan...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/Country211/` |
| task_type | `image_classification` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_63`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nIdentify the country depicted in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 47540,
          "sha1": "c6d693404781",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_63",
      "text": "Fiji",
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
      "c_63"
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
`query_id=q_1`，`positive_candidate_id=c_117`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nIdentify the country depicted in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 19041,
          "sha1": "52d9ca9ee579",
          "format": "JPEG",
          "size": [
            499,
            332
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_117",
      "text": "Maldives",
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
      "c_117"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
