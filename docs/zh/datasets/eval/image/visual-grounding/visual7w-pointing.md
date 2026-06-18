# Visual7W-Pointing

> Visual7W 的指向式问答子任务，评测目标区域定位能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_visual_grounding` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集
### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Visual7W Pointing |
| 论文 | Visual7W: Grounded Question Answering in Images ([arXiv](https://arxiv.org/abs/1511.07332)) |
| 主要作者 | Yuke Zhu, Oliver Groth, Michael Bernstein, Li Fei-Fei |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| 许可证 | 参见原始数据集 |

### Schema
图像、问题、候选框与正确指向目标。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `Visual7W-Pointing` |
| 分割 | `test` |
| 评测解析器 | `image_i2i_vg` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `<|image_pad|>...`
**query.images**: 1 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/Visual7W-Pointing/` |
| task_type | `image_visual_grounding` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_223`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nSelect the portion of the image that answers the question \"Which box frames the toilet?\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 27250,
          "sha1": "6265464273bf",
          "format": "JPEG",
          "size": [
            375,
            500
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_223",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 10008,
            "sha1": "a2ab24abd487",
            "format": "JPEG",
            "size": [
              138,
              179
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_223"
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
`query_id=q_1`，`positive_candidate_id=c_453`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nSelect the portion of the image that answers the question \"Which is the stairs under the skater?\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 38652,
          "sha1": "69e5e40cc21d",
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
      "id": "c_453",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 18206,
            "sha1": "19f1995ae846",
            "format": "JPEG",
            "size": [
              165,
              208
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_453"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
