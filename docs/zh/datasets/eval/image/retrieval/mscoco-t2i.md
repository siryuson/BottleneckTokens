# MSCOCO T2I

> 文本到图像检索评测子集（text-to-image）。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_retrieval` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集
### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | MS COCO Captions |
| 论文 | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| 主要作者 | Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| 许可证 | 参见原始数据集 |

### Schema
图像与描述文本，检索任务以文搜图。

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
| 子集名 | `MSCOCO_t2i` |
| 分割 | `test` |
| 评测解析器 | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

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
| 目录 | `MMEB-V2/image-tasks/MSCOCO_t2i/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_672`，`negative_candidate_id=c_286`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find me an everyday image that matches the given caption: A man with a red helmet on a small moped on a dirt road.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_672",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 26975,
            "sha1": "1bbf892ae698",
            "format": "JPEG",
            "size": [
              455,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_286",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 11677,
            "sha1": "3b99e51fddba",
            "format": "JPEG",
            "size": [
              341,
              256
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_672"
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
`query_id=q_1`，`positive_candidate_id=c_672`，`negative_candidate_id=c_358`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find me an everyday image that matches the given caption: Man riding a motor bike on a dirt road on the countryside.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_672",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 26975,
            "sha1": "1bbf892ae698",
            "format": "JPEG",
            "size": [
              455,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_358",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 16316,
            "sha1": "14c707bb9474",
            "format": "JPEG",
            "size": [
              317,
              256
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_672"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
