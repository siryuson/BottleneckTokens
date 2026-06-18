# CIRR

> 组合式图像检索数据集，查询由参考图像与文本修改指令组成。

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
| 原始名称 | CIRR |
| 论文 | CIRR: Composed Image Retrieval through Referring Relationship ([arXiv](https://arxiv.org/abs/2109.14024)) |
| 主要作者 | Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, Rogerio Feris |
| HuggingFace | [xswu/cirr](https://huggingface.co/datasets/xswu/cirr) |
| 许可证 | 参见原始数据集 |

### Schema
查询图像、文本修改描述、候选图像集合。

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
| 子集名 | `CIRR` |
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
| 目录 | `MMEB-V2/image-tasks/CIRR/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_105`，`negative_candidate_id=c_1979`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nGiven an image, find a similar everyday image with the described changes: Show three bottles of soft drink.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 14234,
          "sha1": "370b012a1ac0",
          "format": "JPEG",
          "size": [
            256,
            416
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_105",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 22491,
            "sha1": "cc7772155abd",
            "format": "JPEG",
            "size": [
              256,
              340
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1979",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 14445,
            "sha1": "5b052143957d",
            "format": "JPEG",
            "size": [
              256,
              384
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
      "c_105"
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
`query_id=q_1`，`positive_candidate_id=c_1372`，`negative_candidate_id=c_150`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nGiven an image, find a similar everyday image with the described changes: Fewer paper towels per pack.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 15115,
          "sha1": "10d01f6ce01b",
          "format": "JPEG",
          "size": [
            384,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1372",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 19802,
            "sha1": "400c566d0533",
            "format": "JPEG",
            "size": [
              256,
              512
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_150",
      "text": "<|image_pad|>\nRepresent the given image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9566,
            "sha1": "56a9b2bd8605",
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
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_1372"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
