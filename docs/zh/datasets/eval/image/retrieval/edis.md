# EDIS

> 新闻域实体驱动图像检索数据集（text-to-image）。

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
| 原始名称 | EDIS |
| 论文 | EDIS: Entity-Driven Image Search over Multimodal Web Content ([arXiv](https://arxiv.org/abs/2305.13631)) |
| 主要作者 | Siqi Liu, Weixi Feng, Tsu-jui Fu, Wenhu Chen, William Yang Wang |
| HuggingFace | [Emerisly/EDIS](https://huggingface.co/datasets/Emerisly/EDIS) |
| 许可证 | 参见原始数据集 |

### Schema
文本查询、图像候选及配套文本描述。

### 样本

**query.text**: `Find a news image that matches the provided caption: Former Israeli president and prime minister Shi...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `EDIS` |
| 分割 | `test` |
| 评测解析器 | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `Find a news image that matches the provided caption: Former Israeli president and prime minister Shi...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/EDIS/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_1505`，`negative_candidate_id=c_3122`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find a news image that matches the provided caption: Former Israeli president and prime minister Shimon Peres right and Palestinian President Mahmoud Abbas arrive at the World Economic Forum in Southern Shuneh Jordan in May 2015.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1505",
      "text": "<|image_pad|>\nRepresent the given image with related text information: WorldAs the West pays tribute to Peres, many Arabs recall a legacy of destruction.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9168,
            "sha1": "ca542c3af0c4",
            "format": "JPEG",
            "size": [
              358,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3122",
      "text": "<|image_pad|>\nRepresent the given image with related text information: • Discovered by the Germans in 1904, they named it San Diego, which of course in German means 'a whale's vagina'.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 21951,
            "sha1": "43274d6ba8b3",
            "format": "JPEG",
            "size": [
              426,
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
      "c_1505"
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
`query_id=q_1`，`positive_candidate_id=c_1595`，`negative_candidate_id=c_1334`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find a news image that matches the provided caption: Tom Holland makes his debut in the Spidey suit in Captain America Civil War.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1595",
      "text": "<|image_pad|>\nRepresent the given image with related text information: Comic RiffsJon Favreau is set to reprise his Iron Man role for Spider Man: Homecoming.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9441,
            "sha1": "a9d04212c901",
            "format": "JPEG",
            "size": [
              454,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1334",
      "text": "<|image_pad|>\nRepresent the given image with related text information: GovBeatColorado students are protesting en masse over a curriculum proposal to promote respect for authority.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 22049,
            "sha1": "83d9ec61b335",
            "format": "JPEG",
            "size": [
              332,
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
      "c_1595"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
