# VisDial

> 多轮视觉对话场景的文本到图像检索子集。

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
| 原始名称 | Visual Dialog |
| 论文 | Visual Dialog ([arXiv](https://arxiv.org/abs/1611.08669)) |
| 主要作者 | Abhishek Das, Satwik Kottur, José M. F. Moura, Stefan Lee, Dhruv Batra |
| HuggingFace | [HuggingFaceM4/VisDial](https://huggingface.co/datasets/HuggingFaceM4/VisDial) |
| 许可证 | 参见原始数据集 |

### Schema
图像、对话历史、当前轮问题与答案候选。

### 样本

**query.text**: `Represent the given dialogue about an image, which is used for image retrieval: Q:is the photo in co...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## MMEB-V2（VLM2Vec 转换后）
### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| 子集名 | `VisDial` |
| 分割 | `test` |
| 评测解析器 | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### 样本

**query.text**: `Represent the given dialogue about an image, which is used for image retrieval: Q:is the photo in co...`
**query.images**: 0 张图像
**candidate.text**: `<|image_pad|>...`
**candidate.images**: 0 张图像

---

## BToks Lance 格式
### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/image-tasks/VisDial/` |
| task_type | `image_retrieval` |
| 评测模式 | local |

### Lance 布局
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`。

### 样本

以下示例直接采样自 `./data/converted`。

- `text` 字段保留 Lance 原值。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_0`，`negative_candidate_id=c_1455`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Represent the given dialogue about an image, which is used for image retrieval:\nQ:is the photo in color\nA:yes\nQ:is it a professional photo\nA:no\nQ:is it well lit\nA:no\nQ:is it daytime\nA:i don't see windows\nQ:does this look like an adults bedroom\nA:maybe\nQ:is the room clean\nA:no\nQ:can you tell what kind of computer it is\nA:no not really\nQ:is it a flat screen\nA:yes\nQ:what's the desk made out of\nA:cheap plastic or wood\nQ:is there a computer chair\nA:yes\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 31610,
            "sha1": "cce44f06331a",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1455",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 47127,
            "sha1": "11f21f4bcd78",
            "format": "JPEG",
            "size": [
              640,
              480
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
      "c_0"
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
`query_id=q_1`，`positive_candidate_id=c_1`，`negative_candidate_id=c_1835`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Represent the given dialogue about an image, which is used for image retrieval:\nQ:is this in a park\nA:yes, i believe it is\nQ:are there others around\nA:no, she is alone\nQ:does she have a collection bucket\nA:no\nQ:is her hair long\nA:yes, pretty long\nQ:is she wearing a dress\nA:i don't think so, hard to tell\nQ:does she have shoes on\nA:yes, flip flops\nQ:is there grass nearby\nA:yes, everywhere\nQ:is it a sunny day\nA:yes\nQ:are there trees\nA:in the background there are trees\nQ:is the guitar new\nA:i don't think so\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 70266,
            "sha1": "10f601a47c31",
            "format": "JPEG",
            "size": [
              640,
              446
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1835",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 28083,
            "sha1": "0f0703191836",
            "format": "JPEG",
            "size": [
              500,
              375
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
      "c_1"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
