# VisDial

> 多轮视觉对话数据集，在 MMEB 训练中用于增强对话式图文检索与语境理解能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_retrieval` |
| 用途 | 训练 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Visual Dialog |
| 论文 | Visual Dialog ([arXiv](https://arxiv.org/abs/1611.08669)) |
| 主要作者 | Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M. F. Moura, Devi Parikh, Dhruv Batra |
| HuggingFace | [HuggingFaceM4/VisDial](https://huggingface.co/datasets/HuggingFaceM4/VisDial) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本包含图像、对话历史、当前问题与候选回答，构成多轮视觉对话检索场景。

### 样本

**查询文本**: `Represent the given dialogue about an image, which is used for image retrieval: Q:is this a child or...`
**正样本文本**: `<|image_pad|>\nRepresent the given image`
**负样本文本**: `<|image_pad|>`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `VisDial` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `Represent the given dialogue about an image, which is used for image retrieval: Q:is this a child or...`
**正样本文本**: `<|image_pad|>\nRepresent the given image`
**负样本文本**: `<|image_pad|>`

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `mmeb_train` |
| 实现文件 | `src/vlm2emb/data/datasets/mmeb_train.py` |
| 注册名 | `mmeb_train` |
| sample 表路径 | `./data/converted` |
| 图像表路径 | `./data/converted` |

### Lance 布局

**sample 表 `data/VisDial/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/VisDial.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| path | string | 原始图片路径 |
| image | binary | 图像二进制 |

### 训练侧映射
- query: runtime 最终形态为 `"{text}\n"`，不带 visual token
- positive: runtime 最终形态为 `"<|image_pad|>\n{text}\n"`，并附带正样本图像
- negative: 若负样本只有图像没有文本，runtime 会补成 `"<|image_pad|>\n"`
- runtime 默认会把历史 `<|image_1|>` 归一成 `<|image_pad|>`

### 样本

以下示例直接采样自真实训练 Lance 表。

- sample 行保留 Lance 原值。
- 图像行中的 `image` 按 `bytes / sha1 / format / size` 展示摘要，不直接展开原始二进制。

#### 示例 1

```json
{
  "data/VisDial/train.lance": {
    "qry": "Represent the given dialogue about an image, which is used for image retrieval: Q:is this a child or adult\nA:adult\nQ:male or female\nA:male\nQ:are they inside or outside\nA:inside\nQ:are they laying on the floor\nA:yes, but there is a blanket in between them and the floor\nQ:is the floor carpeted or wooden\nA:it is tile\nQ:what color is the blanket\nA:red and white\nQ:what color is the tile\nA:orange red\nQ:what breed is the dog\nA:boxer\nQ:does the dog look healthy and happy\nA:yes\nQ:what color is the dog\nA:tan\n\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image\n",
    "pos_image_path": "images/VisDial/Train/VisDial_image_0.jpg",
    "neg_text": "",
    "neg_image_path": "images/VisDial/Train/VisDial_image_61616.jpg"
  },
  "data/images/VisDial.lance": {
    "positive_image": {
      "path": "images/VisDial/Train/VisDial_image_0.jpg",
      "data": {
        "bytes": 57679,
        "sha1": "fe7b7d48a7ac",
        "format": "JPEG",
        "size": [
          612,
          612
        ]
      }
    },
    "negative_image": {
      "path": "images/VisDial/Train/VisDial_image_61616.jpg",
      "data": {
        "bytes": 53400,
        "sha1": "a5968c4411f3",
        "format": "JPEG",
        "size": [
          640,
          425
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/VisDial/train.lance": {
    "qry": "Represent the given dialogue about an image, which is used for image retrieval: Q:does it look like inside a store\nA:i think it is the display window for a store, facing the street\nQ:are the vases all on 1 shelf\nA:yes\nQ:does it look like an antique store\nA:no\nQ:are there any people looking in\nA:no\nQ:does it look like a nice day\nA:no\nQ:are the vases big\nA:some are\nQ:are they all different colors\nA:yes\nQ:is anything else on shelf with them\nA:no\nQ:can you see any price tags\nA:no\nQ:can you see the name of store\nA:no\n\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image\n",
    "pos_image_path": "images/VisDial/Train/VisDial_image_61643.jpg",
    "neg_text": "",
    "neg_image_path": "images/VisDial/Train/VisDial_image_50556.jpg"
  },
  "data/images/VisDial.lance": {
    "positive_image": {
      "path": "images/VisDial/Train/VisDial_image_61643.jpg",
      "data": {
        "bytes": 66555,
        "sha1": "27b4c2a93271",
        "format": "JPEG",
        "size": [
          433,
          640
        ]
      }
    },
    "negative_image": {
      "path": "images/VisDial/Train/VisDial_image_50556.jpg",
      "data": {
        "bytes": 28879,
        "sha1": "0fa051d7d99d",
        "format": "JPEG",
        "size": [
          427,
          640
        ]
      }
    }
  }
}
```
