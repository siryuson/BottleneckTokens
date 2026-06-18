# MSCOCO-t2i

> 文本到图像检索数据集，在 MMEB 训练中用于学习自然语言到图像语义空间的对齐。

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
| 原始名称 | MS COCO Captions |
| 论文 | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| 主要作者 | Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本包含图像与配套描述文本，可构建文本检索图像任务。

### 样本

**查询文本**: `Find me an everyday image that matches the given caption: A teddy bear shop is equipped with a door ...`
**正样本文本**: `<|image_pad|>\nRepresent the given image.`
**负样本文本**: ``

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `MSCOCO_t2i` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `Find me an everyday image that matches the given caption: A teddy bear shop is equipped with a door ...`
**正样本文本**: `<|image_pad|>\nRepresent the given image.`
**负样本文本**: ``

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

**sample 表 `data/MSCOCO_t2i/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/MSCOCO_t2i.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| path | string | 原始图片路径 |
| image | binary | 图像二进制 |

### 训练侧映射
- query: runtime 最终形态为 `"{text}\n"`，不带 visual token
- positive: runtime 最终形态为 `"<|image_pad|>\n{text}\n"`，并附带正样本图像
- negative: runtime 最终为空文本
- runtime 默认会把历史 `<|image_1|>` 归一成 `<|image_pad|>`

### 样本

以下示例直接采样自真实训练 Lance 表。

- sample 行保留 Lance 原值。
- 图像行中的 `image` 按 `bytes / sha1 / format / size` 展示摘要，不直接展开原始二进制。

#### 示例 1

```json
{
  "data/MSCOCO_t2i/train.lance": {
    "qry": "Find me an everyday image that matches the given caption: A teddy bear shop is equipped with a door guard teddy and a neighbor teddy above.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/MSCOCO_t2i/Train/COCO_train2014_000000161071.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_t2i.lance": {
    "positive_image": {
      "path": "images/MSCOCO_t2i/Train/COCO_train2014_000000161071.jpg",
      "data": {
        "bytes": 21012,
        "sha1": "6facc7617f93",
        "format": "JPEG",
        "size": [
          256,
          377
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/MSCOCO_t2i/train.lance": {
    "qry": "Find me an everyday image that matches the given caption: A man that is standing on a plate.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/MSCOCO_t2i/Train/COCO_train2014_000000289425.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_t2i.lance": {
    "positive_image": {
      "path": "images/MSCOCO_t2i/Train/COCO_train2014_000000289425.jpg",
      "data": {
        "bytes": 12498,
        "sha1": "b74322679c0b",
        "format": "JPEG",
        "size": [
          333,
          256
        ]
      }
    }
  }
}
```
