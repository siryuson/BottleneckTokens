# MSCOCO-i2t

> 图像到文本检索数据集，在 MMEB 训练中用于增强图像语义到文本语义的映射能力。

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
原始样本由图像及多条描述组成，可用于图像检索文本任务。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `A skateboarder in mid air following a jump form cincrete.`
**负样本文本**: ``

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `MSCOCO_i2t` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `A skateboarder in mid air following a jump form cincrete.`
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

**sample 表 `data/MSCOCO_i2t/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/MSCOCO_i2t.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| path | string | 原始图片路径 |
| image | binary | 图像二进制 |

### 训练侧映射
- query: `qry` + `qry_image_path` 对应图像（若有）
- positive: `pos_text` + `pos_image_path` 对应图像（若有）
- negative: `neg_text` + `neg_image_path` 对应图像（若有）
- runtime 默认会把历史 `<|image_1|>` 归一成 `<|image_pad|>`

### 样本

以下示例直接采样自真实训练 Lance 表。

- sample 行保留 Lance 原值。
- 图像行中的 `image` 按 `bytes / sha1 / format / size` 展示摘要，不直接展开原始二进制。

#### 示例 1

```json
{
  "data/MSCOCO_i2t/train.lance": {
    "qry": "<|image_1|>\nFind an image caption describing the given everyday image.\n",
    "qry_image_path": "images/MSCOCO_i2t/Train/COCO_train2014_000000376224.jpg",
    "pos_text": "A skateboarder in mid air following a jump form cincrete.",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_i2t.lance": {
    "query_image": {
      "path": "images/MSCOCO_i2t/Train/COCO_train2014_000000376224.jpg",
      "data": {
        "bytes": 13478,
        "sha1": "c105ef7a3d14",
        "format": "JPEG",
        "size": [
          256,
          383
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/MSCOCO_i2t/train.lance": {
    "qry": "<|image_1|>\nFind an image caption describing the given everyday image.\n",
    "qry_image_path": "images/MSCOCO_i2t/Train/COCO_train2014_000000445323.jpg",
    "pos_text": "A group of sailors sitting in front of a control area.",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_i2t.lance": {
    "query_image": {
      "path": "images/MSCOCO_i2t/Train/COCO_train2014_000000445323.jpg",
      "data": {
        "bytes": 21062,
        "sha1": "8efc7c885cbb",
        "format": "JPEG",
        "size": [
          420,
          256
        ]
      }
    }
  }
}
```
