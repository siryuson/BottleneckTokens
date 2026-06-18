# MSCOCO

> 基于 COCO 的视觉定位子集，在 MMEB 训练中用于学习图文短语与图像区域对齐能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_visual_grounding` |
| 用途 | 训练 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | MS COCO |
| 论文 | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| 主要作者 | Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本由图像、对象与描述信息组成，可构建短语到视觉区域的定位学习样本。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `<|image_pad|>\nRepresent the given cropped image of the object`
**负样本文本**: ``

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `MSCOCO` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `<|image_pad|>\nRepresent the given cropped image of the object`
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

**sample 表 `data/MSCOCO/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/MSCOCO.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| path | string | 原始图片路径 |
| image | binary | 图像二进制 |

### 训练侧映射
- query: runtime 最终形态为 `"<|image_pad|>\n{text}\n"`，并附带查询图像
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
  "data/MSCOCO/train.lance": {
    "qry": "<|image_1|>\nSelect the portion of the image that isolates the object labeled as \"hot dog\"\n",
    "qry_image_path": "images/MSCOCO/Train/000000558840.jpg",
    "pos_text": "<|image_1|>\nRepresent the given cropped image of the object",
    "pos_image_path": "images/MSCOCO/Train/558840_object_hot dog.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO.lance": {
    "query_image": {
      "path": "images/MSCOCO/Train/000000558840.jpg",
      "data": {
        "bytes": 173812,
        "sha1": "b4c1b5ccc00c",
        "format": "JPEG",
        "size": [
          640,
          427
        ]
      }
    },
    "positive_image": {
      "path": "images/MSCOCO/Train/558840_object_hot dog.jpg",
      "data": {
        "bytes": 4024,
        "sha1": "78fbc8b3eb85",
        "format": "JPEG",
        "size": [
          77,
          70
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/MSCOCO/train.lance": {
    "qry": "<|image_1|>\nSelect the portion of the image that isolates the object labeled as \"person\"\n",
    "qry_image_path": "images/MSCOCO/Train/000000170389.jpg",
    "pos_text": "<|image_1|>\nRepresent the given cropped image of the object",
    "pos_image_path": "images/MSCOCO/Train/170389_object_person.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO.lance": {
    "query_image": {
      "path": "images/MSCOCO/Train/000000170389.jpg",
      "data": {
        "bytes": 133065,
        "sha1": "cfde9d220814",
        "format": "JPEG",
        "size": [
          375,
          500
        ]
      }
    },
    "positive_image": {
      "path": "images/MSCOCO/Train/170389_object_person.jpg",
      "data": {
        "bytes": 819,
        "sha1": "fe2b085b6d52",
        "format": "JPEG",
        "size": [
          4,
          18
        ]
      }
    }
  }
}
```
