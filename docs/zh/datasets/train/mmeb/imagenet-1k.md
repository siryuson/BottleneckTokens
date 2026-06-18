# ImageNet-1K

> 经典大规模图像分类数据集，在 MMEB 训练中提供通用视觉语义判别能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_classification` |
| 用途 | 训练 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | ImageNet (ILSVRC 2012) |
| 论文 | ImageNet Large Scale Visual Recognition Challenge ([arXiv](https://arxiv.org/abs/1409.0575)) |
| 主要作者 | Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, Li Fei-Fei |
| HuggingFace | [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本通常由图像与类别标签组成，面向大规模图像分类任务。

### 样本

**查询文本**: `<|image_pad|>\nRepresent the given image for classification`
**正样本文本**: `plane, carpenter's plane, woodworking plane`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `ImageNet_1K` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

| 字段 | 值 |
|------|-----|
| qry | `<|image_pad|>\nRepresent the given image for classification` |
| qry_image_path | `n02690373_16364` |
| pos_text | `plane, carpenter's plane, woodworking plane` |
| pos_image_path | `n02690373_16364` |
| neg_text | (空) |
| neg_image_path | (空) |

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

**sample 表 `data/ImageNet_1K/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/ImageNet_1K.lance`**:

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
  "data/ImageNet_1K/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for classification\n",
    "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
    "pos_text": "plane, carpenter's plane, woodworking plane",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/ImageNet_1K.lance": {
    "query_image": {
      "path": "images/ImageNet_1K/Train/image_0.jpg",
      "data": {
        "bytes": 32329,
        "sha1": "5e56f40c6b28",
        "format": "JPEG",
        "size": [
          817,
          363
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/ImageNet_1K/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for classification\n",
    "qry_image_path": "images/ImageNet_1K/Train/image_50000.jpg",
    "pos_text": "mountain tent",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/ImageNet_1K.lance": {
    "query_image": {
      "path": "images/ImageNet_1K/Train/image_50000.jpg",
      "data": {
        "bytes": 39185,
        "sha1": "746c295a5492",
        "format": "JPEG",
        "size": [
          500,
          375
        ]
      }
    }
  }
}
```
