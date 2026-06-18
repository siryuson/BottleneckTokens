# CIRR

> 组合图像检索数据集，在 MMEB 训练中用于学习“图像 + 文本修改”驱动的检索能力。

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
| 原始名称 | CIRR (Compose Image Retrieval on Real-life images) |
| 论文 | Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models ([arXiv](https://arxiv.org/abs/2108.04024)) |
| 主要作者 | Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould |
| HuggingFace | [mteb/mbeir_cirr_task7](https://huggingface.co/datasets/mteb/mbeir_cirr_task7) |
| 许可证 | 参见原始数据集 |

### Schema
原始任务输入由参考图像与文本修改描述构成，目标是在候选图库中检索符合修改语义的图像。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `<|image_pad|>\nRepresent the given image.`
**负样本文本**: ``

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `CIRR` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
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

**sample 表 `data/CIRR/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/CIRR.lance`**:

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
  "data/CIRR/train.lance": {
    "qry": "<|image_1|>\nGiven an image, find a similar everyday image with the described changes: Many hamster together playing in a different background.\n",
    "qry_image_path": "images/CIRR/Train/train-11991-2-img1.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/CIRR/Train/train-5270-1-img0.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/CIRR.lance": {
    "query_image": {
      "path": "images/CIRR/Train/train-11991-2-img1.jpg",
      "data": {
        "bytes": 28519,
        "sha1": "f7f58445f513",
        "format": "JPEG",
        "size": [
          519,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/CIRR/Train/train-5270-1-img0.jpg",
      "data": {
        "bytes": 19069,
        "sha1": "7d7a824bd33c",
        "format": "JPEG",
        "size": [
          341,
          256
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/CIRR/train.lance": {
    "qry": "<|image_1|>\nGiven an image, find a similar everyday image with the described changes: Take narrow wide coverage of haulted trucks with blue sky effect exclusively.\n",
    "qry_image_path": "images/CIRR/Train/train-11910-1-img1.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/CIRR/Train/train-8721-2-img1.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/CIRR.lance": {
    "query_image": {
      "path": "images/CIRR/Train/train-11910-1-img1.jpg",
      "data": {
        "bytes": 19092,
        "sha1": "abfc09e46dcc",
        "format": "JPEG",
        "size": [
          405,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/CIRR/Train/train-8721-2-img1.jpg",
      "data": {
        "bytes": 16622,
        "sha1": "d98540a3af62",
        "format": "JPEG",
        "size": [
          341,
          256
        ]
      }
    }
  }
}
```
