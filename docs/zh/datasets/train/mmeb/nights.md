# NIGHTS

> 夜间场景检索数据集，在 MMEB 训练中用于增强低照度与跨场景视觉检索鲁棒性。

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
| 原始名称 | NIGHTS (M-BEIR task) |
| 论文 | UniIR: Training and Benchmarking Universal Multimodal Information Retrievers ([arXiv](https://arxiv.org/abs/2311.17136)) |
| 主要作者 | Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, Wenhu Chen |
| HuggingFace | [mteb/mbeir_nights_task4](https://huggingface.co/datasets/mteb/mbeir_nights_task4) |
| 许可证 | 参见原始数据集 |

### Schema
原始任务由夜间图像检索对构成，聚焦低光照条件下的视觉匹配与检索。

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
| 子集名 | `NIGHTS` |
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

**sample 表 `data/NIGHTS/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/NIGHTS.lance`**:

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
  "data/NIGHTS/train.lance": {
    "qry": "<|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n",
    "qry_image_path": "images/NIGHTS/Train/ref_000_002.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/NIGHTS/Train/distort_000_002_0.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/NIGHTS.lance": {
    "query_image": {
      "path": "images/NIGHTS/Train/ref_000_002.jpg",
      "data": {
        "bytes": 22602,
        "sha1": "ae9a0e58ce63",
        "format": "JPEG",
        "size": [
          256,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/NIGHTS/Train/distort_000_002_0.jpg",
      "data": {
        "bytes": 20542,
        "sha1": "7a69e8983db1",
        "format": "JPEG",
        "size": [
          256,
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
  "data/NIGHTS/train.lance": {
    "qry": "<|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n",
    "qry_image_path": "images/NIGHTS/Train/ref_050_030.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/NIGHTS/Train/distort_050_030_1.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/NIGHTS.lance": {
    "query_image": {
      "path": "images/NIGHTS/Train/ref_050_030.jpg",
      "data": {
        "bytes": 21330,
        "sha1": "0a8812464f49",
        "format": "JPEG",
        "size": [
          256,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/NIGHTS/Train/distort_050_030_1.jpg",
      "data": {
        "bytes": 19303,
        "sha1": "219b4d899dbf",
        "format": "JPEG",
        "size": [
          256,
          256
        ]
      }
    }
  }
}
```
