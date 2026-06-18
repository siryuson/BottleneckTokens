# ChartQA

> 图表问答数据集，在 MMEB 训练中用于增强图表视觉解析与数值逻辑问答能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_question_answer` |
| 用途 | 训练 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | ChartQA |
| 论文 | ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning ([arXiv](https://arxiv.org/abs/2203.10244)) |
| 主要作者 | Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, Enamul Hoque |
| HuggingFace | [ahmed-masry/ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) |
| 许可证 | 参见原始数据集 |

### Schema
原始数据由图表图像、问题文本与答案构成，关注图表读取与视觉逻辑推理。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `Yes`
**负样本文本**: `No`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `ChartQA` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `Yes`
**负样本文本**: `No`

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

**sample 表 `data/ChartQA/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/ChartQA.lance`**:

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
  "data/ChartQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Is the value of Favorable 38 in 2015?\n",
    "qry_image_path": "images/ChartQA/Train/ChartQA_image_0.jpg",
    "pos_text": "Yes",
    "pos_image_path": "",
    "neg_text": "No",
    "neg_image_path": ""
  },
  "data/images/ChartQA.lance": {
    "query_image": {
      "path": "images/ChartQA/Train/ChartQA_image_0.jpg",
      "data": {
        "bytes": 20857,
        "sha1": "3865210fd03c",
        "format": "JPEG",
        "size": [
          422,
          359
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/ChartQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: How much did Spain spend on clothing between 2007 and 2019?\n",
    "qry_image_path": "images/ChartQA/Train/ChartQA_image_14149.jpg",
    "pos_text": "22605",
    "pos_image_path": "",
    "neg_text": "4304",
    "neg_image_path": ""
  },
  "data/images/ChartQA.lance": {
    "query_image": {
      "path": "images/ChartQA/Train/ChartQA_image_14149.jpg",
      "data": {
        "bytes": 47615,
        "sha1": "e0850c13c6ea",
        "format": "JPEG",
        "size": [
          800,
          557
        ]
      }
    }
  }
}
```
