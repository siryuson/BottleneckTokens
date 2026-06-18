# DocVQA

> 文档图像问答数据集，在 MMEB 训练中用于提升文档页面理解与问答对齐能力。

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
| 原始名称 | DocVQA |
| 论文 | DocVQA: A Dataset for VQA on Document Images ([arXiv](https://arxiv.org/abs/2007.00398)) |
| 主要作者 | Minesh Mathew, Dimosthenis Karatzas, C.V. Jawahar |
| HuggingFace | [cmarkea/docvqa](https://huggingface.co/datasets/cmarkea/docvqa) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本由文档图像、问题与答案构成，强调文本布局和视觉元素联合理解。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `1/8/93`
**负样本文本**: `398,627.92`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `DocVQA` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `1/8/93`
**负样本文本**: `398,627.92`

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

**sample 表 `data/DocVQA/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/DocVQA.lance`**:

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
  "data/DocVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: what is the date mentioned in this letter?\n",
    "qry_image_path": "images/DocVQA/Train/DocVQA_image_0.jpg",
    "pos_text": "1/8/93",
    "pos_image_path": "",
    "neg_text": "398,627.92",
    "neg_image_path": ""
  },
  "data/images/DocVQA.lance": {
    "query_image": {
      "path": "images/DocVQA/Train/DocVQA_image_0.jpg",
      "data": {
        "bytes": 186082,
        "sha1": "dc1b29161139",
        "format": "JPEG",
        "size": [
          1695,
          2025
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/DocVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Who is the Professor of Food and Nutrition in South Carolina State College?\n",
    "qry_image_path": "images/DocVQA/Train/DocVQA_image_19731.jpg",
    "pos_text": "abernathy, miriam m., ph.d.",
    "pos_image_path": "",
    "neg_text": "schultze",
    "neg_image_path": ""
  },
  "data/images/DocVQA.lance": {
    "query_image": {
      "path": "images/DocVQA/Train/DocVQA_image_19731.jpg",
      "data": {
        "bytes": 438316,
        "sha1": "e3782efdb91e",
        "format": "JPEG",
        "size": [
          1784,
          2273
        ]
      }
    }
  }
}
```
