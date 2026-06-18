# A-OKVQA

> 结合视觉证据与常识推理的问答数据集，在 MMEB 训练中用于增强复杂视觉问答能力。

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
| 原始名称 | A-OKVQA |
| 论文 | A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge ([arXiv](https://arxiv.org/abs/2206.01718)) |
| 主要作者 | Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, Roozbeh Mottaghi |
| HuggingFace | [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) |
| 许可证 | 参见原始数据集 |

### Schema
原始数据包括图像、问题、答案与解释性信息，面向开放式视觉问答。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `cab`
**负样本文本**: `train`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `A-OKVQA` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `cab`
**负样本文本**: `train`

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

**sample 表 `data/A-OKVQA/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/A-OKVQA.lance`**:

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
  "data/A-OKVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is the man by the bags awaiting?\n",
    "qry_image_path": "images/A-OKVQA/Train/A-OKVQA_image_0.jpg",
    "pos_text": "cab",
    "pos_image_path": "",
    "neg_text": "train",
    "neg_image_path": ""
  },
  "data/images/A-OKVQA.lance": {
    "query_image": {
      "path": "images/A-OKVQA/Train/A-OKVQA_image_0.jpg",
      "data": {
        "bytes": 77100,
        "sha1": "ec31c8d3b31b",
        "format": "JPEG",
        "size": [
          640,
          480
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/A-OKVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Why are they so close together?\n",
    "qry_image_path": "images/A-OKVQA/Train/A-OKVQA_image_8528.jpg",
    "pos_text": "to talk",
    "pos_image_path": "",
    "neg_text": "need directions",
    "neg_image_path": ""
  },
  "data/images/A-OKVQA.lance": {
    "query_image": {
      "path": "images/A-OKVQA/Train/A-OKVQA_image_8528.jpg",
      "data": {
        "bytes": 44068,
        "sha1": "7847910c43c0",
        "format": "JPEG",
        "size": [
          640,
          361
        ]
      }
    }
  }
}
```
