# OK-VQA

> 需要外部知识的视觉问答数据集，在 MMEB 训练中用于提升图像问答的知识推理能力。

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
| 原始名称 | OK-VQA |
| 论文 | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge ([arXiv](https://arxiv.org/abs/1906.00067)) |
| 主要作者 | Kenneth Marino, Mohammad Rastegari, Ali Farhadi, Roozbeh Mottaghi |
| HuggingFace | [HuggingFaceM4/OK-VQA](https://huggingface.co/datasets/HuggingFaceM4/OK-VQA) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本通常包含图像、问题文本与答案集合，强调视觉与外部知识结合的问答。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `pony tail`
**负样本文本**: `husky`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `OK-VQA` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `pony tail`
**负样本文本**: `husky`

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

**sample 表 `data/OK-VQA/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/OK-VQA.lance`**:

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
  "data/OK-VQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is the hairstyle of the blond called?\n",
    "qry_image_path": "images/OK-VQA/Train/OK-VQA_image_0.jpg",
    "pos_text": "pony tail",
    "pos_image_path": "",
    "neg_text": "husky",
    "neg_image_path": ""
  },
  "data/images/OK-VQA.lance": {
    "query_image": {
      "path": "images/OK-VQA/Train/OK-VQA_image_0.jpg",
      "data": {
        "bytes": 55238,
        "sha1": "4ebb8e070581",
        "format": "JPEG",
        "size": [
          640,
          479
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/OK-VQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What type of numerals are the numbers on this clock?\n",
    "qry_image_path": "images/OK-VQA/Train/OK-VQA_image_4504.jpg",
    "pos_text": "roman",
    "pos_image_path": "",
    "neg_text": "small town",
    "neg_image_path": ""
  },
  "data/images/OK-VQA.lance": {
    "query_image": {
      "path": "images/OK-VQA/Train/OK-VQA_image_4504.jpg",
      "data": {
        "bytes": 74882,
        "sha1": "f19e67f168a8",
        "format": "JPEG",
        "size": [
          610,
          640
        ]
      }
    }
  }
}
```
