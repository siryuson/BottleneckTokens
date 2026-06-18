# Visual7W

> 基于图像与多种问答形式的视觉问答数据集，在 MMEB 训练中用于提升细粒度视觉问答能力。

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
| 原始名称 | Visual7W |
| 论文 | Visual7W: Grounded Question Answering in Images ([arXiv](https://arxiv.org/abs/1511.03416)) |
| 主要作者 | Yuke Zhu, Oliver Groth, Michael Bernstein, Li Fei-Fei |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本通常包含图像、问题、候选答案与标注答案，可用于多项视觉问答子任务。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `Fox's Ginger Biscuits.`
**负样本文本**: `Mac's Macaroni Hut.`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `Visual7W` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `Fox's Ginger Biscuits.`
**负样本文本**: `Mac's Macaroni Hut.`

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

**sample 表 `data/Visual7W/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/Visual7W.lance`**:

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
  "data/Visual7W/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is written on the white square on the bus?\n",
    "qry_image_path": "images/Visual7W/Train/v7w_2359302.jpg",
    "pos_text": "Fox's Ginger Biscuits.",
    "pos_image_path": "",
    "neg_text": "Mac's Macaroni Hut.",
    "neg_image_path": ""
  },
  "data/images/Visual7W.lance": {
    "query_image": {
      "path": "images/Visual7W/Train/v7w_2359302.jpg",
      "data": {
        "bytes": 40675,
        "sha1": "e98ce8127a04",
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

#### 示例 2

```json
{
  "data/Visual7W/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Where is this picture taken?\n",
    "qry_image_path": "images/Visual7W/Train/v7w_2411048.jpg",
    "pos_text": "On a railway.",
    "pos_image_path": "",
    "neg_text": "On a dirt road.",
    "neg_image_path": ""
  },
  "data/images/Visual7W.lance": {
    "query_image": {
      "path": "images/Visual7W/Train/v7w_2411048.jpg",
      "data": {
        "bytes": 44120,
        "sha1": "0aa98dcbc8df",
        "format": "JPEG",
        "size": [
          500,
          331
        ]
      }
    }
  }
}
```
