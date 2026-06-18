# HatefulMemes

> 多模态仇恨内容识别数据集，在 MMEB 训练中用于强化图文联合语义与偏见识别能力。

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
| 原始名称 | Hateful Memes Challenge Dataset |
| 论文 | The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes ([arXiv](https://arxiv.org/abs/2005.04790)) |
| 主要作者 | Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, Davide Testuggine |
| HuggingFace | [neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本由表情图像、配套文本与二分类标签构成，强调跨模态仇恨语义判别。

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
| 子集名 | `HatefulMemes` |
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

**sample 表 `data/HatefulMemes/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/HatefulMemes.lance`**:

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
  "data/HatefulMemes/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
    "qry_image_path": "images/HatefulMemes/Train/HatefulMemes_image_0.jpg",
    "pos_text": "Yes",
    "pos_image_path": "",
    "neg_text": "No",
    "neg_image_path": ""
  },
  "data/images/HatefulMemes.lance": {
    "query_image": {
      "path": "images/HatefulMemes/Train/HatefulMemes_image_0.jpg",
      "data": {
        "bytes": 30407,
        "sha1": "661e30f7cc3e",
        "format": "JPEG",
        "size": [
          550,
          366
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/HatefulMemes/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
    "qry_image_path": "images/HatefulMemes/Train/HatefulMemes_image_4250.jpg",
    "pos_text": "Yes",
    "pos_image_path": "",
    "neg_text": "No",
    "neg_image_path": ""
  },
  "data/images/HatefulMemes.lance": {
    "query_image": {
      "path": "images/HatefulMemes/Train/HatefulMemes_image_4250.jpg",
      "data": {
        "bytes": 42128,
        "sha1": "bad78bec169d",
        "format": "JPEG",
        "size": [
          550,
          366
        ]
      }
    }
  }
}
```
