# N24News

> 多模态新闻分类数据集，在 MMEB 训练中补充新闻语境下的图文类别判别能力。

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
| 原始名称 | N24News |
| 论文 | N24News: A New Dataset for Multimodal News Classification ([arXiv](https://arxiv.org/abs/2108.13327)) |
| 主要作者 | Zhen Wang, Xu Shan, Xiangxie Zhang, Jie Yang |
| HuggingFace | [linxy/Multimodal-N24News](https://huggingface.co/datasets/linxy/Multimodal-N24News) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本包含新闻图像、新闻文本与24类标签，用于多模态新闻分类。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `Theater`
**负样本文本**: `Global Business`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `N24News` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `Theater`
**负样本文本**: `Global Business`

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

**sample 表 `data/N24News/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/N24News.lance`**:

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
  "data/N24News/train.lance": {
    "qry": "<|image_1|>\nRepresent the given news image with the following caption for domain classification: Danai Gurira and André Holland in a theater at New York University, where they met in the Tisch Graduate Acting Program.\n",
    "qry_image_path": "images/N24News/Train/N24News_image_0.jpg",
    "pos_text": "Theater",
    "pos_image_path": "",
    "neg_text": "Global Business",
    "neg_image_path": ""
  },
  "data/images/N24News.lance": {
    "query_image": {
      "path": "images/N24News/Train/N24News_image_0.jpg",
      "data": {
        "bytes": 49525,
        "sha1": "8fad5759eba1",
        "format": "JPEG",
        "size": [
          1050,
          550
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/N24News/train.lance": {
    "qry": "<|image_1|>\nRepresent the given news image with the following caption for domain classification: The face of a volunteer, Miguel Grisanti, is used to cloak the real subject in &ldquo;Welcome to Chechnya.&rdquo;\n",
    "qry_image_path": "images/N24News/Train/N24News_image_24494.jpg",
    "pos_text": "Movies",
    "pos_image_path": "",
    "neg_text": "Music",
    "neg_image_path": ""
  },
  "data/images/N24News.lance": {
    "query_image": {
      "path": "images/N24News/Train/N24News_image_24494.jpg",
      "data": {
        "bytes": 48163,
        "sha1": "7370dbf65288",
        "format": "JPEG",
        "size": [
          1050,
          550
        ]
      }
    }
  }
}
```
