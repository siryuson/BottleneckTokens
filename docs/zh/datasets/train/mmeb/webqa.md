# WebQA

> 多跳多模态问答检索数据集，在 MMEB 训练中用于提升跨来源图文证据检索能力。

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
| 原始名称 | WebQA |
| 论文 | WebQA: Multihop and Multimodal QA ([arXiv](https://arxiv.org/abs/2109.00590)) |
| 主要作者 | Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, Yonatan Bisk |
| HuggingFace | [suolyer/webqa](https://huggingface.co/datasets/suolyer/webqa) |
| 许可证 | 参见原始数据集 |

### Schema
原始任务围绕开放域多源证据检索与问答，样本由问题和候选图文证据构成。

### 样本

**查询文本**: `Find a Wikipedia image that answers this question: ...`
**正样本文本**: `<|image_pad|>\nRepresent the given Wikipedia image with related text information: ...`
**负样本文本**: ``

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `WebQA` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `Find a Wikipedia image that answers this question: ...`
**正样本文本**: `<|image_pad|>\nRepresent the given Wikipedia image with related text information: ...`
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

**sample 表 `data/WebQA/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/WebQA.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| path | string | 原始图片路径 |
| image | binary | 图像二进制 |

### 训练侧映射
- query: runtime 最终形态为 `"{text}\n"`；当 query 实际无图像时，会移除误注入的 legacy visual token
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
  "data/WebQA/train.lance": {
    "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: Which body part is found on both a 1913 D Barber half and a 1914 Barber Quarter?\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given Wikipedia image with related text information: 1913-D Barber half obverse.\n",
    "pos_image_path": "images/WebQA/Train/30272327.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/WebQA.lance": {
    "positive_image": {
      "path": "images/WebQA/Train/30272327.jpg",
      "data": {
        "bytes": 16560,
        "sha1": "77cccbcf3dd8",
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
  "data/WebQA/train.lance": {
    "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: In the painting \"Island Women\" does the woman on the far left or the woman on the far right have longer hair?\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given Wikipedia image with related text information: Island Women by Bakusen Tsuchida, 1912, color on silk - National Museum of Modern Art, Tokyo - DSC06665  Tsuchida Bakusen, 1912. (Island Women). Pair of two folding screens, color on silk, each 166.5 x 184 cm. The National Museum of Modern Art, Tokyo (). This artwork is in the public domain because the artist died more than 70 years ago. Photography was permitted in the museum without restriction.\n",
    "pos_image_path": "images/WebQA/Train/30000598.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/WebQA.lance": {
    "positive_image": {
      "path": "images/WebQA/Train/30000598.jpg",
      "data": {
        "bytes": 14658,
        "sha1": "28621ddd75a3",
        "format": "JPEG",
        "size": [
          375,
          256
        ]
      }
    }
  }
}
```
