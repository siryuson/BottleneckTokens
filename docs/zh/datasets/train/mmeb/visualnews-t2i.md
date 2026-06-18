# VisualNews-t2i

> 新闻文本到图像检索数据集，在 MMEB 训练中用于强化新闻语境下的跨模态检索能力。

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
| 原始名称 | VisualNews |
| 论文 | Visual News: Benchmark and Challenges in News Image Captioning ([arXiv](https://arxiv.org/abs/2010.03743)) |
| 主要作者 | Fuxiao Liu, Yinghan Wang, Tianlu Wang, Vicente Ordonez |
| HuggingFace | [FuxiaoLiu/visualnews](https://huggingface.co/datasets/FuxiaoLiu/visualnews) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本由新闻图像、标题/描述与文章元数据构成，可构建文本到图像检索配对。

### 样本

**查询文本**: `Retrieve an image of this news caption. Turkish President Tayyip Erdogan looks on after arriving at ...`
**正样本文本**: `<|image_pad|>\nRepresent the given image.`
**负样本文本**: ``

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `VisualNews_t2i` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `Retrieve an image of this news caption. Turkish President Tayyip Erdogan looks on after arriving at ...`
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

**sample 表 `data/VisualNews_t2i/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/VisualNews_t2i.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| path | string | 原始图片路径 |
| image | binary | 图像二进制 |

### 训练侧映射
- query: runtime 最终形态为 `"{text}\n"`，不带 visual token
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
  "data/VisualNews_t2i/train.lance": {
    "qry": "Retrieve an image of this news caption. Turkish President Tayyip Erdogan looks on after arriving at Esenboga Airport in Ankara Turkey June 8 2015 Turkey faced the prospect of weeks of political turmoil after the ruling AK Party lost its parliamentary majority REUTERSUmit Bektas.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/VisualNews_t2i/Train/washington_post_images_0528_634.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/VisualNews_t2i.lance": {
    "positive_image": {
      "path": "images/VisualNews_t2i/Train/washington_post_images_0528_634.jpg",
      "data": {
        "bytes": 12395,
        "sha1": "ebb586857d69",
        "format": "JPEG",
        "size": [
          376,
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
  "data/VisualNews_t2i/train.lance": {
    "qry": "Retrieve an image of this news caption. Itte Detenamo ca nt make this lift in the clean and jerk His total of 396kg gave him silver while Australia s Damon Kelly took bronze with 388kg.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/VisualNews_t2i/Train/guardian_images_0266_314.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/VisualNews_t2i.lance": {
    "positive_image": {
      "path": "images/VisualNews_t2i/Train/guardian_images_0266_314.jpg",
      "data": {
        "bytes": 14970,
        "sha1": "020c0dbe40cd",
        "format": "JPEG",
        "size": [
          414,
          256
        ]
      }
    }
  }
}
```
