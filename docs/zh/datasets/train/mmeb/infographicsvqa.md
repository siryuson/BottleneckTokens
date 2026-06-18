# InfographicsVQA

> 信息图视觉问答数据集，在 MMEB 训练中用于强化图文混排与数据可视化问答能力。

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
| 原始名称 | InfographicVQA |
| 论文 | InfographicVQA ([arXiv](https://arxiv.org/abs/2104.12756)) |
| 主要作者 | Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, C.V Jawahar |
| HuggingFace | [vidore/infovqa_test_subsampled_beir](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir) |
| 许可证 | 参见原始数据集 |

### Schema
原始样本包含信息图图像、问题与答案，覆盖版式理解、文本读取与简单推理。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `serif fonts`
**负样本文本**: `his, he, hers`

---

## MMEB-train（VLM2Vec 转换后）

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| 子集名 | `InfographicsVQA` |
| 分割 | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`。

### 样本

**查询文本**: `<|image_pad|>...`
**正样本文本**: `serif fonts`
**负样本文本**: `his, he, hers`

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

**sample 表 `data/InfographicsVQA/train.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| qry | string | 查询文本（原始历史 token 仍可能是 `<|image_1|>`） |
| qry_image_path | string | 查询图像路径 |
| pos_text | string | 正样本文本 |
| pos_image_path | string | 正样本图像路径，可为空 |
| neg_text | string | 负样本文本，可为空 |
| neg_image_path | string | 负样本图像路径，可为空 |

**图像表 `data/images/InfographicsVQA.lance`**:

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
  "data/InfographicsVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Which type of fonts offer better readability in printed works?\n",
    "qry_image_path": "images/InfographicsVQA/Train/20471.jpeg",
    "pos_text": "serif fonts",
    "pos_image_path": "",
    "neg_text": "his, he, hers",
    "neg_image_path": ""
  },
  "data/images/InfographicsVQA.lance": {
    "query_image": {
      "path": "images/InfographicsVQA/Train/20471.jpeg",
      "data": {
        "bytes": 385473,
        "sha1": "a90f538a4680",
        "format": "JPEG",
        "size": [
          596,
          5107
        ]
      }
    }
  }
}
```

#### 示例 2

```json
{
  "data/InfographicsVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is the total percentage of workers who are not an established self-employed(as a main job)?\n",
    "qry_image_path": "images/InfographicsVQA/Train/10461.jpeg",
    "pos_text": "18%",
    "pos_image_path": "",
    "neg_text": "1984, 1992",
    "neg_image_path": ""
  },
  "data/images/InfographicsVQA.lance": {
    "query_image": {
      "path": "images/InfographicsVQA/Train/10461.jpeg",
      "data": {
        "bytes": 729562,
        "sha1": "7fa4535ea4e9",
        "format": "JPEG",
        "size": [
          2263,
          4013
        ]
      }
    }
  }
}
```
