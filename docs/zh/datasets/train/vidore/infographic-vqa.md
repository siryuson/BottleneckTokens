# ViDoRe Infographic-VQA

> ViDoRe 体系中的图表问答文档检索训练子集。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `document_retrieval` |
| 用途 | 训练 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | `vidore/colpali_train_set` (Infographic-VQA subset) |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| 子集名 | Infographic-VQA |
| 许可证 | 参见原始数据集 |

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| image | struct{bytes, path} | 文档页图像 |
| query | string | 查询文本 |
| answer | string | 正样本文本答案 |

### 样本

> 完整示例见下方“BToks Lance 格式”小节。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `vidore_train` |
| 实现文件 | `src/vlm2emb/data/datasets/vidore.py` |
| 注册名 | `vidore_train` |
| Lance 路径 | `./data/converted` |
| 过滤条件 | `source = "Infographic-VQA"` |

### Lance 布局

**train.lance（单表）**:

| 列名 | 类型 | 说明 |
|------|------|------|
| image | struct{bytes: binary, path: string} | 文档页图像 |
| image_filename | string | 原始图像文件名 |
| query | string | 查询文本 |
| answer | string | 正样本文本答案 |
| source | string | 来源子集 |
| options | string | 上游附加 metadata |
| page | string | 页码或页标识（若有） |
| model | string | 上游模型 metadata |
| prompt | string | 上游 prompt metadata |
| answer_type | string | 答案类型 metadata |

### 训练侧映射
- query: 运行时默认渲染为 `"<|image_pad|> {query}"` + 单页图像
- positive: `answer`（纯文本正样本）
- negative: 空文本（占位）
- `source / page / answer_type` 等字段当前保留为存储 metadata，不直接进入最小训练视图

### 样本

以下示例直接采样自 `./data/converted`，并限定 `source = "Infographic-VQA"`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- 其余字段保留 Lance 原值。

#### 示例 1

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 71446,
        "sha1": "7f5a0b4db620",
        "format": "JPEG",
        "size": [
          590,
          843
        ]
      }
    },
    "image_filename": "e205f5ce26ecf2bd455ed1034eaf7764c873a63056b6e3e41a3a953843c0fbc3",
    "query": "How many tips are mentioned for prevention of coronavirus?\nAnswer briefly.",
    "answer": "10.",
    "source": "Infographic-VQA",
    "options": null,
    "page": null,
    "model": null,
    "prompt": null,
    "answer_type": null
  }
}
```

#### 示例 2

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 190966,
        "sha1": "bd5eef9d78e7",
        "format": "JPEG",
        "size": [
          700,
          1700
        ]
      }
    },
    "image_filename": "9a71009d0e95bff815db2547e23c56b8c73e50ff332d1b722bdbf37c81f187ec",
    "query": "What is the inverse of the percentage of total job advertisements in London?\nGive a very brief answer.",
    "answer": "78.",
    "source": "Infographic-VQA",
    "options": null,
    "page": null,
    "model": null,
    "prompt": null,
    "answer_type": null
  }
}
```
