# ViDoRe TAT-DQA

> ViDoRe 体系中的表格问答文档检索训练子集。

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
| 原始名称 | `vidore/colpali_train_set` (tatdqa subset) |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| 子集名 | tatdqa |
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
| 过滤条件 | `source = "tatdqa"` |

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

以下示例直接采样自 `./data/converted`，并限定 `source = "tatdqa"`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- 其余字段保留 Lance 原值。

#### 示例 1

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 516804,
        "sha1": "b28528c51549",
        "format": "PNG",
        "size": [
          1650,
          2150
        ]
      }
    },
    "image_filename": "data/downloaded_datasets/tatdqa/train/7b920fbe828615563dcc4230356c0282.pdf",
    "query": "What was the increase in income before income taxes in Semiconductor Test driven by?",
    "answer": "['by an increase in semiconductor tester sales for 5G infrastructure and image sensors, partially offset by a decrease in sales in the automotive and analog test segments.']",
    "source": "tatdqa",
    "options": null,
    "page": "1",
    "model": "",
    "prompt": "",
    "answer_type": "span"
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
        "bytes": 274117,
        "sha1": "9a10a7ab0b91",
        "format": "PNG",
        "size": [
          1700,
          2200
        ]
      }
    },
    "image_filename": "data/downloaded_datasets/tatdqa/train/d775277402669fd93e81ed268607ba0c.pdf",
    "query": "What was the % change in the free cash flow from 2017 to 2018?",
    "answer": "-456.96",
    "source": "tatdqa",
    "options": null,
    "page": "1",
    "model": "",
    "prompt": "",
    "answer_type": "arithmetic"
  }
}
```
