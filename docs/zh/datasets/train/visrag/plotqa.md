# VisRAG PlotQA

> VisRAG 体系中的图表问答检索训练子集。

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
| 原始名称 | `openbmb/VisRAG-Ret-Train-In-domain-data` (PlotQA subset) |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| 子集名 | PlotQA |
| 许可证 | 参见原始数据集 |

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| query | string | 查询文本 |
| image | struct{bytes, path} | 文档页图像 |
| source | string | 数据来源 |

### 样本

> 完整示例见下方“BToks Lance 格式”小节。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `visrag_train` |
| 实现文件 | `src/vlm2emb/data/datasets/visrag.py` |
| 注册名 | `visrag_train` |
| Lance 路径 | `./data/converted` |
| 过滤条件 | `source = "PlotQA"` |

### Lance 布局

**train.lance（单表）**:

| 列名 | 类型 | 说明 |
|------|------|------|
| query | string | 查询文本 |
| image | struct{bytes: binary, path: string} | 文档页图像 |
| source | string | 来源类别 |

### 训练侧映射
- query: runtime 侧根据 `source` 注入 source-specific query prompt
- positive: runtime 侧根据 `source` 注入 target prompt，并附带单页图像
- negative: 空文本（占位）
- Lance 存储只固定 `query / image / source`，不提前写入 prompt

### 样本

以下示例直接采样自 `./data/converted`，并限定 `source = "PlotQA"`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- `query` 保留 Lance 原值，不提前展开 runtime prompt。

#### 示例 1

```json
{
  "data/train.lance": {
    "query": "How many groups of bars are there ?",
    "image": {
      "path": "257.png",
      "bytes": {
        "bytes": 32739,
        "sha1": "18061e626b38",
        "format": "PNG",
        "size": [
          1319,
          700
        ]
      }
    },
    "source": "PlotQA"
  }
}
```

#### 示例 2

```json
{
  "data/train.lance": {
    "query": "In the year 2010, what is the difference between the number of enrolments in primary education and number of enrolments in secondary general education ?",
    "image": {
      "path": "259.png",
      "bytes": {
        "bytes": 38816,
        "sha1": "c126f78b119b",
        "format": "PNG",
        "size": [
          1224,
          650
        ]
      }
    },
    "source": "PlotQA"
  }
}
```
