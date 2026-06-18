# VisRAG MP-DocVQA

> VisRAG 体系中的多页文档问答检索训练子集。

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
| 原始名称 | `openbmb/VisRAG-Ret-Train-In-domain-data` (MP-DocVQA subset) |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| 子集名 | MP-DocVQA |
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
| 过滤条件 | `source = "MP-DocVQA"` |

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

以下示例直接采样自 `./data/converted`，并限定 `source = "MP-DocVQA"`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- `query` 保留 Lance 原值，不提前展开 runtime prompt。

#### 示例 1

```json
{
  "data/train.lance": {
    "query": "What is the mean value of children below 15 yrs?",
    "image": {
      "path": "2561.png",
      "bytes": {
        "bytes": 249493,
        "sha1": "7364bb730f86",
        "format": "JPEG",
        "size": [
          1685,
          2187
        ]
      }
    },
    "source": "MP-DocVQA"
  }
}
```

#### 示例 2

```json
{
  "data/train.lance": {
    "query": "What is the dosage of Temik used for SI Method at Planting time?",
    "image": {
      "path": "2563.png",
      "bytes": {
        "bytes": 402856,
        "sha1": "feaaec468e65",
        "format": "JPEG",
        "size": [
          1689,
          2194
        ]
      }
    },
    "source": "MP-DocVQA"
  }
}
```
