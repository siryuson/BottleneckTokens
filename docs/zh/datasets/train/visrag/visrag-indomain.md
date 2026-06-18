# VisRAG In-domain

> VisRAG 领域内文档检索训练子集，直接转换为 BToks Lance 单表格式，并按 source 注入来源特定提示词。
> `query/image/source` 属于存储事实；source-specific prompt 与 `<|image_pad|>` 处理属于 runtime transform policy。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `document_retrieval` |
| 用途 | 训练 |
| 溯源层数 | 2 层 |
| 训练权重 | 12 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | `openbmb/VisRAG-Ret-Train-In-domain-data` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) |
| 主要作者 | Yushi Hu, Ziyang Luo, Yixin Cao, Shuai Yan, Jiazhen Gu, Fei Huang, Jingren Zhou, Zhoujun Li |
| HuggingFace | [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) |
| 许可证 | 参见原始数据集 |

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| query | string | 查询文本 |
| image | struct{bytes, path} | 文档页图像 |
| source | string | 数据来源（10 类） |

支持的 source（10 类）：NeurIPS Papers、Textbooks、ICML Papers、Manuallib、ArxivQA、ChartQA、MP-DocVQA、InfoVQA、PlotQA、SlideVQA。

### 多来源样本（格式示意）
- `source=NeurIPS Papers`：query 侧会拼接“顶会论文检索”类提示；positive 侧为论文页图像。
- `source=ChartQA`：query 侧会拼接“图表检索”类提示；positive 侧为图表图像。
- `source=MP-DocVQA`：query 侧会拼接“多页文档定位”类提示；positive 侧为文档页图像。

### 样本

**query**: `which are the countries with Chinese speaking population other than China according to this infographic?`
**image**: 文档页图像（infographic 类型）
**source**: `InfoVQA`

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `visrag_train` |
| 实现文件 | `src/vlm2emb/data/datasets/visrag.py` |
| 注册名 | `visrag_train` |
| Lance 路径 | `./data/converted` |

### Lance 布局

**train.lance（单表）**:

| 列名 | 类型 | 说明 |
|------|------|------|
| query | string | 查询文本 |
| image | struct{bytes: binary, path: string} | 文档页图像 |
| source | string | 来源类别 |

### 训练侧映射
- query: runtime 侧渲染为 `"{source_prompt}{query}\n"`
- positive: runtime 侧渲染为 `"<|image_pad|>\n{target_prompt}\n"`，并附带单页图像
- negative: 空文本（占位）
- Lance 存储只固定 `query / image / source`，不提前写入 prompt

### 样本

以下示例直接采样自 `./data/converted`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- `query` 保留 Lance 原值，不提前展开 runtime prompt。

#### 示例 1

```json
{
  "data/train.lance": {
    "query": "which are the countries with Chinese speaking population other than China according to this infographic?",
    "image": {
      "path": "1.png",
      "bytes": {
        "bytes": 262214,
        "sha1": "cac24738166d",
        "format": "JPEG",
        "size": [
          564,
          3040
        ]
      }
    },
    "source": "InfoVQA"
  }
}
```

#### 示例 2

```json
{
  "data/train.lance": {
    "query": "In subfigure (b), which relationship does the blue dotted line indicate?",
    "image": {
      "path": "129.png",
      "bytes": {
        "bytes": 185227,
        "sha1": "57ba31515f18",
        "format": "JPEG",
        "size": [
          1680,
          1484
        ]
      }
    },
    "source": "ArxivQA"
  }
}
```

#### 示例 3

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
