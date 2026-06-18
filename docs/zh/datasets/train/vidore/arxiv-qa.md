# ViDoRe ArXiv QA

> ViDoRe 体系中的 ArXiv 论文问答检索训练子集。

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
| 原始名称 | `vidore/colpali_train_set` (arxiv_qa subset) |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| 子集名 | arxiv_qa |
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
| 过滤条件 | `source = "arxiv_qa"` |

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

以下示例直接采样自 `./data/converted`，并限定 `source = "arxiv_qa"`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- 其余字段保留 Lance 原值。

#### 示例 1

```json
{
  "data/train.lance": {
    "image": {
      "path": "1810.07757_2.jpg",
      "bytes": {
        "bytes": 167632,
        "sha1": "229b31fafc52",
        "format": "JPEG",
        "size": [
          1000,
          1600
        ]
      }
    },
    "image_filename": "images/1810.07757_2.jpg",
    "query": "Comparing panels a, b, c, and d, which statement best describes the data variance?",
    "answer": "D",
    "source": "arxiv_qa",
    "options": "['A. The variance of the data decreases from panel a to panel d.', 'B. The variance of the data increases from panel a to panel d.', 'C. The data presents no variance in any of the panels.', 'D. The variance of the data is inconsistent across the panels.', '-']",
    "page": "",
    "model": "gpt4V",
    "prompt": "",
    "answer_type": null
  }
}
```

#### 示例 2

```json
{
  "data/train.lance": {
    "image": {
      "path": "1711.09320_2.jpg",
      "bytes": {
        "bytes": 187866,
        "sha1": "d17043b3b2fa",
        "format": "JPEG",
        "size": [
          1705,
          1592
        ]
      }
    },
    "image_filename": "images/1711.09320_2.jpg",
    "query": "What trend can be observed in figure a when the strain (ε) is increased from 0% to 8%?",
    "answer": "C",
    "source": "arxiv_qa",
    "options": "['A. The value of γ_sl increases linearly.', 'B. The value of γ_sl remains constant.', 'C. The value of γ_sv decreases linearly.', 'D. The value of γ_sv increases exponentially.', '-']",
    "page": "",
    "model": "gpt4V",
    "prompt": "",
    "answer_type": null
  }
}
```
