# ScienceQA 训练数据集

> 状态：已实现。当前入口为 `scienceqa_train`，conversion 保留原始问答字段，并把图像写入共享 `data/images.lance`。

> 科学问答数据集，用于训练图像问答能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_question_answer` |
| 用途 | 训练 / OOD 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | ScienceQA |
| 论文 | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (NeurIPS 2022) |
| HuggingFace | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| 许可证 | 参见原始数据集 |

### 样本规模

- 默认训练 split: 6,218 个样本
- validation: 2,097 个样本
- test / MMEB-V2 eval 排除: 2,017 个样本

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| image | PIL.Image / bytes | 图像 |
| question | string | 问题文本 |
| answer | string | 答案文本 |

### 样本

见下方 runtime 输出样例。

---

## BToks 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `scienceqa_train` |
| 实现文件 | `src/vlm2emb/data/datasets/scienceqa_train.py` |
| 注册名 | `scienceqa_train` |
| 转换脚本 | `scripts/convert/train/convert_scienceqa_to_lance.py` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/images.lance.path` | string | 图像查表 key |
| `data/images.lance.image` | binary | 图像二进制 |
| `data/{split}.lance.question` | string | 问题 |
| `data/{split}.lance.answer` | string | 答案 |

### 训练侧映射

- query: `<|image_pad|>\nRepresent the given image with the following question: {question}\n`
- positive: `{answer}`
- negative: 空文本（占位）

### 样本

```json
{
  "query": {
    "text": "<|image_pad|>\nRepresent the given image with the following question: Which of these states is farthest north?\n",
    "media": ["<PIL.Image>"]
  },
  "positive": {
    "text": "West Virginia",
    "media": []
  },
  "negative": {
    "text": "",
    "media": []
  }
}
```
