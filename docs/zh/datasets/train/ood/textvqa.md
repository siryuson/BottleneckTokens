# TextVQA 训练数据集

> 状态：已实现。当前入口为 `textvqa_train`，conversion 保留原始 TextVQA 非图像字段，并把图像写入共享 `data/images.lance`。

> 文本问答数据集，用于训练图像中文字识别和问答能力。

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
| 原始名称 | TextVQA |
| 论文 | Towards VQA Models That Can Read ([arXiv:1904.08920](https://arxiv.org/abs/1904.08920)) |
| HuggingFace | [facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) 或 [lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) |
| 许可证 | 参见原始数据集 |

### 样本规模

- 默认训练 split: 34,496 个样本
- 原始 train: 34,602 个样本
- validation: 5,000 个样本
- test: 5,734 个样本
- MMEB-V2 eval 排除: 106 个样本

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
| 数据加载器 | `textvqa_train` |
| 实现文件 | `src/vlm2emb/data/datasets/textvqa_train.py` |
| 注册名 | `textvqa_train` |
| 转换脚本 | `scripts/convert/train/convert_textvqa_to_lance.py` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/images.lance.path` | string | 图像查表 key |
| `data/images.lance.image` | binary | 图像二进制 |
| `data/{split}.lance.image_id` | string | 图像 ID |
| `data/{split}.lance.question` | string | 问题 |
| `data/{split}.lance.answers` | list | 原始答案列表 |
| `data/{split}.lance.ocr_tokens` | list | 原始 OCR token 列表 |

### 训练侧映射

- query: `<|image_pad|>\nRepresent the given image with the following question: {question}\n`
- positive: 第一条可用答案，不主动追加尾部换行
- negative: 空文本（占位）

### 样本

```json
{
  "query": {
    "text": "<|image_pad|>\nRepresent the given image with the following question: what is the brand of phone?\n",
    "media": ["<PIL.Image>"]
  },
  "positive": {
    "text": "nokia",
    "media": []
  },
  "negative": {
    "text": "",
    "media": []
  }
}
```

当前 runtime 与 archive parser 对齐，不把 `ocr_tokens` 拼入 query；这些字段保留在 split 表中，方便后续单独讨论是否需要 OCR-aware transform。
