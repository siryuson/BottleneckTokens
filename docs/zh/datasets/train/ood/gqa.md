# GQA 训练数据集

> 状态：已实现。当前入口为 `gqa_train`，conversion 保留原始问答字段，并把图像写入共享 `data/images.lance`。

> 视觉推理问答数据集，用于训练组合式视觉推理能力。

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
| 原始名称 | GQA |
| 论文 | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering ([arXiv:1902.09506](https://arxiv.org/abs/1902.09506)) |
| HuggingFace | [lmms-lab/GQA](https://huggingface.co/datasets/lmms-lab/GQA) |
| 许可证 | 参见原始数据集 |

### 样本规模

- 图像: 74,256 张
- `train_all` 问题: 14,305,356 个
- `train_balanced` 问题: 943,000 个
- 默认训练 split: `official_train_balanced_without_mmeb_v2_eval`

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
| 数据加载器 | `gqa_train` |
| 实现文件 | `src/vlm2emb/data/datasets/gqa_train.py` |
| 注册名 | `gqa_train` |
| 转换脚本 | `scripts/convert/train/convert_gqa_to_lance.py` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/images.lance.path` | string | 图像查表 key |
| `data/images.lance.image` | binary | 图像二进制 |
| `data/{split}.lance.imageId` | string | 图像 ID |
| `data/{split}.lance.question` | string | 问题 |
| `data/{split}.lance.fullAnswer` | string | 完整答案 |

### Split Views

| Split | 行数 | 说明 |
|------|------:|------|
| `official_train_all` | 14,305,356 | 完整 `train_all_instructions` 源视图 |
| `official_train_all_without_mmeb_v2_eval` | 14,305,356 | 排除 MMEB-V2 eval 后的完整源视图 |
| `official_train_balanced` | 943,000 | 官方 balanced train 视图 |
| `official_train_balanced_without_mmeb_v2_eval` | 943,000 | 默认训练 split |

### 训练侧映射

- query: `<|image_pad|>\nRepresent the given image with the following question: {question}\n`
- positive: `{fullAnswer}`
- negative: 空文本（占位）

### MMEB-V2 overlap audit

- 当前转换使用精确 QA 样本级 key：`imageId + normalized(question)`。
- MMEB-V2 `GQA` eval 有 `1,000` 条 query、`773` 个唯一 image id、`954` 个唯一 normalized question 和 `999` 个唯一 `image_id + question` key。
- GQA `train_all` 有 `14,305,356` 行、`74,256` 个唯一 train image id。`train_balanced` 有 `943,000` 行，覆盖 `72,140` 张图。
- train image id 与 eval image id 交集为 `0`；`imageId + normalized(question)` 交集为 `0`。
- text-only question overlap 很高：train 中 `191,495` 行 question 命中 eval question；`question + fullAnswer` 也有 `68,741` 行命中 eval 正例文本。但这些命中都发生在不同 image id 上，反映 GQA 模板化语言复用，不作为样本级泄漏排除键。
- 因此 all 和 balanced 的 eval-excluded 训练 split 都与各自源视图行数相同，排除表为 `0` 行。这属于已审计结论而不是转换漏排。

### 样本

```json
{
  "query": {
    "text": "<|image_pad|>\nRepresent the given image with the following question: What is on the white wall?\n",
    "media": ["<PIL.Image>"]
  },
  "positive": {
    "text": "The pipe is on the wall.",
    "media": []
  },
  "negative": {
    "text": "",
    "media": []
  }
}
```
