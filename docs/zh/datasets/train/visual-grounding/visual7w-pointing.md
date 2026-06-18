# Visual7W Pointing 训练数据集

> 视觉指代问答数据集，用于训练视觉定位能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_visual_grounding` |
| 用途 | 训练 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Visual7W Pointing |
| 论文 | Visual7W: Grounded Question Answering in Images ([arXiv:1511.03416](https://arxiv.org/abs/1511.03416)) |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| 许可证 | 参见原始数据集 |

### 样本规模

- `pointing/train`: `93,813` 个样本
- `pointing/val`: `36,990` 个样本
- `pointing/test`: `57,265` 个样本

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| image | PIL.Image / bytes | 原图 |
| question | string | 指向式问题 |
| answer | box id | 正确目标框 ID |
| multiple_choices | list[box id] | 其余候选框 ID |

### 样本

**query.text**: `<|image_pad|>\nSelect the portion of the image that answers the question "Which door is behind a person sitting on a bench?"\n`
**query.images**: 1 张原图
**positive.text**: `<|image_pad|>\nRepresent the given cropped image of the object.\n`
**positive.images**: 1 张正确裁剪框图像
**negative.text**: ``

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `visual7w_pointing_train` |
| 实现文件 | `src/vlm2emb/data/datasets/visual7w_pointing.py` |
| 注册名 | `visual7w_pointing_train` |
| 转换脚本 | `scripts/convert/train/convert_visual7w_pointing_to_lance.py` |
| 根目录 | `./data/converted` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/images.lance.path` | string | 原图或裁剪图 lookup key |
| `data/images.lance.image` | binary | 原图或裁剪图字节 |
| `data/{split}.lance.sample_id` | string | 样本 ID |
| `data/{split}.lance.question` | string | 指向式问题 |
| `data/{split}.lance.query_image_path` | string | 原图 lookup key |
| `data/{split}.lance.positive_image_path` | string | 正确裁剪图 lookup key |
| `data/{split}.lance.answer_box_id` | int | 正确框 ID |
| `data/{split}.lance.answer_name` | string | 正确框语义名称 |
| `data/{split}.lance.crop_*` | int | 原始框位置与尺寸 |

### 训练侧映射

- query: `<|image_pad|>\nSelect the portion of the image that answers the question "{question}"\n`
- positive: `<|image_pad|>\nRepresent the given cropped image of the object.\n`
- negative: 空文本（占位）

### Split 策略

- `official_train_without_mmeb_v2_eval`: 默认训练视图；当前等同于 `official_train`
- `official_train`: 原始训练 split
- `official_val`: 保留用于分析 / 调参
- `official_test`: 保留用于完整性与审计，默认不参与训练

### 与当前 eval 的关系

- 当前 `MMEB-V2 / Visual7W-Pointing` 高度疑似来自原始 `pointing/test` 的 `1000` 条子集。
- 因此当前仓库默认**不**把 `official_test` 接入训练配置。

### 样本

以下为当前 runtime 默认形态：

```json
{
  "query": {
    "text": "<|image_pad|>\nSelect the portion of the image that answers the question \"Which door is behind a person sitting on a bench?\"\n",
    "media": ["<image>"]
  },
  "positive": {
    "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
    "media": ["<image>"]
  },
  "negative": {
    "text": "",
    "media": []
  }
}
```
