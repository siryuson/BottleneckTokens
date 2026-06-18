# Place365 训练数据集

> 场景识别数据集，用于训练图像场景分类能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_classification` |
| 用途 | 训练 / OOD 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Places365 |
| 论文 | Places: A 10 Million Image Database for Scene Recognition (TPAMI 2017) |
| HuggingFace | [dpdl-benchmark/Places365-Validation](https://huggingface.co/datasets/dpdl-benchmark/Places365-Validation) 或 [Andron00e/Places365-custom](https://huggingface.co/datasets/Andron00e/Places365-custom) |
| 许可证 | 参见原始数据集 |

### 样本规模

- `official_train`: `1,803,460` 张图像
- `official_val`: `36,500` 张图像
- 类别数: `365`

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| image | PIL.Image / bytes | 场景图像 |
| label | string | 场景类别标签 |

### 样本

**query.text**: `<|image_pad|>\nIdentify the scene shown in the image\n`
**query.images**: 1 张图像
**positive.text**: `airfield`
**negative.text**: 空

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `place365_train` |
| 实现文件 | `src/vlm2emb/data/datasets/place365_train.py` |
| 注册名 | `place365_train` |
| 转换脚本 | `scripts/convert/train/convert_place365_to_lance.py` |
| 根目录 | `./data/converted` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/images.lance.path` | string | 图像查表 key |
| `data/images.lance.image` | binary | 图像二进制 |
| `data/{split}.lance.path` | string | 图像查表 key |
| `data/{split}.lance.label` | string | 场景类别文本 |
| `data/{split}.lance.class_id` | int | 类别 ID |

### 训练侧映射

- query: `<|image_pad|>\nIdentify the scene shown in the image\n`
- positive: `{label}`
- negative: 空文本（占位）

### Split 策略

- `official_train`: 默认训练视图
- `official_val`: 保留作 held-out 验证视图

### 本地审计结论

- 本地源路径为 `./data/converted`
- 目录同时包含 `train/` 和 `val/`
- `train.txt` 与 `val.txt` 不重叠，因此当前实现按官方两套 split 显式保留，不再像 archive 一样只做目录遍历

### 样本

以下为当前 runtime 默认形态：

```json
{
  "query": {
    "text": "<|image_pad|>\nIdentify the scene shown in the image\n",
    "media": ["<PIL.Image>"]
  },
  "positive": {
    "text": "airfield\n",
    "media": []
  },
  "negative": {
    "text": "",
    "media": []
  }
}
```
