# Country211 训练数据集

> 状态：已实现。当前入口为 `country211_train`，conversion 保留原始 tar / sample key / class 字段，并把图像写入共享 `data/images.lance`。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_classification` |
| 用途 | 训练 / OOD 评测 |
| Runtime | `country211_train` |
| Conversion | `scripts/convert/train/convert_country211_to_lance.py` |
| 默认 split | `official_train_without_mmeb_v2_eval` |

## 来源信息

| 属性 | 值 |
|------|-----|
| 原始名称 | Country211 |
| 论文 | CLIP: Learning Transferable Visual Models From Natural Language Supervision ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) |
| HuggingFace | [clip-benchmark/wds_country211](https://huggingface.co/datasets/clip-benchmark/wds_country211) |
| 许可证 | 参见原始数据集 |

## 行数

| split / table | 行数 |
|---------------|------|
| `data/images.lance` | 52,750 |
| `official_train` | 31,650 |
| `official_train_without_mmeb_v2_eval` | 31,650 |
| `official_test` | 21,100 |

## Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/images.lance.path` | string | 图像查表 key |
| `data/images.lance.image` | binary | 图像二进制 |
| `data/{split}.lance.image_path` | string | 图像查表 key |
| `data/{split}.lance.class_name` | string | 国家名称 |
| `data/{split}.lance.class_index` | int | 类别 ID |
| `data/{split}.lance.source_tar` | string | 原始 tar 文件 |
| `data/{split}.lance.sample_key` | string | 原始样本 key |

## Runtime 输出样例

```json
{
  "query": {
    "text": "<|image_pad|>\nIdentify the country depicted in the image\n",
    "media": ["<PIL.Image>"]
  },
  "positive": {
    "text": "Andorra",
    "media": []
  },
  "negative": {
    "text": "",
    "media": []
  }
}
```
