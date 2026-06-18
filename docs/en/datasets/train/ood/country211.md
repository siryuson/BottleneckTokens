# Country211 Training Dataset

> Status: implemented. The current runtime is `country211_train`. Conversion preserves the original tar / sample key / class fields and stores images in the shared `data/images.lance` table.

| Field | Value |
|-------|-------|
| Modality | image |
| Task Type | `image_classification` |
| Usage | Training / OOD evaluation |
| Runtime | `country211_train` |
| Conversion | `scripts/convert/train/convert_country211_to_lance.py` |
| Default Split | `official_train_without_mmeb_v2_eval` |

## Source Info

| Field | Value |
|-------|-------|
| Original Name | Country211 |
| Paper | CLIP: Learning Transferable Visual Models From Natural Language Supervision ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) |
| HuggingFace | [clip-benchmark/wds_country211](https://huggingface.co/datasets/clip-benchmark/wds_country211) |
| License | See original dataset |

## Row Counts

| Split / Table | Rows |
|---------------|------|
| `data/images.lance` | 52,750 |
| `official_train` | 31,650 |
| `official_train_without_mmeb_v2_eval` | 31,650 |
| `official_test` | 21,100 |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `data/images.lance.path` | string | Image lookup key |
| `data/images.lance.image` | binary | Image bytes |
| `data/{split}.lance.image_path` | string | Image lookup key |
| `data/{split}.lance.class_name` | string | Country name |
| `data/{split}.lance.class_index` | int | Class ID |
| `data/{split}.lance.source_tar` | string | Original tar file |
| `data/{split}.lance.sample_key` | string | Original sample key |

## Runtime Sample

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
