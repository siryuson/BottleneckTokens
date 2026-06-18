# Place365 Training Dataset

> Scene classification dataset used for training and OOD evaluation.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_classification` |
| Usage | Training / OOD evaluation |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | Places365 |
| Paper | Places: A 10 Million Image Database for Scene Recognition (TPAMI 2017) |
| HuggingFace | [dpdl-benchmark/Places365-Validation](https://huggingface.co/datasets/dpdl-benchmark/Places365-Validation) or [Andron00e/Places365-custom](https://huggingface.co/datasets/Andron00e/Places365-custom) |
| License | See original dataset |

### Scale

- `official_train`: `1,803,460` images
- `official_val`: `36,500` images
- Number of classes: `365`

### Schema

| Field | Type | Description |
|------|------|------|
| image | PIL.Image / bytes | Scene image |
| label | string | Scene category label |

### Sample

**query.text**: `<|image_pad|>\nIdentify the scene shown in the image\n`
**query.images**: 1 image
**positive.text**: `airfield`
**negative.text**: empty

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `place365_train` |
| Implementation | `src/vlm2emb/data/datasets/place365_train.py` |
| Registry Name | `place365_train` |
| Conversion Script | `scripts/convert/train/convert_place365_to_lance.py` |
| Root Directory | `./data/converted` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/images.lance.path` | string | Image lookup key |
| `data/images.lance.image` | binary | Image bytes |
| `data/{split}.lance.path` | string | Image lookup key |
| `data/{split}.lance.label` | string | Scene category text |
| `data/{split}.lance.class_id` | int | Class ID |

### Training-side Mapping

- query: `<|image_pad|>\nIdentify the scene shown in the image\n`
- positive: `{label}`
- negative: empty text (placeholder)

### Split Strategy

- `official_train`: default training view
- `official_val`: held out validation view

### Local Audit Result

- Local source root is `./data/converted`
- The source contains both `train/` and `val/`
- `train.txt` and `val.txt` are disjoint, so the current implementation preserves both official splits explicitly instead of following the archive-style directory walk

### Sample

The current runtime default shape is:

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
