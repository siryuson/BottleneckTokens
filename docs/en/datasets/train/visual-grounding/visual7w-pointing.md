# Visual7W Pointing Training Dataset

> Visual grounding dataset used for training pointing-style grounding behavior.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_visual_grounding` |
| Usage | Training |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | Visual7W Pointing |
| Paper | Visual7W: Grounded Question Answering in Images ([arXiv:1511.03416](https://arxiv.org/abs/1511.03416)) |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| License | See original dataset |

### Scale

- `pointing/train`: `93,813` samples
- `pointing/val`: `36,990` samples
- `pointing/test`: `57,265` samples

### Schema

| Field | Type | Description |
|------|------|------|
| image | PIL.Image / bytes | Full image |
| question | string | Pointing question |
| answer | box id | Correct target box ID |
| multiple_choices | list[box id] | Other candidate box IDs |

### Sample

**query.text**: `<|image_pad|>\nSelect the portion of the image that answers the question "Which door is behind a person sitting on a bench?"\n`
**query.images**: 1 full image
**positive.text**: `<|image_pad|>\nRepresent the given cropped image of the object.\n`
**positive.images**: 1 cropped answer region
**negative.text**: ``

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `visual7w_pointing_train` |
| Implementation | `src/vlm2emb/data/datasets/visual7w_pointing.py` |
| Registry Name | `visual7w_pointing_train` |
| Conversion Script | `scripts/convert/train/convert_visual7w_pointing_to_lance.py` |
| Root Directory | `./data/converted` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/images.lance.path` | string | Full-image or crop lookup key |
| `data/images.lance.image` | binary | Full-image or crop bytes |
| `data/{split}.lance.sample_id` | string | Sample ID |
| `data/{split}.lance.question` | string | Pointing question |
| `data/{split}.lance.query_image_path` | string | Full-image lookup key |
| `data/{split}.lance.positive_image_path` | string | Correct crop lookup key |
| `data/{split}.lance.answer_box_id` | int | Correct box ID |
| `data/{split}.lance.answer_name` | string | Correct box label |
| `data/{split}.lance.crop_*` | int | Raw box position and size |

### Training-side Mapping

- query: `<|image_pad|>\nSelect the portion of the image that answers the question "{question}"\n`
- positive: `<|image_pad|>\nRepresent the given cropped image of the object.\n`
- negative: empty text

### Split Strategy

- `official_train_without_mmeb_v2_eval`: default training view; currently identical to `official_train`
- `official_train`: raw training split
- `official_val`: kept for analysis / tuning
- `official_test`: kept for completeness and auditing, not used for default training

### Relation to Current Eval

- The current `MMEB-V2 / Visual7W-Pointing` benchmark is highly likely sampled from raw `pointing/test`.
- Therefore the repository default does **not** add `official_test` to training configs.

### Sample

The current runtime default shape is:

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
