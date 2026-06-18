# TextVQA Training Dataset

> Status: implemented. The current runtime is `textvqa_train`. Conversion preserves the original non-image TextVQA fields and stores images in the shared `data/images.lance` table.

> Scene text question answering dataset used for training and OOD evaluation.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_question_answer` |
| Usage | Training / OOD evaluation |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | TextVQA |
| Paper | Towards VQA Models That Can Read ([arXiv:1904.08920](https://arxiv.org/abs/1904.08920)) |
| HuggingFace | [facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) or [lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) |
| License | See original dataset |

### Scale

- Default training split: 34,496 samples
- raw train: 34,602 samples
- validation: 5,000 samples
- test: 5,734 samples
- MMEB-V2 eval exclusion: 106 samples

### Schema

| Field | Type | Description |
|------|------|------|
| image | PIL.Image / bytes | Image |
| question | string | Question text |
| answer | string | Answer text |

### Sample

See the runtime sample below.

---

## BToks Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `textvqa_train` |
| Implementation | `src/vlm2emb/data/datasets/textvqa_train.py` |
| Registry Name | `textvqa_train` |
| Conversion Script | `scripts/convert/train/convert_textvqa_to_lance.py` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/images.lance.path` | string | Image lookup key |
| `data/images.lance.image` | binary | Image bytes |
| `data/{split}.lance.image_id` | string | Image ID |
| `data/{split}.lance.question` | string | Question |
| `data/{split}.lance.answers` | list | Original answer list |
| `data/{split}.lance.ocr_tokens` | list | Original OCR token list |

### Training-side Mapping

- query: `<|image_pad|>\nRepresent the given image with the following question: {question}\n`
- positive: first available answer without adding a trailing newline
- negative: empty text (placeholder)

### Sample

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

The current runtime follows the archive parser and does not append `ocr_tokens` to the query. Those fields stay in the split table so an OCR-aware transform can be discussed separately later.
