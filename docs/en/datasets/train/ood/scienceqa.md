# ScienceQA Training Dataset

> Status: implemented. The current runtime is `scienceqa_train`. Conversion preserves the original question-answer fields and stores images in the shared `data/images.lance` table.

> Science question answering dataset used for training and OOD evaluation.

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
| Original Name | ScienceQA |
| Paper | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (NeurIPS 2022) |
| HuggingFace | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| License | See original dataset |

### Scale

- Default training split: 6,218 samples
- validation: 2,097 samples
- test / MMEB-V2 eval exclusion: 2,017 samples

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
| Data Loader | `scienceqa_train` |
| Implementation | `src/vlm2emb/data/datasets/scienceqa_train.py` |
| Registry Name | `scienceqa_train` |
| Conversion Script | `scripts/convert/train/convert_scienceqa_to_lance.py` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/images.lance.path` | string | Image lookup key |
| `data/images.lance.image` | binary | Image bytes |
| `data/{split}.lance.question` | string | Question |
| `data/{split}.lance.answer` | string | Answer |

### Training-side Mapping

- query: `<|image_pad|>\nRepresent the given image with the following question: {question}\n`
- positive: `{answer}`
- negative: empty text (placeholder)

### Sample

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
