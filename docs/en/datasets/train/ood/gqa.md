# GQA Training Dataset

> Status: implemented. The current runtime is `gqa_train`. Conversion preserves the original question-answer fields and stores images in the shared `data/images.lance` table.

> Visual reasoning question answering dataset used to train compositional visual reasoning.

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
| Original Name | GQA |
| Paper | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering ([arXiv:1902.09506](https://arxiv.org/abs/1902.09506)) |
| HuggingFace | [lmms-lab/GQA](https://huggingface.co/datasets/lmms-lab/GQA) |
| License | See original dataset |

### Scale

- Images: 74,256
- `train_all` questions: 14,305,356
- `train_balanced` questions: 943,000
- Default training split: `official_train_balanced_without_mmeb_v2_eval`

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
| Data Loader | `gqa_train` |
| Implementation | `src/vlm2emb/data/datasets/gqa_train.py` |
| Registry Name | `gqa_train` |
| Conversion Script | `scripts/convert/train/convert_gqa_to_lance.py` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/images.lance.path` | string | Image lookup key |
| `data/images.lance.image` | binary | Image bytes |
| `data/{split}.lance.imageId` | string | Image ID |
| `data/{split}.lance.question` | string | Question |
| `data/{split}.lance.fullAnswer` | string | Full answer |

### Split Views

| Split | Rows | Notes |
|------|------:|------|
| `official_train_all` | 14,305,356 | Full `train_all_instructions` source view |
| `official_train_all_without_mmeb_v2_eval` | 14,305,356 | Full source view after MMEB-V2 eval exclusion |
| `official_train_balanced` | 943,000 | Official balanced train view |
| `official_train_balanced_without_mmeb_v2_eval` | 943,000 | Default training split |

### Training-side Mapping

- query: `<|image_pad|>\nRepresent the given image with the following question: {question}\n`
- positive: `{fullAnswer}`
- negative: empty text (placeholder)

### MMEB-V2 overlap audit

- The current conversion uses an exact QA sample-level key: `imageId + normalized(question)`.
- MMEB-V2 `GQA` eval has `1,000` query rows, `773` unique image IDs, `954` unique normalized questions, and `999` unique `image_id + question` keys.
- GQA `train_all` has `14,305,356` rows and `74,256` unique train image IDs. `train_balanced` has `943,000` rows over `72,140` images.
- Train image IDs and eval image IDs have `0` intersection; `imageId + normalized(question)` also has `0` intersection.
- Text-only question overlap is high: `191,495` train rows have a question that appears in eval; `question + fullAnswer` also matches eval positive text for `68,741` train rows. These matches occur on different image IDs and reflect GQA's templated language reuse, so they are not used as sample-level leakage keys.
- Therefore the eval-excluded all and balanced training splits keep the same row counts as their source views, and the exclusion tables have `0` rows. This is the audited result, not a missed conversion exclusion.

### Sample

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
