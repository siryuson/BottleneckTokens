# A-OKVQA

> Question answering dataset combining visual evidence and common sense reasoning, used in MMEB training to enhance complex visual question answering capability.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_question_answer` |
| Usage | Training |
| Provenance Layers | 3 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | A-OKVQA |
| Paper | A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge ([arXiv](https://arxiv.org/abs/2206.01718)) |
| Primary Authors | Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, Roozbeh Mottaghi |
| HuggingFace | [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) |
| License | See original dataset |

### Schema
Original data includes images, questions, answers, and explanatory information for open-ended visual question answering.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `A-OKVQA` |
| Split | `train` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `pos_text`, `pos_img_path`, `neg_text`, `neg_img_path`.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `mmeb_train` |
| Implementation | `src/vlm2emb/data/datasets/mmeb_train.py` |
| Registry Name | `mmeb_train` |
| Sample Table Path | `./data/converted` |
| Image Table Path | `./data/converted` |

### Lance Layout

**Sample Table `data/A-OKVQA/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/A-OKVQA.lance`**:

| Column | Type | Description |
|------|------|------|
| path | string | Raw image path |
| image | binary | Image bytes |

### Training-side Mapping
- query: `qry` plus the image resolved from `qry_image_path` when present
- positive: `pos_text` plus the image resolved from `pos_image_path` when present
- negative: `neg_text` plus the image resolved from `neg_image_path` when present
- runtime transform normalizes legacy `<|image_1|>` tokens to `<|image_pad|>` by default

### Sample

The examples below are sampled directly from the real training Lance tables.

- Sample rows keep the original Lance values.
- `image` in the image table is summarized as `bytes / sha1 / format / size` instead of raw binary.

#### Example 1

```json
{
  "data/A-OKVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is the man by the bags awaiting?\n",
    "qry_image_path": "images/A-OKVQA/Train/A-OKVQA_image_0.jpg",
    "pos_text": "cab",
    "pos_image_path": "",
    "neg_text": "train",
    "neg_image_path": ""
  },
  "data/images/A-OKVQA.lance": {
    "query_image": {
      "path": "images/A-OKVQA/Train/A-OKVQA_image_0.jpg",
      "data": {
        "bytes": 77100,
        "sha1": "ec31c8d3b31b",
        "format": "JPEG",
        "size": [
          640,
          480
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/A-OKVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Why are they so close together?\n",
    "qry_image_path": "images/A-OKVQA/Train/A-OKVQA_image_8528.jpg",
    "pos_text": "to talk",
    "pos_image_path": "",
    "neg_text": "need directions",
    "neg_image_path": ""
  },
  "data/images/A-OKVQA.lance": {
    "query_image": {
      "path": "images/A-OKVQA/Train/A-OKVQA_image_8528.jpg",
      "data": {
        "bytes": 44068,
        "sha1": "7847910c43c0",
        "format": "JPEG",
        "size": [
          640,
          361
        ]
      }
    }
  }
}
```
