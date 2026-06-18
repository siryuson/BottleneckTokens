# DocVQA

> Document image question answering dataset, used in MMEB training to enhance document page understanding and question answering alignment capability.

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
| Original Name | DocVQA |
| Paper | DocVQA: A Dataset for VQA on Document Images ([arXiv](https://arxiv.org/abs/2007.00398)) |
| Primary Authors | Minesh Mathew, Dimosthenis Karatzas, C.V. Jawahar |
| HuggingFace | [cmarkea/docvqa](https://huggingface.co/datasets/cmarkea/docvqa) |
| License | See original dataset |

### Schema
Original samples consist of document images, questions, and answers, emphasizing joint understanding of text layout and visual elements.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `DocVQA` |
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

**Sample Table `data/DocVQA/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/DocVQA.lance`**:

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
  "data/DocVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: what is the date mentioned in this letter?\n",
    "qry_image_path": "images/DocVQA/Train/DocVQA_image_0.jpg",
    "pos_text": "1/8/93",
    "pos_image_path": "",
    "neg_text": "398,627.92",
    "neg_image_path": ""
  },
  "data/images/DocVQA.lance": {
    "query_image": {
      "path": "images/DocVQA/Train/DocVQA_image_0.jpg",
      "data": {
        "bytes": 186082,
        "sha1": "dc1b29161139",
        "format": "JPEG",
        "size": [
          1695,
          2025
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/DocVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Who is the Professor of Food and Nutrition in South Carolina State College?\n",
    "qry_image_path": "images/DocVQA/Train/DocVQA_image_19731.jpg",
    "pos_text": "abernathy, miriam m., ph.d.",
    "pos_image_path": "",
    "neg_text": "schultze",
    "neg_image_path": ""
  },
  "data/images/DocVQA.lance": {
    "query_image": {
      "path": "images/DocVQA/Train/DocVQA_image_19731.jpg",
      "data": {
        "bytes": 438316,
        "sha1": "e3782efdb91e",
        "format": "JPEG",
        "size": [
          1784,
          2273
        ]
      }
    }
  }
}
```
