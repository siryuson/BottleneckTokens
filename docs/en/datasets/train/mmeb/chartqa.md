# ChartQA

> Chart question answering dataset, used in MMEB training to enhance chart visual parsing and numerical logic question answering capability.

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
| Original Name | ChartQA |
| Paper | ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning ([arXiv](https://arxiv.org/abs/2203.10244)) |
| Primary Authors | Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, Enamul Hoque |
| HuggingFace | [ahmed-masry/ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) |
| License | See original dataset |

### Schema
Original data consists of chart images, question text, and answers, focusing on chart reading and visual logic reasoning.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `ChartQA` |
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

**Sample Table `data/ChartQA/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/ChartQA.lance`**:

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
  "data/ChartQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Is the value of Favorable 38 in 2015?\n",
    "qry_image_path": "images/ChartQA/Train/ChartQA_image_0.jpg",
    "pos_text": "Yes",
    "pos_image_path": "",
    "neg_text": "No",
    "neg_image_path": ""
  },
  "data/images/ChartQA.lance": {
    "query_image": {
      "path": "images/ChartQA/Train/ChartQA_image_0.jpg",
      "data": {
        "bytes": 20857,
        "sha1": "3865210fd03c",
        "format": "JPEG",
        "size": [
          422,
          359
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/ChartQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: How much did Spain spend on clothing between 2007 and 2019?\n",
    "qry_image_path": "images/ChartQA/Train/ChartQA_image_14149.jpg",
    "pos_text": "22605",
    "pos_image_path": "",
    "neg_text": "4304",
    "neg_image_path": ""
  },
  "data/images/ChartQA.lance": {
    "query_image": {
      "path": "images/ChartQA/Train/ChartQA_image_14149.jpg",
      "data": {
        "bytes": 47615,
        "sha1": "e0850c13c6ea",
        "format": "JPEG",
        "size": [
          800,
          557
        ]
      }
    }
  }
}
```
