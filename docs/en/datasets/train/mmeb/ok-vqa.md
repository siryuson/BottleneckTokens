# OK-VQA

> Visual question answering dataset requiring external knowledge, used in MMEB training to enhance knowledge reasoning capability for image question answering.

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
| Original Name | OK-VQA |
| Paper | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge ([arXiv](https://arxiv.org/abs/1906.00067)) |
| Primary Authors | Kenneth Marino, Mohammad Rastegari, Ali Farhadi, Roozbeh Mottaghi |
| HuggingFace | [HuggingFaceM4/OK-VQA](https://huggingface.co/datasets/HuggingFaceM4/OK-VQA) |
| License | See original dataset |

### Schema
Original samples typically contain images, question text, and answer sets, emphasizing the combination of visual and external knowledge for question answering.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `OK-VQA` |
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

**Sample Table `data/OK-VQA/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/OK-VQA.lance`**:

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
  "data/OK-VQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is the hairstyle of the blond called?\n",
    "qry_image_path": "images/OK-VQA/Train/OK-VQA_image_0.jpg",
    "pos_text": "pony tail",
    "pos_image_path": "",
    "neg_text": "husky",
    "neg_image_path": ""
  },
  "data/images/OK-VQA.lance": {
    "query_image": {
      "path": "images/OK-VQA/Train/OK-VQA_image_0.jpg",
      "data": {
        "bytes": 55238,
        "sha1": "4ebb8e070581",
        "format": "JPEG",
        "size": [
          640,
          479
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/OK-VQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What type of numerals are the numbers on this clock?\n",
    "qry_image_path": "images/OK-VQA/Train/OK-VQA_image_4504.jpg",
    "pos_text": "roman",
    "pos_image_path": "",
    "neg_text": "small town",
    "neg_image_path": ""
  },
  "data/images/OK-VQA.lance": {
    "query_image": {
      "path": "images/OK-VQA/Train/OK-VQA_image_4504.jpg",
      "data": {
        "bytes": 74882,
        "sha1": "f19e67f168a8",
        "format": "JPEG",
        "size": [
          610,
          640
        ]
      }
    }
  }
}
```
