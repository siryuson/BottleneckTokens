# HatefulMemes

> Multimodal hate content recognition dataset used in MMEB training to enhance image-text joint semantic and bias recognition capability.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_classification` |
| Usage | Training |
| Provenance Layers | 3 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | Hateful Memes Challenge Dataset |
| Paper | The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes ([arXiv](https://arxiv.org/abs/2005.04790)) |
| Primary Authors | Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, Davide Testuggine |
| HuggingFace | [neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes) |
| License | See original dataset |

### Schema
Original samples consist of meme images, accompanying text, and binary labels, emphasizing cross-modal hate semantic discrimination.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `HatefulMemes` |
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

**Sample Table `data/HatefulMemes/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/HatefulMemes.lance`**:

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
  "data/HatefulMemes/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
    "qry_image_path": "images/HatefulMemes/Train/HatefulMemes_image_0.jpg",
    "pos_text": "Yes",
    "pos_image_path": "",
    "neg_text": "No",
    "neg_image_path": ""
  },
  "data/images/HatefulMemes.lance": {
    "query_image": {
      "path": "images/HatefulMemes/Train/HatefulMemes_image_0.jpg",
      "data": {
        "bytes": 30407,
        "sha1": "661e30f7cc3e",
        "format": "JPEG",
        "size": [
          550,
          366
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/HatefulMemes/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
    "qry_image_path": "images/HatefulMemes/Train/HatefulMemes_image_4250.jpg",
    "pos_text": "Yes",
    "pos_image_path": "",
    "neg_text": "No",
    "neg_image_path": ""
  },
  "data/images/HatefulMemes.lance": {
    "query_image": {
      "path": "images/HatefulMemes/Train/HatefulMemes_image_4250.jpg",
      "data": {
        "bytes": 42128,
        "sha1": "bad78bec169d",
        "format": "JPEG",
        "size": [
          550,
          366
        ]
      }
    }
  }
}
```
