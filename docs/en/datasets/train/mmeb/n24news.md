# N24News

> Multimodal news classification dataset supplementing image-text category discrimination capability in news context for MMEB training.

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
| Original Name | N24News |
| Paper | N24News: A New Dataset for Multimodal News Classification ([arXiv](https://arxiv.org/abs/2108.13327)) |
| Primary Authors | Zhen Wang, Xu Shan, Xiangxie Zhang, Jie Yang |
| HuggingFace | [linxy/Multimodal-N24News](https://huggingface.co/datasets/linxy/Multimodal-N24News) |
| License | See original dataset |

### Schema
Original samples contain news images, news text, and 24-category labels for multimodal news classification.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `N24News` |
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

**Sample Table `data/N24News/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/N24News.lance`**:

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
  "data/N24News/train.lance": {
    "qry": "<|image_1|>\nRepresent the given news image with the following caption for domain classification: Danai Gurira and André Holland in a theater at New York University, where they met in the Tisch Graduate Acting Program.\n",
    "qry_image_path": "images/N24News/Train/N24News_image_0.jpg",
    "pos_text": "Theater",
    "pos_image_path": "",
    "neg_text": "Global Business",
    "neg_image_path": ""
  },
  "data/images/N24News.lance": {
    "query_image": {
      "path": "images/N24News/Train/N24News_image_0.jpg",
      "data": {
        "bytes": 49525,
        "sha1": "8fad5759eba1",
        "format": "JPEG",
        "size": [
          1050,
          550
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/N24News/train.lance": {
    "qry": "<|image_1|>\nRepresent the given news image with the following caption for domain classification: The face of a volunteer, Miguel Grisanti, is used to cloak the real subject in &ldquo;Welcome to Chechnya.&rdquo;\n",
    "qry_image_path": "images/N24News/Train/N24News_image_24494.jpg",
    "pos_text": "Movies",
    "pos_image_path": "",
    "neg_text": "Music",
    "neg_image_path": ""
  },
  "data/images/N24News.lance": {
    "query_image": {
      "path": "images/N24News/Train/N24News_image_24494.jpg",
      "data": {
        "bytes": 48163,
        "sha1": "7370dbf65288",
        "format": "JPEG",
        "size": [
          1050,
          550
        ]
      }
    }
  }
}
```
