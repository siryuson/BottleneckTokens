# ImageNet-1K

> Classic large-scale image classification dataset providing general visual semantic discrimination capability for MMEB training.

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
| Original Name | ImageNet (ILSVRC 2012) |
| Paper | ImageNet Large Scale Visual Recognition Challenge ([arXiv](https://arxiv.org/abs/1409.0575)) |
| Primary Authors | Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, Li Fei-Fei |
| HuggingFace | [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| License | See original dataset |

### Schema
Typical samples consist of images and class labels for large-scale image classification tasks.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `ImageNet_1K` |
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

**Sample Table `data/ImageNet_1K/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/ImageNet_1K.lance`**:

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
  "data/ImageNet_1K/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for classification\n",
    "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
    "pos_text": "plane, carpenter's plane, woodworking plane",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/ImageNet_1K.lance": {
    "query_image": {
      "path": "images/ImageNet_1K/Train/image_0.jpg",
      "data": {
        "bytes": 32329,
        "sha1": "5e56f40c6b28",
        "format": "JPEG",
        "size": [
          817,
          363
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/ImageNet_1K/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image for classification\n",
    "qry_image_path": "images/ImageNet_1K/Train/image_50000.jpg",
    "pos_text": "mountain tent",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/ImageNet_1K.lance": {
    "query_image": {
      "path": "images/ImageNet_1K/Train/image_50000.jpg",
      "data": {
        "bytes": 39185,
        "sha1": "746c295a5492",
        "format": "JPEG",
        "size": [
          500,
          375
        ]
      }
    }
  }
}
```
