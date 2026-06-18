# MSCOCO-i2t

> Image-to-text retrieval dataset, used in MMEB training to enhance mapping capabilities from image semantics to text semantics.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_retrieval` |
| Usage | Training |
| Provenance Layers | 3 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | MS COCO Captions |
| Paper | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| Primary Authors | Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| License | See original dataset |

### Schema
Original samples consist of images with multiple descriptions, which can be used for image-to-text retrieval tasks.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `MSCOCO_i2t` |
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

**Sample Table `data/MSCOCO_i2t/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/MSCOCO_i2t.lance`**:

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
  "data/MSCOCO_i2t/train.lance": {
    "qry": "<|image_1|>\nFind an image caption describing the given everyday image.\n",
    "qry_image_path": "images/MSCOCO_i2t/Train/COCO_train2014_000000376224.jpg",
    "pos_text": "A skateboarder in mid air following a jump form cincrete.",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_i2t.lance": {
    "query_image": {
      "path": "images/MSCOCO_i2t/Train/COCO_train2014_000000376224.jpg",
      "data": {
        "bytes": 13478,
        "sha1": "c105ef7a3d14",
        "format": "JPEG",
        "size": [
          256,
          383
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/MSCOCO_i2t/train.lance": {
    "qry": "<|image_1|>\nFind an image caption describing the given everyday image.\n",
    "qry_image_path": "images/MSCOCO_i2t/Train/COCO_train2014_000000445323.jpg",
    "pos_text": "A group of sailors sitting in front of a control area.",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_i2t.lance": {
    "query_image": {
      "path": "images/MSCOCO_i2t/Train/COCO_train2014_000000445323.jpg",
      "data": {
        "bytes": 21062,
        "sha1": "8efc7c885cbb",
        "format": "JPEG",
        "size": [
          420,
          256
        ]
      }
    }
  }
}
```
