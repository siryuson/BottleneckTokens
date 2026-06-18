# MSCOCO-t2i

> Text-to-image retrieval dataset, used in MMEB training to learn alignment between natural language and image semantic spaces.

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
Original samples contain images with paired descriptive text, which can be used to construct text-to-image retrieval tasks.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `MSCOCO_t2i` |
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

**Sample Table `data/MSCOCO_t2i/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/MSCOCO_t2i.lance`**:

| Column | Type | Description |
|------|------|------|
| path | string | Raw image path |
| image | binary | Image bytes |

### Training-side Mapping
- query: final runtime shape is `"{text}\n"` without a visual token
- positive: final runtime shape is `"<|image_pad|>\n{text}\n"` with the resolved positive image
- negative: final runtime shape is empty text
- runtime transform normalizes legacy `<|image_1|>` tokens to `<|image_pad|>` by default

### Sample

The examples below are sampled directly from the real training Lance tables.

- Sample rows keep the original Lance values.
- `image` in the image table is summarized as `bytes / sha1 / format / size` instead of raw binary.

#### Example 1

```json
{
  "data/MSCOCO_t2i/train.lance": {
    "qry": "Find me an everyday image that matches the given caption: A teddy bear shop is equipped with a door guard teddy and a neighbor teddy above.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/MSCOCO_t2i/Train/COCO_train2014_000000161071.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_t2i.lance": {
    "positive_image": {
      "path": "images/MSCOCO_t2i/Train/COCO_train2014_000000161071.jpg",
      "data": {
        "bytes": 21012,
        "sha1": "6facc7617f93",
        "format": "JPEG",
        "size": [
          256,
          377
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/MSCOCO_t2i/train.lance": {
    "qry": "Find me an everyday image that matches the given caption: A man that is standing on a plate.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/MSCOCO_t2i/Train/COCO_train2014_000000289425.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO_t2i.lance": {
    "positive_image": {
      "path": "images/MSCOCO_t2i/Train/COCO_train2014_000000289425.jpg",
      "data": {
        "bytes": 12498,
        "sha1": "b74322679c0b",
        "format": "JPEG",
        "size": [
          333,
          256
        ]
      }
    }
  }
}
```
