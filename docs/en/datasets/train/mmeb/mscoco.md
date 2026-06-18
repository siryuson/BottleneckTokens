# MSCOCO

> Visual grounding subset based on COCO, used in MMEB training to learn alignment between image-text phrases and image regions.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_visual_grounding` |
| Usage | Training |
| Provenance Layers | 3 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | MS COCO |
| Paper | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| Primary Authors | Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| License | See original dataset |

### Schema
Original samples consist of images, objects, and description information, which can be used to construct phrase-to-visual-region grounding learning samples.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `MSCOCO` |
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

**Sample Table `data/MSCOCO/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/MSCOCO.lance`**:

| Column | Type | Description |
|------|------|------|
| path | string | Raw image path |
| image | binary | Image bytes |

### Training-side Mapping
- query: final runtime shape is `"<|image_pad|>\n{text}\n"` with the resolved query image
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
  "data/MSCOCO/train.lance": {
    "qry": "<|image_1|>\nSelect the portion of the image that isolates the object labeled as \"hot dog\"\n",
    "qry_image_path": "images/MSCOCO/Train/000000558840.jpg",
    "pos_text": "<|image_1|>\nRepresent the given cropped image of the object",
    "pos_image_path": "images/MSCOCO/Train/558840_object_hot dog.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO.lance": {
    "query_image": {
      "path": "images/MSCOCO/Train/000000558840.jpg",
      "data": {
        "bytes": 173812,
        "sha1": "b4c1b5ccc00c",
        "format": "JPEG",
        "size": [
          640,
          427
        ]
      }
    },
    "positive_image": {
      "path": "images/MSCOCO/Train/558840_object_hot dog.jpg",
      "data": {
        "bytes": 4024,
        "sha1": "78fbc8b3eb85",
        "format": "JPEG",
        "size": [
          77,
          70
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/MSCOCO/train.lance": {
    "qry": "<|image_1|>\nSelect the portion of the image that isolates the object labeled as \"person\"\n",
    "qry_image_path": "images/MSCOCO/Train/000000170389.jpg",
    "pos_text": "<|image_1|>\nRepresent the given cropped image of the object",
    "pos_image_path": "images/MSCOCO/Train/170389_object_person.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/MSCOCO.lance": {
    "query_image": {
      "path": "images/MSCOCO/Train/000000170389.jpg",
      "data": {
        "bytes": 133065,
        "sha1": "cfde9d220814",
        "format": "JPEG",
        "size": [
          375,
          500
        ]
      }
    },
    "positive_image": {
      "path": "images/MSCOCO/Train/170389_object_person.jpg",
      "data": {
        "bytes": 819,
        "sha1": "fe2b085b6d52",
        "format": "JPEG",
        "size": [
          4,
          18
        ]
      }
    }
  }
}
```
