# CIRR

> Compositional image retrieval dataset, used in MMEB training to learn retrieval capabilities driven by "image + text modification".

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
| Original Name | CIRR (Compose Image Retrieval on Real-life images) |
| Paper | Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models ([arXiv](https://arxiv.org/abs/2108.04024)) |
| Primary Authors | Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould |
| HuggingFace | [mteb/mbeir_cirr_task7](https://huggingface.co/datasets/mteb/mbeir_cirr_task7) |
| License | See original dataset |

### Schema
Original task input consists of a reference image and text modification description, with the goal of retrieving images from a candidate gallery that match the modification semantics.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `CIRR` |
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

**Sample Table `data/CIRR/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/CIRR.lance`**:

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
  "data/CIRR/train.lance": {
    "qry": "<|image_1|>\nGiven an image, find a similar everyday image with the described changes: Many hamster together playing in a different background.\n",
    "qry_image_path": "images/CIRR/Train/train-11991-2-img1.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/CIRR/Train/train-5270-1-img0.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/CIRR.lance": {
    "query_image": {
      "path": "images/CIRR/Train/train-11991-2-img1.jpg",
      "data": {
        "bytes": 28519,
        "sha1": "f7f58445f513",
        "format": "JPEG",
        "size": [
          519,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/CIRR/Train/train-5270-1-img0.jpg",
      "data": {
        "bytes": 19069,
        "sha1": "7d7a824bd33c",
        "format": "JPEG",
        "size": [
          341,
          256
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/CIRR/train.lance": {
    "qry": "<|image_1|>\nGiven an image, find a similar everyday image with the described changes: Take narrow wide coverage of haulted trucks with blue sky effect exclusively.\n",
    "qry_image_path": "images/CIRR/Train/train-11910-1-img1.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/CIRR/Train/train-8721-2-img1.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/CIRR.lance": {
    "query_image": {
      "path": "images/CIRR/Train/train-11910-1-img1.jpg",
      "data": {
        "bytes": 19092,
        "sha1": "abfc09e46dcc",
        "format": "JPEG",
        "size": [
          405,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/CIRR/Train/train-8721-2-img1.jpg",
      "data": {
        "bytes": 16622,
        "sha1": "d98540a3af62",
        "format": "JPEG",
        "size": [
          341,
          256
        ]
      }
    }
  }
}
```
