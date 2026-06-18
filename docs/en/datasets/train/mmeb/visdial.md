# VisDial

> Multi-round visual dialogue dataset, used in MMEB training to enhance conversational image-text retrieval and contextual understanding capabilities.

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
| Original Name | Visual Dialog |
| Paper | Visual Dialog ([arXiv](https://arxiv.org/abs/1611.08669)) |
| Primary Authors | Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M. F. Moura, Devi Parikh, Dhruv Batra |
| HuggingFace | [HuggingFaceM4/VisDial](https://huggingface.co/datasets/HuggingFaceM4/VisDial) |
| License | See original dataset |

### Schema
Original samples contain images, dialogue history, current questions, and candidate answers, forming a multi-round visual dialogue retrieval scenario.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `VisDial` |
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

**Sample Table `data/VisDial/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/VisDial.lance`**:

| Column | Type | Description |
|------|------|------|
| path | string | Raw image path |
| image | binary | Image bytes |

### Training-side Mapping
- query: final runtime shape is `"{text}\n"` without a visual token
- positive: final runtime shape is `"<|image_pad|>\n{text}\n"` with the resolved positive image
- negative: when the negative side has image-only content, runtime fills it as `"<|image_pad|>\n"`
- runtime transform normalizes legacy `<|image_1|>` tokens to `<|image_pad|>` by default

### Sample

The examples below are sampled directly from the real training Lance tables.

- Sample rows keep the original Lance values.
- `image` in the image table is summarized as `bytes / sha1 / format / size` instead of raw binary.

#### Example 1

```json
{
  "data/VisDial/train.lance": {
    "qry": "Represent the given dialogue about an image, which is used for image retrieval: Q:is this a child or adult\nA:adult\nQ:male or female\nA:male\nQ:are they inside or outside\nA:inside\nQ:are they laying on the floor\nA:yes, but there is a blanket in between them and the floor\nQ:is the floor carpeted or wooden\nA:it is tile\nQ:what color is the blanket\nA:red and white\nQ:what color is the tile\nA:orange red\nQ:what breed is the dog\nA:boxer\nQ:does the dog look healthy and happy\nA:yes\nQ:what color is the dog\nA:tan\n\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image\n",
    "pos_image_path": "images/VisDial/Train/VisDial_image_0.jpg",
    "neg_text": "",
    "neg_image_path": "images/VisDial/Train/VisDial_image_61616.jpg"
  },
  "data/images/VisDial.lance": {
    "positive_image": {
      "path": "images/VisDial/Train/VisDial_image_0.jpg",
      "data": {
        "bytes": 57679,
        "sha1": "fe7b7d48a7ac",
        "format": "JPEG",
        "size": [
          612,
          612
        ]
      }
    },
    "negative_image": {
      "path": "images/VisDial/Train/VisDial_image_61616.jpg",
      "data": {
        "bytes": 53400,
        "sha1": "a5968c4411f3",
        "format": "JPEG",
        "size": [
          640,
          425
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/VisDial/train.lance": {
    "qry": "Represent the given dialogue about an image, which is used for image retrieval: Q:does it look like inside a store\nA:i think it is the display window for a store, facing the street\nQ:are the vases all on 1 shelf\nA:yes\nQ:does it look like an antique store\nA:no\nQ:are there any people looking in\nA:no\nQ:does it look like a nice day\nA:no\nQ:are the vases big\nA:some are\nQ:are they all different colors\nA:yes\nQ:is anything else on shelf with them\nA:no\nQ:can you see any price tags\nA:no\nQ:can you see the name of store\nA:no\n\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image\n",
    "pos_image_path": "images/VisDial/Train/VisDial_image_61643.jpg",
    "neg_text": "",
    "neg_image_path": "images/VisDial/Train/VisDial_image_50556.jpg"
  },
  "data/images/VisDial.lance": {
    "positive_image": {
      "path": "images/VisDial/Train/VisDial_image_61643.jpg",
      "data": {
        "bytes": 66555,
        "sha1": "27b4c2a93271",
        "format": "JPEG",
        "size": [
          433,
          640
        ]
      }
    },
    "negative_image": {
      "path": "images/VisDial/Train/VisDial_image_50556.jpg",
      "data": {
        "bytes": 28879,
        "sha1": "0fa051d7d99d",
        "format": "JPEG",
        "size": [
          427,
          640
        ]
      }
    }
  }
}
```
