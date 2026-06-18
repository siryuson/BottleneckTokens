# NIGHTS

> Night scene retrieval dataset, used in MMEB training to enhance robustness of low-illumination and cross-scene visual retrieval.

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
| Original Name | NIGHTS (M-BEIR task) |
| Paper | UniIR: Training and Benchmarking Universal Multimodal Information Retrievers ([arXiv](https://arxiv.org/abs/2311.17136)) |
| Primary Authors | Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, Wenhu Chen |
| HuggingFace | [mteb/mbeir_nights_task4](https://huggingface.co/datasets/mteb/mbeir_nights_task4) |
| License | See original dataset |

### Schema
Original task consists of night image retrieval pairs, focusing on visual matching and retrieval under low-light conditions.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `NIGHTS` |
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

**Sample Table `data/NIGHTS/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/NIGHTS.lance`**:

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
  "data/NIGHTS/train.lance": {
    "qry": "<|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n",
    "qry_image_path": "images/NIGHTS/Train/ref_000_002.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/NIGHTS/Train/distort_000_002_0.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/NIGHTS.lance": {
    "query_image": {
      "path": "images/NIGHTS/Train/ref_000_002.jpg",
      "data": {
        "bytes": 22602,
        "sha1": "ae9a0e58ce63",
        "format": "JPEG",
        "size": [
          256,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/NIGHTS/Train/distort_000_002_0.jpg",
      "data": {
        "bytes": 20542,
        "sha1": "7a69e8983db1",
        "format": "JPEG",
        "size": [
          256,
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
  "data/NIGHTS/train.lance": {
    "qry": "<|image_1|>\nFind a day-to-day image that looks similar to the provided image.\n",
    "qry_image_path": "images/NIGHTS/Train/ref_050_030.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/NIGHTS/Train/distort_050_030_1.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/NIGHTS.lance": {
    "query_image": {
      "path": "images/NIGHTS/Train/ref_050_030.jpg",
      "data": {
        "bytes": 21330,
        "sha1": "0a8812464f49",
        "format": "JPEG",
        "size": [
          256,
          256
        ]
      }
    },
    "positive_image": {
      "path": "images/NIGHTS/Train/distort_050_030_1.jpg",
      "data": {
        "bytes": 19303,
        "sha1": "219b4d899dbf",
        "format": "JPEG",
        "size": [
          256,
          256
        ]
      }
    }
  }
}
```
