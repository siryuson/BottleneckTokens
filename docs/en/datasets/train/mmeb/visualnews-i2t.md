# VisualNews-i2t

> News image-to-text retrieval dataset, used in MMEB training to improve alignment between news image semantics and text.

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
| Original Name | VisualNews |
| Paper | Visual News: Benchmark and Challenges in News Image Captioning ([arXiv](https://arxiv.org/abs/2010.03743)) |
| Primary Authors | Fuxiao Liu, Yinghan Wang, Tianlu Wang, Vicente Ordonez |
| HuggingFace | [FuxiaoLiu/visualnews](https://huggingface.co/datasets/FuxiaoLiu/visualnews) |
| License | See original dataset |

### Schema
Original data consists of news images and news text, which can be used to construct image-to-text retrieval training samples.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `VisualNews_i2t` |
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

**Sample Table `data/VisualNews_i2t/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/VisualNews_i2t.lance`**:

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
  "data/VisualNews_i2t/train.lance": {
    "qry": "<|image_1|>\nFind a caption for the news in the given photo.\n",
    "qry_image_path": "images/VisualNews_i2t/Train/washington_post_images_0528_634.jpg",
    "pos_text": "Turkish President Tayyip Erdogan looks on after arriving at Esenboga Airport in Ankara Turkey June 8 2015 Turkey faced the prospect of weeks of political turmoil after the ruling AK Party lost its parliamentary majority REUTERSUmit Bektas.",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/VisualNews_i2t.lance": {
    "query_image": {
      "path": "images/VisualNews_i2t/Train/washington_post_images_0528_634.jpg",
      "data": {
        "bytes": 12395,
        "sha1": "ebb586857d69",
        "format": "JPEG",
        "size": [
          376,
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
  "data/VisualNews_i2t/train.lance": {
    "qry": "<|image_1|>\nFind a caption for the news in the given photo.\n",
    "qry_image_path": "images/VisualNews_i2t/Train/bbc_images_0061_223.jpg",
    "pos_text": "Monitoring equipment has been installed on some sections of the Forth Road Bridge as part of the repair work.",
    "pos_image_path": "",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/VisualNews_i2t.lance": {
    "query_image": {
      "path": "images/VisualNews_i2t/Train/bbc_images_0061_223.jpg",
      "data": {
        "bytes": 21347,
        "sha1": "8abbaa1e47cb",
        "format": "JPEG",
        "size": [
          455,
          256
        ]
      }
    }
  }
}
```
