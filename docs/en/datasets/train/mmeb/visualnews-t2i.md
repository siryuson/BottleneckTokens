# VisualNews-t2i

> News text-to-image retrieval dataset, used in MMEB training to enhance cross-modal retrieval capabilities in news contexts.

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
Original samples consist of news images, headlines/descriptions, and article metadata, which can be used to construct text-to-image retrieval pairs.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `VisualNews_t2i` |
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

**Sample Table `data/VisualNews_t2i/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/VisualNews_t2i.lance`**:

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
  "data/VisualNews_t2i/train.lance": {
    "qry": "Retrieve an image of this news caption. Turkish President Tayyip Erdogan looks on after arriving at Esenboga Airport in Ankara Turkey June 8 2015 Turkey faced the prospect of weeks of political turmoil after the ruling AK Party lost its parliamentary majority REUTERSUmit Bektas.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/VisualNews_t2i/Train/washington_post_images_0528_634.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/VisualNews_t2i.lance": {
    "positive_image": {
      "path": "images/VisualNews_t2i/Train/washington_post_images_0528_634.jpg",
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
  "data/VisualNews_t2i/train.lance": {
    "qry": "Retrieve an image of this news caption. Itte Detenamo ca nt make this lift in the clean and jerk His total of 396kg gave him silver while Australia s Damon Kelly took bronze with 388kg.\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image_path": "images/VisualNews_t2i/Train/guardian_images_0266_314.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/VisualNews_t2i.lance": {
    "positive_image": {
      "path": "images/VisualNews_t2i/Train/guardian_images_0266_314.jpg",
      "data": {
        "bytes": 14970,
        "sha1": "020c0dbe40cd",
        "format": "JPEG",
        "size": [
          414,
          256
        ]
      }
    }
  }
}
```
