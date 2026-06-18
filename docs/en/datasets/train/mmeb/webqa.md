# WebQA

> Multi-hop multi-modal question answering retrieval dataset, used in MMEB training to enhance cross-source image-text evidence retrieval capabilities.

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
| Original Name | WebQA |
| Paper | WebQA: Multihop and Multimodal QA ([arXiv](https://arxiv.org/abs/2109.00590)) |
| Primary Authors | Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, Yonatan Bisk |
| HuggingFace | [suolyer/webqa](https://huggingface.co/datasets/suolyer/webqa) |
| License | See original dataset |

### Schema
Original task focuses on open-domain multi-source evidence retrieval and question answering, with samples consisting of questions and candidate image-text evidence.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `WebQA` |
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

**Sample Table `data/WebQA/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/WebQA.lance`**:

| Column | Type | Description |
|------|------|------|
| path | string | Raw image path |
| image | binary | Image bytes |

### Training-side Mapping
- query: final runtime shape is `"{text}\n"`; when the query truly has no image, the injected legacy visual token is stripped
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
  "data/WebQA/train.lance": {
    "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: Which body part is found on both a 1913 D Barber half and a 1914 Barber Quarter?\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given Wikipedia image with related text information: 1913-D Barber half obverse.\n",
    "pos_image_path": "images/WebQA/Train/30272327.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/WebQA.lance": {
    "positive_image": {
      "path": "images/WebQA/Train/30272327.jpg",
      "data": {
        "bytes": 16560,
        "sha1": "77cccbcf3dd8",
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
  "data/WebQA/train.lance": {
    "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: In the painting \"Island Women\" does the woman on the far left or the woman on the far right have longer hair?\n",
    "qry_image_path": "",
    "pos_text": "<|image_1|>\nRepresent the given Wikipedia image with related text information: Island Women by Bakusen Tsuchida, 1912, color on silk - National Museum of Modern Art, Tokyo - DSC06665  Tsuchida Bakusen, 1912. (Island Women). Pair of two folding screens, color on silk, each 166.5 x 184 cm. The National Museum of Modern Art, Tokyo (). This artwork is in the public domain because the artist died more than 70 years ago. Photography was permitted in the museum without restriction.\n",
    "pos_image_path": "images/WebQA/Train/30000598.jpg",
    "neg_text": "",
    "neg_image_path": ""
  },
  "data/images/WebQA.lance": {
    "positive_image": {
      "path": "images/WebQA/Train/30000598.jpg",
      "data": {
        "bytes": 14658,
        "sha1": "28621ddd75a3",
        "format": "JPEG",
        "size": [
          375,
          256
        ]
      }
    }
  }
}
```
