# Visual7W

> Visual question answering dataset based on images and multiple QA formats, used in MMEB training to enhance fine-grained visual question answering capabilities.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_question_answer` |
| Usage | Training |
| Provenance Layers | 3 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | Visual7W |
| Paper | Visual7W: Grounded Question Answering in Images ([arXiv](https://arxiv.org/abs/1511.03416)) |
| Primary Authors | Yuke Zhu, Oliver Groth, Michael Bernstein, Li Fei-Fei |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| License | See original dataset |

### Schema
Original samples typically contain images, questions, candidate answers, and annotated answers, which can be used for various visual question answering subtasks.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `Visual7W` |
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

**Sample Table `data/Visual7W/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/Visual7W.lance`**:

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
  "data/Visual7W/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is written on the white square on the bus?\n",
    "qry_image_path": "images/Visual7W/Train/v7w_2359302.jpg",
    "pos_text": "Fox's Ginger Biscuits.",
    "pos_image_path": "",
    "neg_text": "Mac's Macaroni Hut.",
    "neg_image_path": ""
  },
  "data/images/Visual7W.lance": {
    "query_image": {
      "path": "images/Visual7W/Train/v7w_2359302.jpg",
      "data": {
        "bytes": 40675,
        "sha1": "e98ce8127a04",
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

#### Example 2

```json
{
  "data/Visual7W/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Where is this picture taken?\n",
    "qry_image_path": "images/Visual7W/Train/v7w_2411048.jpg",
    "pos_text": "On a railway.",
    "pos_image_path": "",
    "neg_text": "On a dirt road.",
    "neg_image_path": ""
  },
  "data/images/Visual7W.lance": {
    "query_image": {
      "path": "images/Visual7W/Train/v7w_2411048.jpg",
      "data": {
        "bytes": 44120,
        "sha1": "0aa98dcbc8df",
        "format": "JPEG",
        "size": [
          500,
          331
        ]
      }
    }
  }
}
```
