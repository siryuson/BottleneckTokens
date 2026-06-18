# InfographicsVQA

> Infographic visual question answering dataset, used in MMEB training to enhance image-text layout and data visualization question answering capability.

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
| Original Name | InfographicVQA |
| Paper | InfographicVQA ([arXiv](https://arxiv.org/abs/2104.12756)) |
| Primary Authors | Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, C.V Jawahar |
| HuggingFace | [vidore/infovqa_test_subsampled_beir](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir) |
| License | See original dataset |

### Schema
Original samples contain infographic images, questions, and answers, covering layout understanding, text reading, and simple reasoning.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## MMEB-train (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [siyrus/BToks-MMEB-train](https://huggingface.co/datasets/siyrus/BToks-MMEB-train) |
| Subset Name | `InfographicsVQA` |
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

**Sample Table `data/InfographicsVQA/train.lance`**:

| Column | Type | Description |
|------|------|------|
| qry | string | Query text. Legacy `<|image_1|>` tokens may still appear in storage. |
| qry_image_path | string | Query image path |
| pos_text | string | Positive text |
| pos_image_path | string | Positive image path, optional |
| neg_text | string | Negative text, optional |
| neg_image_path | string | Negative image path, optional |

**Image Table `data/images/InfographicsVQA.lance`**:

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
  "data/InfographicsVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: Which type of fonts offer better readability in printed works?\n",
    "qry_image_path": "images/InfographicsVQA/Train/20471.jpeg",
    "pos_text": "serif fonts",
    "pos_image_path": "",
    "neg_text": "his, he, hers",
    "neg_image_path": ""
  },
  "data/images/InfographicsVQA.lance": {
    "query_image": {
      "path": "images/InfographicsVQA/Train/20471.jpeg",
      "data": {
        "bytes": 385473,
        "sha1": "a90f538a4680",
        "format": "JPEG",
        "size": [
          596,
          5107
        ]
      }
    }
  }
}
```

#### Example 2

```json
{
  "data/InfographicsVQA/train.lance": {
    "qry": "<|image_1|>\nRepresent the given image with the following question: What is the total percentage of workers who are not an established self-employed(as a main job)?\n",
    "qry_image_path": "images/InfographicsVQA/Train/10461.jpeg",
    "pos_text": "18%",
    "pos_image_path": "",
    "neg_text": "1984, 1992",
    "neg_image_path": ""
  },
  "data/images/InfographicsVQA.lance": {
    "query_image": {
      "path": "images/InfographicsVQA/Train/10461.jpeg",
      "data": {
        "bytes": 729562,
        "sha1": "7fa4535ea4e9",
        "format": "JPEG",
        "size": [
          2263,
          4013
        ]
      }
    }
  }
}
```
