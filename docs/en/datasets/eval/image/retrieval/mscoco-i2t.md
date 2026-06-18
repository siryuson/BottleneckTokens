# MSCOCO I2T

> Image-to-text retrieval evaluation subset (image-to-text).

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_retrieval` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset
### Source Info
| Property | Value |
|------|-----|
| Original Name | MS COCO Captions |
| Paper | Microsoft COCO: Common Objects in Context ([arXiv](https://arxiv.org/abs/1405.0312)) |
| Primary Authors | Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick |
| HuggingFace | [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO) |
| License | See original dataset |

### Schema
Image and multiple caption texts; retrieval task is image-to-text.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `MSCOCO_i2t` |
| Split | `test` |
| Eval Parser | `image_i2t` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/MSCOCO_i2t/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_2183`, `negative_candidate_id=c_2081`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nFind an image caption describing the given everyday image.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 26975,
          "sha1": "1bbf892ae698",
          "format": "JPEG",
          "size": [
            455,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_2183",
      "text": "A man with a red helmet on a small moped on a dirt road.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2081",
      "text": "A man standing in front of an old truck.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_2183"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```

#### Example 2
`query_id=q_1`, `positive_candidate_id=c_3646`, `negative_candidate_id=c_2705`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nFind an image caption describing the given everyday image.\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 13565,
          "sha1": "bda536a3ac8c",
          "format": "JPEG",
          "size": [
            383,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_3646",
      "text": "A young girl inhales with the intent of blowing out a candle.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2705",
      "text": "A red fire hydrant is in the snow near a mountain.",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_3646"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
