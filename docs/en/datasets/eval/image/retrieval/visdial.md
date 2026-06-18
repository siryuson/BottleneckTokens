# VisDial

> Text-to-image retrieval subset for multi-round visual dialogue scenarios.

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
| Original Name | Visual Dialog |
| Paper | Visual Dialog ([arXiv](https://arxiv.org/abs/1611.08669)) |
| Primary Authors | Abhishek Das, Satwik Kottur, José M. F. Moura, Stefan Lee, Dhruv Batra |
| HuggingFace | [HuggingFaceM4/VisDial](https://huggingface.co/datasets/HuggingFaceM4/VisDial) |
| License | See original dataset |

### Schema
Image, dialogue history, current-round question, and answer candidates.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `VisDial` |
| Split | `test` |
| Eval Parser | `image_t2i` |

### Schema
`qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`。

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/VisDial/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_0`, `negative_candidate_id=c_1455`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Represent the given dialogue about an image, which is used for image retrieval:\nQ:is the photo in color\nA:yes\nQ:is it a professional photo\nA:no\nQ:is it well lit\nA:no\nQ:is it daytime\nA:i don't see windows\nQ:does this look like an adults bedroom\nA:maybe\nQ:is the room clean\nA:no\nQ:can you tell what kind of computer it is\nA:no not really\nQ:is it a flat screen\nA:yes\nQ:what's the desk made out of\nA:cheap plastic or wood\nQ:is there a computer chair\nA:yes\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 31610,
            "sha1": "cce44f06331a",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1455",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 47127,
            "sha1": "11f21f4bcd78",
            "format": "JPEG",
            "size": [
              640,
              480
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_0"
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
`query_id=q_1`, `positive_candidate_id=c_1`, `negative_candidate_id=c_1835`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Represent the given dialogue about an image, which is used for image retrieval:\nQ:is this in a park\nA:yes, i believe it is\nQ:are there others around\nA:no, she is alone\nQ:does she have a collection bucket\nA:no\nQ:is her hair long\nA:yes, pretty long\nQ:is she wearing a dress\nA:i don't think so, hard to tell\nQ:does she have shoes on\nA:yes, flip flops\nQ:is there grass nearby\nA:yes, everywhere\nQ:is it a sunny day\nA:yes\nQ:are there trees\nA:in the background there are trees\nQ:is the guitar new\nA:i don't think so\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 70266,
            "sha1": "10f601a47c31",
            "format": "JPEG",
            "size": [
              640,
              446
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1835",
      "text": "<|image_pad|>\nRepresent the given image\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 28083,
            "sha1": "0f0703191836",
            "format": "JPEG",
            "size": [
              500,
              375
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_1"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
