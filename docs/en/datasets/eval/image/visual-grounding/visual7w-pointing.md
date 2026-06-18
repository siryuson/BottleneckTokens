# Visual7W-Pointing

> Pointing-style QA subtask of Visual7W for evaluating target-region localization ability.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_visual_grounding` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset
### Source Info
| Property | Value |
|------|-----|
| Original Name | Visual7W Pointing |
| Paper | Visual7W: Grounded Question Answering in Images ([arXiv](https://arxiv.org/abs/1511.07332)) |
| Primary Authors | Yuke Zhu, Oliver Groth, Michael Bernstein, Li Fei-Fei |
| HuggingFace | [HuggingFaceM4/Visual7W](https://huggingface.co/datasets/HuggingFaceM4/Visual7W) |
| License | See original dataset |

### Schema
Image, question, candidate boxes, and correct pointed target.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `Visual7W-Pointing` |
| Split | `test` |
| Eval Parser | `image_i2i_vg` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/Visual7W-Pointing/` |
| task_type | `image_visual_grounding` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_223`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nSelect the portion of the image that answers the question \"Which box frames the toilet?\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 27250,
          "sha1": "6265464273bf",
          "format": "JPEG",
          "size": [
            375,
            500
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_223",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 10008,
            "sha1": "a2ab24abd487",
            "format": "JPEG",
            "size": [
              138,
              179
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_223"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### Example 2
`query_id=q_1`, `positive_candidate_id=c_453`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nSelect the portion of the image that answers the question \"Which is the stairs under the skater?\"\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 38652,
          "sha1": "69e5e40cc21d",
          "format": "JPEG",
          "size": [
            500,
            375
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_453",
      "text": "<|image_pad|>\nRepresent the given cropped image of the object.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 18206,
            "sha1": "19f1995ae846",
            "format": "JPEG",
            "size": [
              165,
              208
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_453"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
