# Country211

> Country geolocation classification subset for evaluating visual geo-semantic recognition ability.

| Property | Value |
|------|-----|
| Modality | image |
| Task Type | `image_classification` |
| Usage | Evaluation |
| Provenance Layers | 3 layers |

## Original Dataset
### Source Info
| Property | Value |
|------|-----|
| Original Name | Country211 |
| Paper | Learning Transferable Visual Models From Natural Language Supervision (Country211 provided in appendix)([arXiv](https://arxiv.org/abs/2103.00020)) |
| Primary Authors | Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever |
| HuggingFace | [osv5m/country211](https://huggingface.co/datasets/osv5m/country211) |
| License | See original dataset |

### Schema
Image + ISO country label.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `Country211` |
| Split | `test` |
| Eval Parser | `image_cls` |

### Schema
`qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`.

### Sample
> Sample screenshots to be added.

---

## BToks Lance Format
### Source Info
| Property | Value |
|------|-----|
| Directory | `MMEB-V2/image-tasks/Country211/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_63`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nIdentify the country depicted in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 47540,
          "sha1": "c6d693404781",
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
      "id": "c_63",
      "text": "Fiji",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_63"
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
`query_id=q_1`, `positive_candidate_id=c_117`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nIdentify the country depicted in the image\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 19041,
          "sha1": "52d9ca9ee579",
          "format": "JPEG",
          "size": [
            499,
            332
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_117",
      "text": "Maldives",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_117"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
