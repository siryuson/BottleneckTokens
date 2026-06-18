# EDIS

> News-domain entity-driven image retrieval dataset (text-to-image).

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
| Original Name | EDIS |
| Paper | EDIS: Entity-Driven Image Search over Multimodal Web Content ([arXiv](https://arxiv.org/abs/2305.13631)) |
| Primary Authors | Siqi Liu, Weixi Feng, Tsu-jui Fu, Wenhu Chen, William Yang Wang |
| HuggingFace | [Emerisly/EDIS](https://huggingface.co/datasets/Emerisly/EDIS) |
| License | See original dataset |

### Schema
Text query, image candidates, and accompanying text descriptions.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `EDIS` |
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
| Directory | `MMEB-V2/image-tasks/EDIS/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_1505`, `negative_candidate_id=c_3122`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Find a news image that matches the provided caption: Former Israeli president and prime minister Shimon Peres right and Palestinian President Mahmoud Abbas arrive at the World Economic Forum in Southern Shuneh Jordan in May 2015.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1505",
      "text": "<|image_pad|>\nRepresent the given image with related text information: WorldAs the West pays tribute to Peres, many Arabs recall a legacy of destruction.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9168,
            "sha1": "ca542c3af0c4",
            "format": "JPEG",
            "size": [
              358,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_3122",
      "text": "<|image_pad|>\nRepresent the given image with related text information: • Discovered by the Germans in 1904, they named it San Diego, which of course in German means 'a whale's vagina'.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 21951,
            "sha1": "43274d6ba8b3",
            "format": "JPEG",
            "size": [
              426,
              256
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
      "c_1505"
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
`query_id=q_1`, `positive_candidate_id=c_1595`, `negative_candidate_id=c_1334`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Find a news image that matches the provided caption: Tom Holland makes his debut in the Spidey suit in Captain America Civil War.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1595",
      "text": "<|image_pad|>\nRepresent the given image with related text information: Comic RiffsJon Favreau is set to reprise his Iron Man role for Spider Man: Homecoming.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 9441,
            "sha1": "a9d04212c901",
            "format": "JPEG",
            "size": [
              454,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1334",
      "text": "<|image_pad|>\nRepresent the given image with related text information: GovBeatColorado students are protesting en masse over a curriculum proposal to promote respect for authority.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 22049,
            "sha1": "83d9ec61b335",
            "format": "JPEG",
            "size": [
              332,
              256
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
      "c_1595"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
