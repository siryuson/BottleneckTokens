# OVEN

> Open-domain visual entity recognition/retrieval subset requiring matching in a massive entity space.

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
| Original Name | OVEN |
| Paper | Open-domain Visual Entity Recognition: Towards Recognizing Millions of Wikipedia Entities ([arXiv](https://arxiv.org/abs/2302.11154)) |
| Primary Authors | Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, Ming-Wei Chang |
| HuggingFace | [openbmb/OVEN](https://huggingface.co/datasets/openbmb/OVEN) |
| License | See original dataset |

### Schema
Image, text query, and entity label (mapped to Wikipedia entities).

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)
### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `OVEN` |
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
| Directory | `MMEB-V2/image-tasks/OVEN/` |
| task_type | `image_retrieval` |
| Eval Mode | local |

### Lance Layout
`queries.lance` / `candidates.lance` / `qrels.lance` / `metadata.json`.

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_4128`, `negative_candidate_id=c_2183`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image: What is the name of this place?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 13586,
          "sha1": "c0bc03ff806e",
          "format": "JPEG",
          "size": [
            341,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_4128",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Titisee. The Titisee is a lake in the southern Black Forest in Baden-Württemberg. It covers an area of 1.3 (km2) and is an average of 20 (m) deep. It owes its formation to the Feldberg glacier, the moraines of which were formed in the Pleistocene epoch and nowadays form the shores of the lake. The lake's outflow, at 840 (m) above sea level, is the River Gutach, which merges with the Haslach stream below Kappel to form the Wutach. The waters of the Titisee thus drain eventually into the Upper Rhine between Tiengen and Waldshut. On the north shore lies the.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 12407,
            "sha1": "9f2ee9ec2d01",
            "format": "JPEG",
            "size": [
              385,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_2183",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Megumi Nakajima. is a Japanese voice actress and singer, who is affiliated with Stay Luck. Her involvement in the entertainment industry began in 2003 when she participated in an audition held by the talent agency Stardust Promotion, becoming affiliated with them after passing the audition. Her debut came in 2007 when she passed an audition held by the music company Victor Entertainment; she was then cast as the character Ranka Lee in the 2008 anime series \"Macross Frontier\". Her first solo single \"Tenshi ni Naritai\" was released in 2009, which was followed by her first solo album \"I Love You\" in 2010.Apart.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 14185,
            "sha1": "708f2776d6cb",
            "format": "JPEG",
            "size": [
              256,
              384
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
      "c_4128"
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
`query_id=q_1`, `positive_candidate_id=c_1671`, `negative_candidate_id=c_4332`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image: What is the category of this location?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 9984,
          "sha1": "41e2020ce5ae",
          "format": "JPEG",
          "size": [
            256,
            256
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1671",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: OLED. An organic light-emitting diode (OLED or organic LED), also known as organic electroluminescent (organic EL) diode, is a light-emitting diode (LED) in which the emissive electroluminescent layer is a film of organic compound that emits light in response to an electric current. This organic layer is situated between two electrodes; typically, at least one of these electrodes is transparent. OLEDs are used to create digital displays in devices such as television screens, computer monitors, and portable systems such as smartphones and handheld game consoles. A major area of research is the development of white OLED devices for use in solid-state.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 12297,
            "sha1": "507ecf063742",
            "format": "JPEG",
            "size": [
              374,
              256
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_4332",
      "text": "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Selfridges Building, Birmingham. The Selfridges Building is a landmark building in Birmingham, England. The building is part of the Bullring Shopping Centre and houses Selfridges Department Store. The building was completed in 2003 at a cost of £60 million and designed by the architecture firm Future Systems. It has a steel framework with sprayed concrete facade. Since its construction, the building has become an iconic architectural landmark and seen as a major contribution to the regeneration of Birmingham. It is included as a desktop background as part of the Architecture theme in Windows 7.## Architecture.The architecture firm Future Systems were appointed by Selfridge's.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 26217,
            "sha1": "895b5997623b",
            "format": "JPEG",
            "size": [
              402,
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
      "c_1671"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1000,
    "negative_candidate_count": 999
  }
}
```
