# ImageNet-1K

> Classic large-scale image classification benchmark for evaluating general visual semantic discrimination.

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
| Original Name | ImageNet (ILSVRC 2012) |
| Paper | ImageNet Large Scale Visual Recognition Challenge ([arXiv](https://arxiv.org/abs/1409.0575)) |
| Primary Authors | Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, Li Fei-Fei |
| HuggingFace | [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| License | See original dataset |

### Schema
Typical classification samples are image + class label; in MMEB they are uniformly mapped to instruction-style queries and candidate label sets.

### Sample
> Sample screenshots to be added.

---

## MMEB-V2 (VLM2Vec Converted)

### Source Info
| Property | Value |
|------|-----|
| HuggingFace | [ziyjiang/MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct) |
| Subset Name | `ImageNet-1K` |
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
| Directory | `MMEB-V2/image-tasks/ImageNet-1K/` |
| task_type | `image_classification` |
| Eval Mode | local |

### Lance Layout

**queries.lance**: `id`, `text`, `images`  
**candidates.lance**: `id`, `text`, `images`  
**qrels.lance**: `query_id`, `mode`, `candidate_ids`, `candidate_scores`

**metadata.json**:
```json
{
  "name": "ImageNet-1K",
  "task_type": "image_classification",
  "num_queries": "...",
  "num_candidates": "..."
}
```

### Sample

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=q_0`, `positive_candidate_id=c_317`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|image_pad|>\nRepresent the given image for classification\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 39025,
          "sha1": "7ddac4291320",
          "format": "JPEG",
          "size": [
            408,
            500
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_317",
      "text": "coucal",
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
      "c_317"
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
`query_id=q_1`, `positive_candidate_id=c_65`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|image_pad|>\nRepresent the given image for classification\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 34385,
          "sha1": "a86566606b54",
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
      "id": "c_65",
      "text": "Italian greyhound",
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
      "c_65"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
