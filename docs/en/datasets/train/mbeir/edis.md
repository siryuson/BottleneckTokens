# EDIS Training Dataset

> Status: implemented. The current runtime is `mbeir_train` with `dataset_name: EDIS_train`.

| Field | Value |
|-------|-------|
| Source | [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| Original Task | Entity-Driven Image Search over Multimodal Web Content |
| Runtime | `mbeir_train` |
| Conversion | `convert_mbeir_to_lance.py` |
| task_id | 2 |
| Default Split | `official_train_without_mmeb_v2_eval` |

## Row Counts

| Split | Rows |
|-------|------|
| `official_train` | 25,917 |
| `official_train_without_mmeb_v2_eval` | 22,676 |
| `data/exclusions/EDIS_train_mmeb_v2_eval.lance` | 3,241 |
| `data/candidates/EDIS_train.lance` | 25,917 |

## Schema

The query table preserves the original M-BEIR fields: `qid`, `query_txt`, `query_img_path`, `query_modality`, `query_src_content`, `pos_cand_list`, `neg_cand_list`, and `task_id`.

The candidate table preserves `did`, `txt`, `img_path`, `modality`, and `src_content`.

## Runtime Sample

```yaml
query:
  text: "Find a news image that matches the provided caption: The president, speaking in Camden, N.J., on Monday, announced new policies that will limit military-style equipment being provided to local police forces by the federa...\n"
  media: []
positive:
  text: "<|image_pad|>\nRepresent the given image with related text information: Obama on ‘Militarized Gear’ for Police.\n"
  media: 1 image
negative:
  text: ""
  media: []
```
