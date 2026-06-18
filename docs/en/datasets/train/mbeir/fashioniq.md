# FashionIQ Training Dataset

> Status: implemented. The current runtime is `mbeir_train` with `dataset_name: FashionIQ_train`. FashionIQ eval remains an MMEB-V2 eval boundary and does not get a training runtime.

| Field | Value |
|-------|-------|
| Source | [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| Original Task | Fashion Image Query |
| Runtime | `mbeir_train` |
| Conversion | `convert_mbeir_to_lance.py` |
| task_id | 7 |
| Default Split | `official_train_without_mmeb_v2_eval` |

## Row Counts

| Split | Rows |
|-------|------|
| `official_train` | 16,176 |
| `official_train_without_mmeb_v2_eval` | 10,755 |
| `data/exclusions/FashionIQ_train_mmeb_v2_eval.lance` | 5,421 |
| `data/candidates/FashionIQ_train.lance` | 16,064 |

## Schema

The query table preserves the original M-BEIR fields: `qid`, `query_txt`, `query_img_path`, `query_modality`, `query_src_content`, `pos_cand_list`, `neg_cand_list`, and `task_id`.

The candidate table preserves `did`, `txt`, `img_path`, `modality`, and `src_content`.

## Runtime Sample

```yaml
query:
  text: "<|image_pad|>\nFind an image to match the fashion image and style note: Is red black and gray and is a darker print.\n"
  media: 1 image
positive:
  text: "<|image_pad|>\nRepresent the given image.\n"
  media: 1 image
negative:
  text: ""
  media: []
```
