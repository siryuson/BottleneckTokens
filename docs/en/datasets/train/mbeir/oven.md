# OVEN Training Dataset

> Status: implemented. OVEN currently provides two M-BEIR training `dataset_name` values: `OVEN_it2it_train` for image+text -> image+text retrieval and `OVEN_it2t_train` for image+text -> text retrieval.

| Field | Value |
|-------|-------|
| Source | [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| Original Task | Open-domain Visual Entity Recognition |
| Runtime | `mbeir_train` |
| Conversion | `convert_mbeir_to_lance.py` |
| Default Split | `official_train_without_mmeb_v2_eval` |

## Row Counts

| dataset_name | task_id | official_train | Default Train Split | MMEB-V2 Eval Exclusion |
|--------------|---------|----------------|---------------------|------------------------|
| `OVEN_it2it_train` | 8 | 154,460 | 143,051 | 11,409 |
| `OVEN_it2t_train` | 6 | 150,753 | 132,399 | 18,354 |

## Schema

The query table preserves the original M-BEIR fields: `qid`, `query_txt`, `query_img_path`, `query_modality`, `query_src_content`, `pos_cand_list`, `neg_cand_list`, and `task_id`.

The candidate table preserves `did`, `txt`, `img_path`, `modality`, and `src_content`.

Images are stored in the shared `data/images.lance` table with `path` and `image`. Runtime lookup uses `query_img_path` and the positive candidate `img_path`.

## Runtime Samples

`OVEN_it2it_train`:

```yaml
query:
  text: "<|image_pad|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image. What is the model of this aircraft?\n"
  media: 1 image
positive:
  text: "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Dassault Falcon 900. The Dassault Falcon 900, commonly abbreviated as the F900, is a French-built corporate trijet aircraft made by Dassaul..."
  media: 1 image
negative:
  text: ""
  media: []
```

`OVEN_it2t_train`:

```yaml
query:
  text: "<|image_pad|>\nRetrieve a Wikipedia paragraph that provides an answer to the given query about the image. What is the manufacturer of this aircraft?\n"
  media: 1 image
positive:
  text: "Antonov. Antonov State Enterprise (), formerly the Aeronautical Scientific-Technical Complex named after Antonov (Antonov ASTC) (), and earlier the Antonov Design Bureau..."
  media: []
negative:
  text: ""
  media: []
```
