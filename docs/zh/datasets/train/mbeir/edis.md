# EDIS 训练数据集

> 状态：已实现。当前入口为 `mbeir_train`，`dataset_name` 为 `EDIS_train`。

| 属性 | 值 |
|------|-----|
| 来源 | [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| 原始任务 | Entity-Driven Image Search over Multimodal Web Content |
| Runtime | `mbeir_train` |
| Conversion | `convert_mbeir_to_lance.py` |
| task_id | 2 |
| 默认 split | `official_train_without_mmeb_v2_eval` |

## 行数

| split | 行数 |
|-------|------|
| `official_train` | 25,917 |
| `official_train_without_mmeb_v2_eval` | 22,676 |
| `data/exclusions/EDIS_train_mmeb_v2_eval.lance` | 3,241 |
| `data/candidates/EDIS_train.lance` | 25,917 |

## Schema

query 表保留原始 M-BEIR 字段：`qid`、`query_txt`、`query_img_path`、`query_modality`、`query_src_content`、`pos_cand_list`、`neg_cand_list`、`task_id`。

candidate 表保留：`did`、`txt`、`img_path`、`modality`、`src_content`。

## Runtime 输出样例

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
