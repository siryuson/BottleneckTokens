# FashionIQ 训练数据集

> 状态：已实现。当前入口为 `mbeir_train`，`dataset_name` 为 `FashionIQ_train`。FashionIQ eval 保持 MMEB-V2 eval 边界，不创建训练 runtime。

| 属性 | 值 |
|------|-----|
| 来源 | [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| 原始任务 | Fashion Image Query |
| Runtime | `mbeir_train` |
| Conversion | `convert_mbeir_to_lance.py` |
| task_id | 7 |
| 默认 split | `official_train_without_mmeb_v2_eval` |

## 行数

| split | 行数 |
|-------|------|
| `official_train` | 16,176 |
| `official_train_without_mmeb_v2_eval` | 10,755 |
| `data/exclusions/FashionIQ_train_mmeb_v2_eval.lance` | 5,421 |
| `data/candidates/FashionIQ_train.lance` | 16,064 |

## Schema

query 表保留原始 M-BEIR 字段：`qid`、`query_txt`、`query_img_path`、`query_modality`、`query_src_content`、`pos_cand_list`、`neg_cand_list`、`task_id`。

candidate 表保留：`did`、`txt`、`img_path`、`modality`、`src_content`。

## Runtime 输出样例

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
