# OVEN 训练数据集

> 状态：已实现。OVEN 在当前 M-BEIR 训练入口中包含 `OVEN_it2it_train` 和 `OVEN_it2t_train` 两个 `dataset_name`，分别对应 image+text -> image+text 与 image+text -> text。

| 属性 | 值 |
|------|-----|
| 来源 | [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| 原始任务 | Open-domain Visual Entity Recognition |
| Runtime | `mbeir_train` |
| Conversion | `convert_mbeir_to_lance.py` |
| 默认 split | `official_train_without_mmeb_v2_eval` |

## 行数

| dataset_name | task_id | official_train | 默认训练 split | MMEB-V2 eval 排除 |
|--------------|---------|----------------|----------------|-------------------|
| `OVEN_it2it_train` | 8 | 154,460 | 143,051 | 11,409 |
| `OVEN_it2t_train` | 6 | 150,753 | 132,399 | 18,354 |

## Schema

query 表保留原始 M-BEIR 字段：`qid`、`query_txt`、`query_img_path`、`query_modality`、`query_src_content`、`pos_cand_list`、`neg_cand_list`、`task_id`。

candidate 表保留：`did`、`txt`、`img_path`、`modality`、`src_content`。

图像统一写入共享表 `data/images.lance`，字段为 `path` 和 `image`。runtime 通过 `query_img_path` 和 positive candidate 的 `img_path` 查表。

## Runtime 输出样例

`OVEN_it2it_train`：

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

`OVEN_it2t_train`：

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
