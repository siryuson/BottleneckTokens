# M-BEIR 训练数据集

> 状态：已实现。当前训练入口为 `mbeir_train`，每个 `dataset_name` 保留 M-BEIR 原始 query / candidate 字段，并通过共享 `data/images.lance` 查表得到图像。

Multimodal Benchmark for Information Retrieval (M-BEIR) 训练数据集。

**HuggingFace:** TIGER-Lab/M-BEIR

**论文:** One Embedder, Any Task: Instruction-Finetuned Text Embeddings / UniIR: Training and Benchmarking Universal Multimodal Information Retrievers ([arXiv:2311.17136](https://arxiv.org/abs/2311.17136))

**当前实现:** `src/vlm2emb/data/conversion/mbeir_train.py`、`src/vlm2emb/data/datasets/mbeir_train.py`、`scripts/convert/train/convert_mbeir_to_lance.py`

runtime 只负责按 task 语义组织 query / positive / negative、插入必要的 visual token 并统一尾部换行。M-BEIR 原始 query / candidate 文本保持原样，不在本轮执行 detokenize、段落标题清理、句间补空格或其它内容级文本清洗。

---

## 子集列表

| 子集 | 配置名 | 任务 ID | 默认训练行数 | 排除行数 | 任务形式 |
|------|--------|---------|--------------|----------|----------|
| [OVEN](./oven.md) | `OVEN_it2it_train` | 8 | 143,051 | 11,409 | image+text -> image+text |
| [OVEN](./oven.md) | `OVEN_it2t_train` | 6 | 132,399 | 18,354 | image+text -> text |
| [EDIS](./edis.md) | `EDIS_train` | 2 | 22,676 | 3,241 | text -> image+text |
| [FashionIQ](./fashioniq.md) | `FashionIQ_train` | 7 | 10,755 | 5,421 | image+text -> image |
| Fashion200K | `Fashion200K_t2i` | 0 | 14,123 | 877 | text -> image |
| Fashion200K | `Fashion200K_i2t` | 3 | 12,558 | 2,442 | image -> text |
| INFOSEEK | `INFOSEEK_it2t` | 6 | 135,613 | 5,643 | image+text -> text |
| INFOSEEK | `INFOSEEK_it2it` | 8 | 134,330 | 8,801 | image+text -> image+text |

默认 split 为 `official_train_without_mmeb_v2_eval`。`official_train` 和 `data/exclusions/*_mmeb_v2_eval.lance` 保留为审计视图，便于复查与 MMEB-V2 eval 的重叠。

## Lance 结构

| 路径 | 内容 |
|------|------|
| `data/<dataset_name>/official_train.lance` | 原始训练 query 表 |
| `data/<dataset_name>/official_train_without_mmeb_v2_eval.lance` | 默认训练 query 表 |
| `data/exclusions/<dataset_name>_mmeb_v2_eval.lance` | 被排除的 eval 边界样本 |
| `data/candidates/<dataset_name>.lance` | 正样本候选表 |
| `data/images.lance` | 共享图像表，字段为 `path`、`image` |

query 表字段保留为 `qid`、`query_txt`、`query_img_path`、`query_modality`、`query_src_content`、`pos_cand_list`、`neg_cand_list`、`task_id`。candidate 表字段保留为 `did`、`txt`、`img_path`、`modality`、`src_content`。
