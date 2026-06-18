# M-BEIR Training Datasets

> Status: implemented. The training registry is `mbeir_train`. Each `dataset_name` keeps the original M-BEIR query / candidate fields and looks up images from the shared `data/images.lance` table.

**HuggingFace:** TIGER-Lab/M-BEIR

**Paper:** One Embedder, Any Task: Instruction-Finetuned Text Embeddings / UniIR: Training and Benchmarking Universal Multimodal Information Retrievers ([arXiv:2311.17136](https://arxiv.org/abs/2311.17136))

**Implementation:** `src/vlm2emb/data/conversion/mbeir_train.py`, `src/vlm2emb/data/datasets/mbeir_train.py`, `scripts/convert/train/convert_mbeir_to_lance.py`

Runtime only organizes query / positive / negative samples according to task semantics, inserts required visual tokens, and normalizes trailing newlines. Original M-BEIR query / candidate text is preserved as source text; this round does not perform detokenization, section-heading cleanup, sentence-boundary spacing fixes, or other content-level text cleaning.

---

## Subsets

| Subset | Config Name | Task ID | Default Train Rows | Excluded Rows | Task Form |
|--------|-------------|---------|--------------------|---------------|-----------|
| [OVEN](./oven.md) | `OVEN_it2it_train` | 8 | 143,051 | 11,409 | image+text -> image+text |
| [OVEN](./oven.md) | `OVEN_it2t_train` | 6 | 132,399 | 18,354 | image+text -> text |
| [EDIS](./edis.md) | `EDIS_train` | 2 | 22,676 | 3,241 | text -> image+text |
| [FashionIQ](./fashioniq.md) | `FashionIQ_train` | 7 | 10,755 | 5,421 | image+text -> image |
| Fashion200K | `Fashion200K_t2i` | 0 | 14,123 | 877 | text -> image |
| Fashion200K | `Fashion200K_i2t` | 3 | 12,558 | 2,442 | image -> text |
| INFOSEEK | `INFOSEEK_it2t` | 6 | 135,613 | 5,643 | image+text -> text |
| INFOSEEK | `INFOSEEK_it2it` | 8 | 134,330 | 8,801 | image+text -> image+text |

The default split is `official_train_without_mmeb_v2_eval`. `official_train` and `data/exclusions/*_mmeb_v2_eval.lance` are kept as audit views for reviewing overlap with MMEB-V2 eval.

## Lance Layout

| Path | Contents |
|------|----------|
| `data/<dataset_name>/official_train.lance` | Original training query table |
| `data/<dataset_name>/official_train_without_mmeb_v2_eval.lance` | Default training query table |
| `data/exclusions/<dataset_name>_mmeb_v2_eval.lance` | Excluded eval-boundary samples |
| `data/candidates/<dataset_name>.lance` | Positive candidate table |
| `data/images.lance` | Shared image table with `path` and `image` |

The query table preserves `qid`, `query_txt`, `query_img_path`, `query_modality`, `query_src_content`, `pos_cand_list`, `neg_cand_list`, and `task_id`. The candidate table preserves `did`, `txt`, `img_path`, `modality`, and `src_content`.
