# ViDoRe Training Datasets

Visual Document Retrieval (ViDoRe) training data. 6 subsets with 2-layer provenance, single-table Lance schema.

> Training does not use the `MMEB-V2 eval` strict/canonical storage contract.
> ViDoRe queries are built as "image token + query text"; positive keeps answer text only, and negative is empty.

**HuggingFace:** vidore/colpali_train_set

**Schema:** Single table (image struct embedded)

**Provenance:** 2 layers (raw data -> registered)

---

## Subsets

| Subset | Config Name | Weight | Samples |
|--------|-------------|--------|---------|
| [ColPali Train Set](./colpali-train-set.md) | colpali_train_set | 10 | 118,195 |
| [ArXiv QA](./arxiv-qa.md) | colpali_train_set_arxiv_qa | 1 | 9,949 |
| [PDF](./pdf.md) | colpali_train_set_pdf | 2 | 45,718 |
| [DocVQA](./docvqa.md) | colpali_train_set_docvqa | 2 | 39,302 |
| [TAT-DQA](./tatdqa.md) | colpali_train_set_tatdqa | 1 | 13,188 |
| [Infographic-VQA](./infographic-vqa.md) | colpali_train_set_Infographic-VQA | 1 | 10,038 |
