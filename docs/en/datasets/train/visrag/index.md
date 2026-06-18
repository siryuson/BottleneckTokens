# VisRAG Training Datasets

Visual Retrieval-Augmented Generation training data. 7 subsets with 2-layer provenance, single-table Lance schema.

**HuggingFace:** openbmb/VisRAG-Ret-Train-In-domain-data

**Schema:** Single table (source-specific prompt)

**Provenance:** 2 layers (raw data -> registered)

---

## Subsets

| Subset | Config Name | Weight | Samples |
|--------|-------------|--------|---------|
| [VisRAG In-domain](./visrag-indomain.md) | visrag-indomain | 12 | 122,752 |
| [InfoVQA](./infovqa.md) | visrag-indomain_InfoVQA | 2 | 17,664 |
| [ArXiv QA](./arxivqa.md) | visrag-indomain_ArxivQA | 3 | 25,856 |
| [PlotQA](./plotqa.md) | visrag-indomain_PlotQA | 3 | 56,192 |
| [SlideVQA](./slidevqa.md) | visrag-indomain_SlideVQA | 1 | 8,192 |
| [MP-DocVQA](./mp-docvqa.md) | visrag-indomain_MP-DocVQA | 1 | 10,624 |
| [ChartQA](./chartqa.md) | visrag-indomain_ChartQA | 0.5 | 4,224 |
