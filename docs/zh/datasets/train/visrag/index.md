# VisRAG 训练数据集

Visual Retrieval-Augmented Generation 训练数据。7 个子集，2 层溯源，单表 Lance schema。

**HuggingFace:** openbmb/VisRAG-Ret-Train-In-domain-data

**Schema:** 单表（source-specific prompt）

**溯源:** 2 层（原始数据 → 注册）

---

## 子集列表

| 子集 | 配置名 | 权重 | 样本数 |
|------|--------|------|--------|
| [VisRAG In-domain](./visrag-indomain.md) | visrag-indomain | 12 | 122,752 |
| [InfoVQA](./infovqa.md) | visrag-indomain_InfoVQA | 2 | 17,664 |
| [ArXiv QA](./arxivqa.md) | visrag-indomain_ArxivQA | 3 | 25,856 |
| [PlotQA](./plotqa.md) | visrag-indomain_PlotQA | 3 | 56,192 |
| [SlideVQA](./slidevqa.md) | visrag-indomain_SlideVQA | 1 | 8,192 |
| [MP-DocVQA](./mp-docvqa.md) | visrag-indomain_MP-DocVQA | 1 | 10,624 |
| [ChartQA](./chartqa.md) | visrag-indomain_ChartQA | 0.5 | 4,224 |
