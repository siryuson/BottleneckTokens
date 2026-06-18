# ViDoRe 训练数据集

Visual Document Retrieval (ViDoRe) 训练数据。6 个子集，2 层溯源，单表 Lance schema。

> 训练侧不使用 `MMEB-V2 eval` 的 `strict / canonical` 存储语义。
> ViDoRe 的 query 规则是“image token + query text”，positive 仅保留答案文本，negative 为空。

**HuggingFace:** vidore/colpali_train_set

**Schema:** 单表（image struct 内嵌）

**溯源:** 2 层（原始数据 → 注册）

---

## 子集列表

| 子集 | 配置名 | 权重 | 样本数 |
|------|--------|------|--------|
| [ColPali Train Set](./colpali-train-set.md) | colpali_train_set | 10 | 118,195 |
| [ArXiv QA](./arxiv-qa.md) | colpali_train_set_arxiv_qa | 1 | 9,949 |
| [PDF](./pdf.md) | colpali_train_set_pdf | 2 | 45,718 |
| [DocVQA](./docvqa.md) | colpali_train_set_docvqa | 2 | 39,302 |
| [TAT-DQA](./tatdqa.md) | colpali_train_set_tatdqa | 1 | 13,188 |
| [Infographic-VQA](./infographic-vqa.md) | colpali_train_set_Infographic-VQA | 1 | 10,038 |
