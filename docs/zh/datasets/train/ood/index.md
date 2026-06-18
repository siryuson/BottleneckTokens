# OOD 训练数据集

分布外（Out-of-Distribution）图像和文档训练数据集，用于增强模型泛化能力。

**溯源:** 2 层（原始数据 → 注册）

---

## 子集列表

### OOD 图像数据集

| 子集 | 配置名 | 权重 | 样本数 | 任务类型 |
|------|--------|------|--------|----------|
| [Place365](./place365.md) | Place365 | 1 | 1,803,460（默认 `official_train`） | 场景分类 |
| [Country211](./country211.md) | Country211 | 1 | 31,650 | 地理分类 |
| [ScienceQA](./scienceqa.md) | ScienceQA | 1 | 6,218（默认训练 split） | VQA |
| [GQA](./gqa.md) | GQA | 9.2 | 943,000（默认 balanced 训练 split） | VQA |
| [TextVQA](./textvqa.md) | TextVQA | 2 | 34,496（默认训练 split） | VQA |
| [Wiki-SS-NQ](./wiki-ss-nq.md) | Wiki-SS-NQ | 1 | `332,518` 默认训练样本 | 图像检索 |

### OOD 文档数据集

| 子集 | 配置名 | 权重 | 样本数 | 任务类型 |
|------|--------|------|--------|----------|
| [MMLongBench-Doc](./mmlongbench-doc.md) | MMLongBench-doc | 0.3 | 4（默认 split）/ 473（archive 可用） | 文档图像检索 |
