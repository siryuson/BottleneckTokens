# 评测数据集总览

共 78 个评测子集，覆盖 3 大模态，统一使用 Lance 三文件布局（queries / candidates / qrels）。

**指标:** hit@1 (image/video), ndcg@5 (visdoc)

---

## 模态

| 模态 | 子集数 | 任务类型 |
|------|--------|---------|
| [Image](./image/index.md) | 36 | 分类、VQA、检索、视觉定位 |
| [Video](./video/index.md) | 18 | 分类、QA、检索、时序检索 |
| [VisDoc](./visdoc/index.md) | 24 | ViDoRe V1、ViDoRe V2、VisRAG、OOD |

---

### Image

图像理解评测数据集。36 个子集，4 种任务类型。

详见 [Image 评测数据集](./image/index.md)。

### Video

视频理解评测数据集。18 个子集，4 种任务类型。

详见 [Video 评测数据集](./video/index.md)。

### VisDoc

视觉文档理解评测数据集。24 个子集，4 种分组。

详见 [VisDoc 评测数据集](./visdoc/index.md)。
