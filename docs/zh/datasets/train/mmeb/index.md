# MMEB 训练数据集

Massive Multimodal Embedding Benchmark (MMEB) 训练数据。20 个子集，使用 raw sample 表 + image side table 的 Lance schema。

> 训练侧不使用 `MMEB-V2 eval` 的 `strict / canonical` 存储语义。
> MMEB 历史源 token 为 `<|image_1|>`；运行时输入默认使用 `STANDARD_IMAGE_TOKEN` 作为标准口径。
> 也就是说，raw parquet 字段与 `path/image` 图像表 join 属于存储事实；legacy token 归一属于 runtime transform policy。

**HuggingFace:** TIGER-Lab/MMEB-train

**Schema:** raw sample 表 + image side table

---

## 子集列表

| 子集 | 配置名 | 任务类型 | 权重 | 最大样本数 |
|------|--------|---------|------|-----------|
| [ImageNet-1K](./imagenet-1k.md) | ImageNet_1K | 分类 | 1 | 100,000 |
| [N24News](./n24news.md) | N24News | 分类 | 1 | 50,000 |
| [HatefulMemes](./hateful-memes.md) | HatefulMemes | 分类 | 0.5 | 10,000 |
| [VOC2007](./voc2007.md) | VOC2007 | 分类 | 0.5 | 10,000 |
| [SUN397](./sun397.md) | SUN397 | 分类 | 0.5 | 20,000 |
| [OK-VQA](./ok-vqa.md) | OK-VQA | VQA | 0.5 | 10,000 |
| [A-OKVQA](./a-okvqa.md) | A-OKVQA | VQA | 0.5 | 20,000 |
| [DocVQA](./docvqa.md) | DocVQA | VQA | 1 | 40,000 |
| [InfographicsVQA](./infographicsvqa.md) | InfographicsVQA | VQA | 0.5 | 25,000 |
| [ChartQA](./chartqa.md) | ChartQA | VQA | 0.5 | 28,000 |
| [Visual7W](./visual7w.md) | Visual7W | VQA | 1 | 70,000 |
| [VisDial](./visdial.md) | VisDial | 检索 (T→I) | 1 | 130,000 |
| [CIRR](./cirr.md) | CIRR | 检索 (I+T→I) | 0.5 | 30,000 |
| [VisualNews T2I](./visualnews-t2i.md) | VisualNews_t2i | 检索 (T→I) | 1 | 100,000 |
| [VisualNews I2T](./visualnews-i2t.md) | VisualNews_i2t | 检索 (I→T) | 1 | 100,000 |
| [MSCOCO T2I](./mscoco-t2i.md) | MSCOCO_t2i | 检索 (T→I) | 1 | 100,000 |
| [MSCOCO I2T](./mscoco-i2t.md) | MSCOCO_i2t | 检索 (I→T) | 1 | 120,000 |
| [NIGHTS](./nights.md) | NIGHTS | 检索 (I→I) | 0.5 | 20,000 |
| [WebQA](./webqa.md) | WebQA | 检索 (T→I+T) | 0.5 | 20,000 |
| [MSCOCO VG](./mscoco.md) | MSCOCO | 视觉定位 | 1 | 100,000 |
