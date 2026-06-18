# MMEB Training Datasets

Massive Multimodal Embedding Benchmark (MMEB) training data. 20 subsets using a raw sample table plus an image side table.

> Training does not use the `MMEB-V2 eval` strict/canonical storage contract.
> MMEB historically used `<|image_1|>` as the source token; runtime input uses `STANDARD_IMAGE_TOKEN` as the standard token by default.
> In other words, raw parquet fields and the `path/image` image-table join remain stored facts, while legacy-token normalization is runtime transform policy.

**HuggingFace:** TIGER-Lab/MMEB-train

**Schema:** raw sample table + image side table

---

## Subsets

| Subset | Config Name | Task Type | Weight | Max Samples |
|--------|-------------|-----------|--------|-------------|
| [ImageNet-1K](./imagenet-1k.md) | ImageNet_1K | Classification | 1 | 100,000 |
| [N24News](./n24news.md) | N24News | Classification | 1 | 50,000 |
| [HatefulMemes](./hateful-memes.md) | HatefulMemes | Classification | 0.5 | 10,000 |
| [VOC2007](./voc2007.md) | VOC2007 | Classification | 0.5 | 10,000 |
| [SUN397](./sun397.md) | SUN397 | Classification | 0.5 | 20,000 |
| [OK-VQA](./ok-vqa.md) | OK-VQA | VQA | 0.5 | 10,000 |
| [A-OKVQA](./a-okvqa.md) | A-OKVQA | VQA | 0.5 | 20,000 |
| [DocVQA](./docvqa.md) | DocVQA | VQA | 1 | 40,000 |
| [InfographicsVQA](./infographicsvqa.md) | InfographicsVQA | VQA | 0.5 | 25,000 |
| [ChartQA](./chartqa.md) | ChartQA | VQA | 0.5 | 28,000 |
| [Visual7W](./visual7w.md) | Visual7W | VQA | 1 | 70,000 |
| [VisDial](./visdial.md) | VisDial | Retrieval (T→I) | 1 | 130,000 |
| [CIRR](./cirr.md) | CIRR | Retrieval (I+T→I) | 0.5 | 30,000 |
| [VisualNews T2I](./visualnews-t2i.md) | VisualNews_t2i | Retrieval (T→I) | 1 | 100,000 |
| [VisualNews I2T](./visualnews-i2t.md) | VisualNews_i2t | Retrieval (I→T) | 1 | 100,000 |
| [MSCOCO T2I](./mscoco-t2i.md) | MSCOCO_t2i | Retrieval (T→I) | 1 | 100,000 |
| [MSCOCO I2T](./mscoco-i2t.md) | MSCOCO_i2t | Retrieval (I→T) | 1 | 120,000 |
| [NIGHTS](./nights.md) | NIGHTS | Retrieval (I→I) | 0.5 | 20,000 |
| [WebQA](./webqa.md) | WebQA | Retrieval (T→I+T) | 0.5 | 20,000 |
| [MSCOCO VG](./mscoco.md) | MSCOCO | Visual Grounding | 1 | 100,000 |
