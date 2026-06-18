# OOD Training Datasets

**Provenance:** 2 layers (raw data -> registry)

---

## Subsets

### OOD Image Datasets

| Subset | Config Name | Weight | Samples | Task Type |
|------|--------|------|--------|----------|
| [Place365](./place365.md) | Place365 | 1 | 1,803,460 (default `official_train`) | Scene classification |
| [Country211](./country211.md) | Country211 | 1 | 31,650 | Geographic classification |
| [ScienceQA](./scienceqa.md) | ScienceQA | 1 | 6,218 (default train split) | VQA |
| [GQA](./gqa.md) | GQA | 9.2 | 943,000 (default balanced train split) | VQA |
| [TextVQA](./textvqa.md) | TextVQA | 2 | 34,496 (default train split) | VQA |
| [Wiki-SS-NQ](./wiki-ss-nq.md) | Wiki-SS-NQ | 1 | `332,518` default training samples | Image retrieval |

### OOD Document Datasets

| Subset | Config Name | Weight | Samples | Task Type |
|------|--------|------|--------|----------|
| [MMLongBench-Doc](./mmlongbench-doc.md) | MMLongBench-doc | 0.3 | 4 (default split) / 473 (archive-usable) | Document image retrieval |
