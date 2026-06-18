# Training Datasets Overview

The training dataset docs are maintained as implemented families, standalone runtimes, and candidates that still need source confirmation. Rewritten training entries output `TrainSample`, preserve raw table fields during conversion, and store images or videos in shared Lance tables.

## Implemented Training Dataset Families

| Family | Subsets | Storage Boundary | Lance Schema |
|--------|---------|------------------|--------------|
| [MMEB](./mmeb/index.md) | 20 | raw-preserving | sample table + image side table |
| [ViDoRe](./vidore/index.md) | 6 | raw-preserving | Single table |
| [VisRAG](./visrag/index.md) | 7 | raw-preserving | Single table |
| [LLaVA-Hound](./llavahound/index.md) | 3 | rewritten | Dual-table (frames + instructions) |
| Standalone video-classification runtimes | 5 | Implemented; v1-enabled production roots and the v2 `K700` production root have been migrated to frame units | Split tables + `data/videos.lance` + `data/frames.lance` |
| Standalone video-retrieval runtimes | 5 | Implemented; v1-enabled production roots and the second-round `VATEX` production root have been migrated to frame units | Split tables + `data/videos.lance` + `data/frames.lance` |
| Standalone video-moment-retrieval runtimes | 3 | Implemented; production roots have been migrated to mixed-view / segment frame units | Split tables + `data/videos.lance` + `data/frames.lance` |
| Standalone OOD / VQA runtimes | 9 | raw-preserving | Split tables + image/page side table |
| [M-BEIR](./mbeir/index.md) | 8 | raw-preserving | Query tables + candidate tables + `data/images.lance` |
| [Visual Grounding](./visual-grounding/index.md) | 1 | raw-preserving | Split tables + `data/images.lance` |

## Design-only Cards (Not Yet Landed as Production Training Implementations)

| Family | Subsets | Current Status |
|--------|---------|----------------|
| OOD remaining candidates | 0 | No downloaded OOD design-only card is ready to enter training directly |
| Long-term pretraining candidates | several | `LAION`, `CC3M/CC12M`, `WebVid`, and similar sources are outside this finetuning rewrite |
| eval-only candidates | several | `ImageNet-A/R`, `ObjectNet`, `RefCOCO+ / RefCOCOg eval`, `FashionIQ eval`, and similar sources keep their eval boundary |
| [MMEB-V2 Training Source Survey](./mmeb-v2-training-source-survey.md) | 78 | Records whether eval subsets have training sources and follow-up audit entries only; no downloads or implementation |
| [v2 Supplement Source Manifest](./v2-supplement-source-manifest.md) | several | Records post-v3 v2 supplement download sources, media dependencies, local targets, and follow-up audit items |

## Notes

- Expanded training configs now use versioned layers: `configs/datasets/expanded_train_datasets_v1.yaml` is the first-round baseline, and `configs/datasets/expanded_train_datasets_v2.yaml` inherits v1 and appends the completed second-round datasets, so it can be used directly as the full v1 + v2 training config. The current v2 layer appends `VizWiz`, `RefCOCO`, `RefCOCO-Matching`, `VATEX`, `K700`, and `VideoChat2-IT`. `configs/datasets/expanded_train_datasets_v3.yaml` inherits v2 and lists the Stage 1 full-training candidates under `_stage1_candidates`; these candidates are audit, conversion, and sampling-policy entries, not trainable top-level dataset entries. The matching BToks layers are `configs/datasets/btoks_expanded_train_datasets_v1.yaml` and `configs/datasets/btoks_expanded_train_datasets_v2.yaml`. The v2 / v3 files are not loaded by existing default training presets; recipes that need the second-round expansion or Stage 1 candidate catalog should explicitly reference the corresponding version.
- “Implemented” on this page means the repository has the production training registry, conversion/runtime/tests, and config entry. It does not mean the current production data root already satisfies default sampling requirements. Video datasets also need `data/frames.lance` or an equivalent frame-unit root mapping before they can train through the default config; missing tables should fail unless a raw-video compatibility path is explicitly enabled. Registered entries include: `mmeb_train`, `vidore_train`, `visrag_train`, `llavahound_train`, `ssv2_train`, `hmdb51_train`, `ucf101_train`, `kinetics700_train`, `breakfast_train`, `msrvtt_train`, `msvd_train`, `didemo_train`, `youcook2_train`, `charades_sta_train`, `qvhighlight_train`, `activitynet_captions_train`, `place365_train`, `country211_train`, `scienceqa_train`, `gqa_train`, `textvqa_train`, `vizwiz_train`, `vatex_train`, `nextqa_train`, `mbeir_train`, `wikissnq_train`, `mmlongbench_doc_train`, `refcoco_train`, `visual7w_pointing_train`, and `videochat2_it_train`. `VATEX`, `K700`, and `VideoChat2-IT` are enabled in v2.
- The MMEB-V2 eval-subset training-source expansion is tracked in [MMEB-V2 Training Source Survey](./mmeb-v2-training-source-survey.md), which is only an entry point for future single-dataset audits.
- Some standalone video, retrieval, OOD, and VQA runtimes still do not have separate card directories. Their audit conclusions, split views, and legacy-entry cleanup status are recorded in [Pending Training Datasets](./pending-datasets.md).
- The target default for video training runtimes is converted `data/frames.lance` media, avoiding raw video decode during training-time sampling. Current v1-enabled video production roots, the v2 `VATEX` production root, and the v2 `K700` production root now have the `data/frames.lance` table required by default training; `LLaVA-Hound` uses the separate frame table `data/train_300k.lance`. Full videos or already clipped short videos keep up to 64 frames; explicit temporal segments and row-level segments keep up to 8 frames.
- `Charades-STA` and `QVHighlights` use a mixed-view layout: the query side samples a full-video frame unit, while the positive side samples the target segment frame unit. `YouCook2` rebuilds segment frame units from upstream parquet `binary_frames` rows and no longer uses full videos as positives.
- Remaining design cards describe future candidates or audit boundaries. They should not be read as datasets that are ready for training.
- Additional archive-era datasets that still have not landed are tracked in [Pending Training Datasets](./pending-datasets.md).
