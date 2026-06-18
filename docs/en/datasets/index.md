# Datasets Overview

This page serves as the unified entry point for all BToks dataset documentation.

At the project level, this documentation focuses on the `source / conversion / runtime` boundary. `strict / canonical` are only used for the special `MMEB-V2 eval` case and are not generalized to other training or evaluation datasets.

## Quick Links

| Documentation | Description |
|---------------|-------------|
| [Format Specifications](./format-spec.md) | Training/evaluation dataset format standards, runtime surface rules, MMEB-V2 strict/canonical exception, and the onboarding checklist |
| [Dataset Onboarding Guide](./guide-custom-dataset.md) | Project-level onboarding rules, train/eval onboarding flow, and runtime transform boundaries |
| [Data Processing Pipeline](./architecture-data-pipeline.md) | Dataset → Collator → Trainer data flow |
| [Evaluation Dataset Lance Layout Design](./design-eval-lance-layout.md) | Design decisions for evaluation dataset Lance directory structure |
| [Runtime Transform Audit](./runtime-audit.md) | Side-by-side audit of raw storage samples and runtime dataset outputs for train/eval families |

Runtime surface rules, approved canonical diffs, and upstream issue coverage for `MMEB-V2 eval` are documented centrally in [Format Specifications](./format-spec.md) instead of being repeated across multiple design pages.
The retained scope of training conversion tooling is also documented centrally in [Format Specifications](./format-spec.md) and [Custom Dataset Development Guide](./guide-custom-dataset.md).

---

## Training Datasets

25 training subsets across 4 dataset families, using 4 different Lance schemas. Migrated training Datasets output `TrainSample`.

See [Training Datasets Overview](./train/index.md) for details.

| Family | Registration Name | Subsets | Storage Boundary | Lance Schema |
|--------|-------------------|---------|------------------|--------------|
| [MMEB](./train/mmeb/index.md) | `mmeb_train` | 20 | raw-preserving | sample table + image side table |
| [ViDoRe](./train/vidore/index.md) | `vidore_train` | 1 | raw-preserving | Single table (image struct embedded) |
| [VisRAG](./train/visrag/index.md) | `visrag_train` | 1 | raw-preserving | Single table (source-specific prompt) |
| [LLaVA-Hound](./train/llavahound/index.md) | `llavahound_train` | 3 | rewritten | Dual-table (frames + instructions) |

### MMEB Training Subsets (20, raw-preserving conversion)

| Subset | Config Name | Task Type | Weight | Max Samples |
|--------|-------------|-----------|--------|-------------|
| [ImageNet-1K](./train/mmeb/imagenet-1k.md) | ImageNet_1K | Classification | 1 | 100,000 |
| [N24News](./train/mmeb/n24news.md) | N24News | Classification | 1 | 50,000 |
| [HatefulMemes](./train/mmeb/hateful-memes.md) | HatefulMemes | Classification | 0.5 | 10,000 |
| [VOC2007](./train/mmeb/voc2007.md) | VOC2007 | Classification | 0.5 | 10,000 |
| [SUN397](./train/mmeb/sun397.md) | SUN397 | Classification | 0.5 | 20,000 |
| [OK-VQA](./train/mmeb/ok-vqa.md) | OK-VQA | VQA | 0.5 | 10,000 |
| [A-OKVQA](./train/mmeb/a-okvqa.md) | A-OKVQA | VQA | 0.5 | 20,000 |
| [DocVQA](./train/mmeb/docvqa.md) | DocVQA | VQA | 1 | 40,000 |
| [InfographicsVQA](./train/mmeb/infographicsvqa.md) | InfographicsVQA | VQA | 0.5 | 25,000 |
| [ChartQA](./train/mmeb/chartqa.md) | ChartQA | VQA | 0.5 | 28,000 |
| [Visual7W](./train/mmeb/visual7w.md) | Visual7W | VQA | 1 | 70,000 |
| [VisDial](./train/mmeb/visdial.md) | VisDial | Retrieval (T→I) | 1 | 130,000 |
| [CIRR](./train/mmeb/cirr.md) | CIRR | Retrieval (I+T→I) | 0.5 | 30,000 |
| [VisualNews T2I](./train/mmeb/visualnews-t2i.md) | VisualNews_t2i | Retrieval (T→I) | 1 | 100,000 |
| [VisualNews I2T](./train/mmeb/visualnews-i2t.md) | VisualNews_i2t | Retrieval (I→T) | 1 | 100,000 |
| [MSCOCO T2I](./train/mmeb/mscoco-t2i.md) | MSCOCO_t2i | Retrieval (T→I) | 1 | 100,000 |
| [MSCOCO I2T](./train/mmeb/mscoco-i2t.md) | MSCOCO_i2t | Retrieval (I→T) | 1 | 120,000 |
| [NIGHTS](./train/mmeb/nights.md) | NIGHTS | Retrieval (I→I) | 0.5 | 20,000 |
| [WebQA](./train/mmeb/webqa.md) | WebQA | Retrieval (T→I+T) | 0.5 | 20,000 |
| [MSCOCO VG](./train/mmeb/mscoco.md) | MSCOCO | Visual Grounding | 1 | 100,000 |

### Other Training Subsets (5, two-layer provenance)

| Subset | Family | Config Name | Weight |
|--------|--------|-------------|--------|
| [ColPali Train Set](./train/vidore/colpali-train-set.md) | ViDoRe | colpali_train_set | 10 |
| [VisRAG In-domain](./train/visrag/visrag-indomain.md) | VisRAG | visrag-indomain | 12 |
| [Video Caption 300K](./train/llavahound/video-caption-300k.md) | LLaVA-Hound | video_caption_300k | 5 |
| [Video Caption 300K (Video)](./train/llavahound/video-caption-300k-video.md) | LLaVA-Hound | video_caption_300k-video | 5 |
| [Video QA 240K](./train/llavahound/video-qa-240k.md) | LLaVA-Hound | video_qa_240k | 5 |

---

## Evaluation Datasets

`MMEB-V2 eval` contains 78 evaluation subsets across 3 modalities (image / video / visdoc) and uses a fixed Lance three-file layout (`queries / candidates / qrels`).

Other evaluation datasets in this project do not automatically inherit this `strict / canonical + three-table` storage contract. By default they should keep original content and schema as much as possible, with adaptation handled by the unified retrieval runtime.

See [Evaluation Datasets Overview](./eval/index.md) for details.

### Image Tasks (36, metric: hit@1)

See [Image Evaluation Datasets](./eval/image/index.md) for details.

#### Classification (10) — task_type: `image_classification`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [ImageNet-1K](./eval/image/classification/imagenet-1k.md) | 3 layers | local |
| [N24News](./eval/image/classification/n24news.md) | 3 layers | local |
| [HatefulMemes](./eval/image/classification/hateful-memes.md) | 3 layers | local |
| [VOC2007](./eval/image/classification/voc2007.md) | 3 layers | local |
| [SUN397](./eval/image/classification/sun397.md) | 3 layers | local |
| [Place365](./eval/image/classification/place365.md) | 3 layers | local |
| [ImageNet-A](./eval/image/classification/imagenet-a.md) | 3 layers | local |
| [ImageNet-R](./eval/image/classification/imagenet-r.md) | 3 layers | local |
| [ObjectNet](./eval/image/classification/objectnet.md) | 3 layers | local |
| [Country211](./eval/image/classification/country211.md) | 3 layers | local |

#### Visual QA (10) — task_type: `image_question_answer`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [OK-VQA](./eval/image/qa/ok-vqa.md) | 3 layers | local |
| [A-OKVQA](./eval/image/qa/a-okvqa.md) | 3 layers | local |
| [DocVQA](./eval/image/qa/docvqa.md) | 3 layers | local |
| [InfographicsVQA](./eval/image/qa/infographicsvqa.md) | 3 layers | local |
| [ChartQA](./eval/image/qa/chartqa.md) | 3 layers | local |
| [Visual7W](./eval/image/qa/visual7w.md) | 3 layers | local |
| [ScienceQA](./eval/image/qa/scienceqa.md) | 3 layers | local |
| [VizWiz](./eval/image/qa/vizwiz.md) | 3 layers | local |
| [GQA](./eval/image/qa/gqa.md) | 3 layers | local |
| [TextVQA](./eval/image/qa/textvqa.md) | 3 layers | local |

#### Retrieval (12) — task_type: `image_retrieval`

| Subset | Direction | Provenance Layers | Evaluation Mode |
|--------|-----------|-------------------|-----------------|
| [MSCOCO I2T](./eval/image/retrieval/mscoco-i2t.md) | I→T | 3 layers | local |
| [VisualNews I2T](./eval/image/retrieval/visualnews-i2t.md) | I→T | 3 layers | local |
| [VisDial](./eval/image/retrieval/visdial.md) | T→I | 3 layers | local |
| [MSCOCO T2I](./eval/image/retrieval/mscoco-t2i.md) | T→I | 3 layers | local |
| [VisualNews T2I](./eval/image/retrieval/visualnews-t2i.md) | T→I | 3 layers | local |
| [WebQA](./eval/image/retrieval/webqa.md) | T→I+T | 3 layers | local |
| [EDIS](./eval/image/retrieval/edis.md) | T→I+T | 3 layers | local |
| [Wiki-SS-NQ](./eval/image/retrieval/wiki-ss-nq.md) | T→I | 3 layers | local |
| [CIRR](./eval/image/retrieval/cirr.md) | I+T→I | 3 layers | local |
| [NIGHTS](./eval/image/retrieval/nights.md) | I→I | 3 layers | local |
| [OVEN](./eval/image/retrieval/oven.md) | I+T→I+T | 3 layers | local |
| [FashionIQ](./eval/image/retrieval/fashioniq.md) | I+T→I | 3 layers | local |

#### Visual Grounding (4) — task_type: `image_visual_grounding`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [MSCOCO](./eval/image/visual-grounding/mscoco.md) | 3 layers | local |
| [RefCOCO](./eval/image/visual-grounding/refcoco.md) | 3 layers | local |
| [RefCOCO-Matching](./eval/image/visual-grounding/refcoco-matching.md) | 3 layers | local |
| [Visual7W-Pointing](./eval/image/visual-grounding/visual7w-pointing.md) | 3 layers | local |

### Video Tasks (18, metric: hit@1)

See [Video Evaluation Datasets](./eval/video/index.md) for details.

#### Classification (5) — task_type: `video_classification`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [K700](./eval/video/classification/k700.md) | 3 layers | global |
| [UCF101](./eval/video/classification/ucf101.md) | 3 layers | global |
| [HMDB51](./eval/video/classification/hmdb51.md) | 3 layers | global |
| [SmthSmthV2](./eval/video/classification/smthsmthv2.md) | 3 layers | local |
| [Breakfast](./eval/video/classification/breakfast.md) | 3 layers | global |

#### Video QA (5) — task_type: `video_question_answer`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [Video-MME](./eval/video/qa/video-mme.md) | 3 layers | local |
| [MVBench](./eval/video/qa/mvbench.md) | 3 layers | local |
| [NExTQA](./eval/video/qa/nextqa.md) | 3 layers | local |
| [EgoSchema](./eval/video/qa/egoschema.md) | 3 layers | local |
| [ActivityNetQA](./eval/video/qa/activitynetqa.md) | 3 layers | local |

#### Video Retrieval (5) — task_type: `video_retrieval`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [MSR-VTT](./eval/video/retrieval/msr-vtt.md) | 3 layers | global |
| [MSVD](./eval/video/retrieval/msvd.md) | 3 layers | global |
| [DiDeMo](./eval/video/retrieval/didemo.md) | 3 layers | global |
| [VATEX](./eval/video/retrieval/vatex.md) | 3 layers | global |
| [YouCook2](./eval/video/retrieval/youcook2.md) | 3 layers | global |

#### Moment Retrieval (3) — task_type: `video_moment_retrieval`

| Subset | Provenance Layers | Evaluation Mode |
|--------|-------------------|-----------------|
| [QVHighlight](./eval/video/moment-retrieval/qvhighlight.md) | 3 layers | local |
| [Charades-STA](./eval/video/moment-retrieval/charades-sta.md) | 3 layers | local |
| [MomentSeeker](./eval/video/moment-retrieval/momentseeker.md) | 3 layers | local |

### VisDoc Tasks (24, metric: ndcg@5)

See [VisDoc Evaluation Datasets](./eval/visdoc/index.md) for details.

#### ViDoRe V1 (10) — task_type: `visdoc_vidore_v1`

| Subset | Provenance Layers |
|--------|-------------------|
| [ArxivQA](./eval/visdoc/vidore-v1/arxivqa.md) | 2 layers |
| [DocVQA](./eval/visdoc/vidore-v1/docvqa.md) | 2 layers |
| [InfoVQA](./eval/visdoc/vidore-v1/infovqa.md) | 2 layers |
| [TabFQuAD](./eval/visdoc/vidore-v1/tabfquad.md) | 2 layers |
| [TATDQA](./eval/visdoc/vidore-v1/tatdqa.md) | 2 layers |
| [ShiftProject](./eval/visdoc/vidore-v1/shiftproject.md) | 2 layers |
| [SyntheticDocQA AI](./eval/visdoc/vidore-v1/syntheticdocqa-ai.md) | 2 layers |
| [SyntheticDocQA Energy](./eval/visdoc/vidore-v1/syntheticdocqa-energy.md) | 2 layers |
| [SyntheticDocQA Gov](./eval/visdoc/vidore-v1/syntheticdocqa-gov.md) | 2 layers |
| [SyntheticDocQA Healthcare](./eval/visdoc/vidore-v1/syntheticdocqa-healthcare.md) | 2 layers |

#### ViDoRe V2 (4) — task_type: `visdoc_vidore_v2`

| Subset | Provenance Layers |
|--------|-------------------|
| [ESG Reports Human Labeled V2](./eval/visdoc/vidore-v2/esg-reports-human-labeled-v2.md) | 2 layers |
| [Biomedical Lectures V2 Multilingual](./eval/visdoc/vidore-v2/biomedical-lectures-v2-multilingual.md) | 2 layers |
| [Economics Reports V2 Multilingual](./eval/visdoc/vidore-v2/economics-reports-v2-multilingual.md) | 2 layers |
| [ESG Reports V2 Multilingual](./eval/visdoc/vidore-v2/esg-reports-v2-multilingual.md) | 2 layers |

#### VisRAG (6) — task_type: `visdoc_visrag`

| Subset | Provenance Layers |
|--------|-------------------|
| [VisRAG ArxivQA](./eval/visdoc/visrag/arxivqa.md) | 2 layers |
| [VisRAG ChartQA](./eval/visdoc/visrag/chartqa.md) | 2 layers |
| [VisRAG MP-DocVQA](./eval/visdoc/visrag/mp-docvqa.md) | 2 layers |
| [VisRAG SlideVQA](./eval/visdoc/visrag/slidevqa.md) | 2 layers |
| [VisRAG InfoVQA](./eval/visdoc/visrag/infovqa.md) | 2 layers |
| [VisRAG PlotQA](./eval/visdoc/visrag/plotqa.md) | 2 layers |

#### VisDoc OOD (4) — task_type: `visdoc_ood`

| Subset | Provenance Layers |
|--------|-------------------|
| [ViDoSeek-page](./eval/visdoc/ood/vidoseek-page.md) | 2 layers |
| [ViDoSeek-doc](./eval/visdoc/ood/vidoseek-doc.md) | 2 layers |
| [MMLongBench-page](./eval/visdoc/ood/mmlongbench-page.md) | 2 layers |
| [MMLongBench-doc](./eval/visdoc/ood/mmlongbench-doc.md) | 2 layers |
