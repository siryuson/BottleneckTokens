# 数据集总览

本页面是 BToks 所有数据集文档的统一入口。

项目级文档统一强调 `source / conversion / runtime` 三层边界；`strict / canonical` 仅用于 `MMEB-V2 eval` 这一特殊评测数据集，不推广到其它训练或评测数据集。

## 快速链接

| 文档 | 说明 |
|------|------|
| [格式规范](./format-spec.md) | 训练/评测数据集格式标准、runtime surface 规则、MMEB-V2 strict/canonical 例外说明、接入检查清单 |
| [数据集接入指南](./guide-custom-dataset.md) | 项目级接入准则、训练/评测接入流程、runtime transform 边界 |
| [数据处理流水线](./architecture-data-pipeline.md) | Dataset → Collator → Trainer 数据流 |
| [评测数据集 Lance 布局设计](./design-eval-lance-layout.md) | 评测数据集的 Lance 目录结构设计决策 |
| [Runtime Transform 审计](./runtime-audit.md) | 训练/评测数据族的 raw 存储样本与 runtime 输出样本对照审计 |

关于 runtime surface 规则、`MMEB-V2 eval` 已批准的 canonical 差异和上游问题覆盖状态，统一记录在 [格式规范](./format-spec.md) 中，不再分散到多个设计页面。
训练数据转换工具的保留范围和使用边界，也统一记录在 [格式规范](./format-spec.md) 与 [自定义数据集开发指南](./guide-custom-dataset.md) 中。

---

## 训练数据集

共 25 个训练子集，分属 4 个数据集家族，使用 4 种不同的 Lance schema；已迁移训练 Dataset 统一输出 `TrainSample` 格式。

详见 [训练数据集总览](./train/index.md)。

| 家族 | 注册名 | 子集数 | 存储边界 | Lance Schema |
|------|--------|--------|---------|-------------|
| [MMEB](./train/mmeb/index.md) | `mmeb_train` | 20 | raw-preserving | sample 表 + image side table |
| [ViDoRe](./train/vidore/index.md) | `vidore_train` | 1 | raw-preserving | 单表（image struct 内嵌） |
| [VisRAG](./train/visrag/index.md) | `visrag_train` | 1 | raw-preserving | 单表（source-specific prompt） |
| [LLaVA-Hound](./train/llavahound/index.md) | `llavahound_train` | 3 | rewritten | 双表（帧表 + 指令表） |

### MMEB 训练子集（20 个，raw-preserving conversion）

| 子集 | 配置名 | 任务类型 | 权重 | 最大样本数 |
|------|--------|---------|------|-----------|
| [ImageNet-1K](./train/mmeb/imagenet-1k.md) | ImageNet_1K | 分类 | 1 | 100,000 |
| [N24News](./train/mmeb/n24news.md) | N24News | 分类 | 1 | 50,000 |
| [HatefulMemes](./train/mmeb/hateful-memes.md) | HatefulMemes | 分类 | 0.5 | 10,000 |
| [VOC2007](./train/mmeb/voc2007.md) | VOC2007 | 分类 | 0.5 | 10,000 |
| [SUN397](./train/mmeb/sun397.md) | SUN397 | 分类 | 0.5 | 20,000 |
| [OK-VQA](./train/mmeb/ok-vqa.md) | OK-VQA | VQA | 0.5 | 10,000 |
| [A-OKVQA](./train/mmeb/a-okvqa.md) | A-OKVQA | VQA | 0.5 | 20,000 |
| [DocVQA](./train/mmeb/docvqa.md) | DocVQA | VQA | 1 | 40,000 |
| [InfographicsVQA](./train/mmeb/infographicsvqa.md) | InfographicsVQA | VQA | 0.5 | 25,000 |
| [ChartQA](./train/mmeb/chartqa.md) | ChartQA | VQA | 0.5 | 28,000 |
| [Visual7W](./train/mmeb/visual7w.md) | Visual7W | VQA | 1 | 70,000 |
| [VisDial](./train/mmeb/visdial.md) | VisDial | 检索 (T→I) | 1 | 130,000 |
| [CIRR](./train/mmeb/cirr.md) | CIRR | 检索 (I+T→I) | 0.5 | 30,000 |
| [VisualNews T2I](./train/mmeb/visualnews-t2i.md) | VisualNews_t2i | 检索 (T→I) | 1 | 100,000 |
| [VisualNews I2T](./train/mmeb/visualnews-i2t.md) | VisualNews_i2t | 检索 (I→T) | 1 | 100,000 |
| [MSCOCO T2I](./train/mmeb/mscoco-t2i.md) | MSCOCO_t2i | 检索 (T→I) | 1 | 100,000 |
| [MSCOCO I2T](./train/mmeb/mscoco-i2t.md) | MSCOCO_i2t | 检索 (I→T) | 1 | 120,000 |
| [NIGHTS](./train/mmeb/nights.md) | NIGHTS | 检索 (I→I) | 0.5 | 20,000 |
| [WebQA](./train/mmeb/webqa.md) | WebQA | 检索 (T→I+T) | 0.5 | 20,000 |
| [MSCOCO VG](./train/mmeb/mscoco.md) | MSCOCO | 视觉定位 | 1 | 100,000 |

### 其他训练子集（5 个，两层溯源）

| 子集 | 家族 | 配置名 | 权重 |
|------|------|--------|------|
| [ColPali Train Set](./train/vidore/colpali-train-set.md) | ViDoRe | colpali_train_set | 10 |
| [VisRAG In-domain](./train/visrag/visrag-indomain.md) | VisRAG | visrag-indomain | 12 |
| [Video Caption 300K](./train/llavahound/video-caption-300k.md) | LLaVA-Hound | video_caption_300k | 5 |
| [Video Caption 300K (Video)](./train/llavahound/video-caption-300k-video.md) | LLaVA-Hound | video_caption_300k-video | 5 |
| [Video QA 240K](./train/llavahound/video-qa-240k.md) | LLaVA-Hound | video_qa_240k | 5 |

---

## 评测数据集

`MMEB-V2 eval` 共 78 个评测子集，覆盖 3 大模态（image / video / visdoc），并使用固定的 Lance 三文件布局（`queries / candidates / qrels`）。

项目中的其它评测数据集不自动继承这套 `strict / canonical + 三表` 存储语义；默认仍应尽量保持原始内容与原 schema，再由统一 retrieval runtime 适配。

详见 [评测数据集总览](./eval/index.md)。

### Image Tasks（36 个，指标：hit@1）

详见 [Image 评测数据集](./eval/image/index.md)。

#### Classification（10 个）— task_type: `image_classification`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [ImageNet-1K](./eval/image/classification/imagenet-1k.md) | 3 层 | local |
| [N24News](./eval/image/classification/n24news.md) | 3 层 | local |
| [HatefulMemes](./eval/image/classification/hateful-memes.md) | 3 层 | local |
| [VOC2007](./eval/image/classification/voc2007.md) | 3 层 | local |
| [SUN397](./eval/image/classification/sun397.md) | 3 层 | local |
| [Place365](./eval/image/classification/place365.md) | 3 层 | local |
| [ImageNet-A](./eval/image/classification/imagenet-a.md) | 3 层 | local |
| [ImageNet-R](./eval/image/classification/imagenet-r.md) | 3 层 | local |
| [ObjectNet](./eval/image/classification/objectnet.md) | 3 层 | local |
| [Country211](./eval/image/classification/country211.md) | 3 层 | local |

#### Visual QA（10 个）— task_type: `image_question_answer`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [OK-VQA](./eval/image/qa/ok-vqa.md) | 3 层 | local |
| [A-OKVQA](./eval/image/qa/a-okvqa.md) | 3 层 | local |
| [DocVQA](./eval/image/qa/docvqa.md) | 3 层 | local |
| [InfographicsVQA](./eval/image/qa/infographicsvqa.md) | 3 层 | local |
| [ChartQA](./eval/image/qa/chartqa.md) | 3 层 | local |
| [Visual7W](./eval/image/qa/visual7w.md) | 3 层 | local |
| [ScienceQA](./eval/image/qa/scienceqa.md) | 3 层 | local |
| [VizWiz](./eval/image/qa/vizwiz.md) | 3 层 | local |
| [GQA](./eval/image/qa/gqa.md) | 3 层 | local |
| [TextVQA](./eval/image/qa/textvqa.md) | 3 层 | local |

#### Retrieval（12 个）— task_type: `image_retrieval`

| 子集 | 方向 | 溯源层数 | 评测模式 |
|------|------|---------|---------|
| [MSCOCO I2T](./eval/image/retrieval/mscoco-i2t.md) | I→T | 3 层 | local |
| [VisualNews I2T](./eval/image/retrieval/visualnews-i2t.md) | I→T | 3 层 | local |
| [VisDial](./eval/image/retrieval/visdial.md) | T→I | 3 层 | local |
| [MSCOCO T2I](./eval/image/retrieval/mscoco-t2i.md) | T→I | 3 层 | local |
| [VisualNews T2I](./eval/image/retrieval/visualnews-t2i.md) | T→I | 3 层 | local |
| [WebQA](./eval/image/retrieval/webqa.md) | T→I+T | 3 层 | local |
| [EDIS](./eval/image/retrieval/edis.md) | T→I+T | 3 层 | local |
| [Wiki-SS-NQ](./eval/image/retrieval/wiki-ss-nq.md) | T→I | 3 层 | local |
| [CIRR](./eval/image/retrieval/cirr.md) | I+T→I | 3 层 | local |
| [NIGHTS](./eval/image/retrieval/nights.md) | I→I | 3 层 | local |
| [OVEN](./eval/image/retrieval/oven.md) | I+T→I+T | 3 层 | local |
| [FashionIQ](./eval/image/retrieval/fashioniq.md) | I+T→I | 3 层 | local |

#### Visual Grounding（4 个）— task_type: `image_visual_grounding`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [MSCOCO](./eval/image/visual-grounding/mscoco.md) | 3 层 | local |
| [RefCOCO](./eval/image/visual-grounding/refcoco.md) | 3 层 | local |
| [RefCOCO-Matching](./eval/image/visual-grounding/refcoco-matching.md) | 3 层 | local |
| [Visual7W-Pointing](./eval/image/visual-grounding/visual7w-pointing.md) | 3 层 | local |

### Video Tasks（18 个，指标：hit@1）

详见 [Video 评测数据集](./eval/video/index.md)。

#### Classification（5 个）— task_type: `video_classification`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [K700](./eval/video/classification/k700.md) | 3 层 | global |
| [UCF101](./eval/video/classification/ucf101.md) | 3 层 | global |
| [HMDB51](./eval/video/classification/hmdb51.md) | 3 层 | global |
| [SmthSmthV2](./eval/video/classification/smthsmthv2.md) | 3 层 | local |
| [Breakfast](./eval/video/classification/breakfast.md) | 3 层 | global |

#### Video QA（5 个）— task_type: `video_question_answer`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [Video-MME](./eval/video/qa/video-mme.md) | 3 层 | local |
| [MVBench](./eval/video/qa/mvbench.md) | 3 层 | local |
| [NExTQA](./eval/video/qa/nextqa.md) | 3 层 | local |
| [EgoSchema](./eval/video/qa/egoschema.md) | 3 层 | local |
| [ActivityNetQA](./eval/video/qa/activitynetqa.md) | 3 层 | local |

#### Video Retrieval（5 个）— task_type: `video_retrieval`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [MSR-VTT](./eval/video/retrieval/msr-vtt.md) | 3 层 | global |
| [MSVD](./eval/video/retrieval/msvd.md) | 3 层 | global |
| [DiDeMo](./eval/video/retrieval/didemo.md) | 3 层 | global |
| [VATEX](./eval/video/retrieval/vatex.md) | 3 层 | global |
| [YouCook2](./eval/video/retrieval/youcook2.md) | 3 层 | global |

#### Moment Retrieval（3 个）— task_type: `video_moment_retrieval`

| 子集 | 溯源层数 | 评测模式 |
|------|---------|---------|
| [QVHighlight](./eval/video/moment-retrieval/qvhighlight.md) | 3 层 | local |
| [Charades-STA](./eval/video/moment-retrieval/charades-sta.md) | 3 层 | local |
| [MomentSeeker](./eval/video/moment-retrieval/momentseeker.md) | 3 层 | local |

### VisDoc Tasks（24 个，指标：ndcg@5）

详见 [VisDoc 评测数据集](./eval/visdoc/index.md)。

#### ViDoRe V1（10 个）— task_type: `visdoc_vidore_v1`

| 子集 | 溯源层数 |
|------|---------|
| [ArxivQA](./eval/visdoc/vidore-v1/arxivqa.md) | 2 层 |
| [DocVQA](./eval/visdoc/vidore-v1/docvqa.md) | 2 层 |
| [InfoVQA](./eval/visdoc/vidore-v1/infovqa.md) | 2 层 |
| [TabFQuAD](./eval/visdoc/vidore-v1/tabfquad.md) | 2 层 |
| [TATDQA](./eval/visdoc/vidore-v1/tatdqa.md) | 2 层 |
| [ShiftProject](./eval/visdoc/vidore-v1/shiftproject.md) | 2 层 |
| [SyntheticDocQA AI](./eval/visdoc/vidore-v1/syntheticdocqa-ai.md) | 2 层 |
| [SyntheticDocQA Energy](./eval/visdoc/vidore-v1/syntheticdocqa-energy.md) | 2 层 |
| [SyntheticDocQA Gov](./eval/visdoc/vidore-v1/syntheticdocqa-gov.md) | 2 层 |
| [SyntheticDocQA Healthcare](./eval/visdoc/vidore-v1/syntheticdocqa-healthcare.md) | 2 层 |

#### ViDoRe V2（4 个）— task_type: `visdoc_vidore_v2`

| 子集 | 溯源层数 |
|------|---------|
| [ESG Reports Human Labeled V2](./eval/visdoc/vidore-v2/esg-reports-human-labeled-v2.md) | 2 层 |
| [Biomedical Lectures V2 Multilingual](./eval/visdoc/vidore-v2/biomedical-lectures-v2-multilingual.md) | 2 层 |
| [Economics Reports V2 Multilingual](./eval/visdoc/vidore-v2/economics-reports-v2-multilingual.md) | 2 层 |
| [ESG Reports V2 Multilingual](./eval/visdoc/vidore-v2/esg-reports-v2-multilingual.md) | 2 层 |

#### VisRAG（6 个）— task_type: `visdoc_visrag`

| 子集 | 溯源层数 |
|------|---------|
| [VisRAG ArxivQA](./eval/visdoc/visrag/arxivqa.md) | 2 层 |
| [VisRAG ChartQA](./eval/visdoc/visrag/chartqa.md) | 2 层 |
| [VisRAG MP-DocVQA](./eval/visdoc/visrag/mp-docvqa.md) | 2 层 |
| [VisRAG SlideVQA](./eval/visdoc/visrag/slidevqa.md) | 2 层 |
| [VisRAG InfoVQA](./eval/visdoc/visrag/infovqa.md) | 2 层 |
| [VisRAG PlotQA](./eval/visdoc/visrag/plotqa.md) | 2 层 |

#### VisDoc OOD（4 个）— task_type: `visdoc_ood`

| 子集 | 溯源层数 |
|------|---------|
| [ViDoSeek-page](./eval/visdoc/ood/vidoseek-page.md) | 2 层 |
| [ViDoSeek-doc](./eval/visdoc/ood/vidoseek-doc.md) | 2 层 |
| [MMLongBench-page](./eval/visdoc/ood/mmlongbench-page.md) | 2 层 |
| [MMLongBench-doc](./eval/visdoc/ood/mmlongbench-doc.md) | 2 层 |
