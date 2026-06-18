# MMEB-V2 训练源调研

本页只记录 MMEB-V2 78 个评测子集是否存在可用于训练的对应来源，以及后续单数据集 rewrite 前需要审计什么。它不执行下载，不新增 conversion/runtime，也不修改默认训练配置。

## 结论摘要

- MMEB-V2 评测清单共有 78 个子集：image 36 个、video 18 个、visdoc 24 个。
- 不是每个评测子集都有一一对应的公开训练集。`TIGER-Lab/MMEB-V2` 是 eval frames / parquet 入口，`TIGER-Lab/MMEB_Raw_Video` 是 raw video 入口，二者都不能当作统一训练源。
- 当前已覆盖的训练来源主要是 `TIGER-Lab/MMEB-train` 的 20 个 image 子集、已实现的独立 image/video runtime、M-BEIR、ViDoRe、VisRAG 和 LLaVA-Hound。`NExTQA`、`YouCook2`、`ActivityNet Captions`、`VizWiz`、`RefCOCO`、`RefCOCO-Matching` 和 `VATEX` 已从本页候选转为正式训练 runtime。
- 第二轮仍需继续审计、补转换产物或保持占位的主要对象是 `ActivityNetQA`、`ViDoSeek-*` 和 `MMLongBench-page`；`K700` frame-unit 重转已经完成并进入 v2 enabled 配置。
- `Video-MME`、`MVBench`、`EgoSchema`、`MomentSeeker` 当前能确认的是 benchmark / test mirror，不应直接进入默认训练配置；其中 `MVBench` / `EgoSchema` 方向可以用 `OpenGVLab/VideoChat2-IT-clean` 的拆分子集作为训练 proxy 候选，而不是训练 benchmark 本体。

## 状态词表

本页只使用 `expand-training-dataset-catalog` 规格中的候选状态。更细的判断，例如“同家族 proxy 训练源”“test mirror”或“没有同名公开 train split”，写在覆盖入口、source kind、split / 媒体线索和后续动作中，不新增状态词。

| 状态 | 含义 |
|------|------|
| `implemented` | 当前仓库已有正式训练入口、独立 runtime 或正式 eval 支持 |
| `covered_by_existing_family` | 已由家族级训练源覆盖，但没有同名训练子集 |
| `design_only` | 只有设计占位，还没有 loader / registry / 本地 Lance 样例 |
| `pending_download` | 尚未本地下载，后续需要下载 |
| `downloaded_pending_audit` | 本地源数据已存在，但尚未完成 schema / split / overlap 审计 |
| `audit_blocked` | 需要进一步确认 annotation、媒体包、provenance 或许可 |
| `train_candidate` | 已看到可审计的 train split 或训练语料入口 |
| `formal_candidate` | 已进入下一批正式推进队列，但仍必须完成 source、license、split、media、overlap 和真实 root smoke 后才能启用 |
| `eval_only` | 当前可见入口是 benchmark、validation、test mirror 或 OOD eval set |
| `pretrain_only` | 更适合大规模预训练，不进入当前微调训练主线 |
| `defer` | 本轮不进入训练扩展 |

| Source kind | 含义 |
|-------------|------|
| `official_upstream` | 官方站点或论文项目发布入口 |
| `hf_official_or_author` | HF 上的官方或作者维护入口 |
| `hf_eval_mirror` | HF 上的评测镜像 |
| `community_mirror` | 社区镜像，需要额外 provenance 审计 |
| `local_only` | 当前仓库或本地审计记录可见，外部入口仍需复核 |
| `community_cleaned_manifest` | 社区清洗或重组的 annotation / manifest，需要额外 provenance、license 和 media policy 审计 |

## 外部证据入口

| 对象 | 入口 | 结论 |
|------|------|------|
| VLM2Vec 官方仓库 | <https://github.com/TIGER-AI-Lab/VLM2Vec> | 官方训练配置主要列出 MMEB-train、ViDoRe、VisRAG 和 LLaVA-Hound |
| 官方训练配置 | <https://raw.githubusercontent.com/TIGER-AI-Lab/VLM2Vec/main/experiments/public/train/train_alltasks.yaml> | 可用于确认 20 个 MMEB-train image 子集和若干家族级训练源 |
| MMEB-V2 eval | <https://huggingface.co/datasets/TIGER-Lab/MMEB-V2> | 评测 frames / parquet，不是训练总入口 |
| MMEB raw video | <https://huggingface.co/datasets/TIGER-Lab/MMEB_Raw_Video> | 视频原始媒体入口，不是训练 annotation |
| VizWiz VQA | <https://vizwiz.org/tasks-and-datasets/vqa/> | 官方 VQA train / validation / test 入口，适合 P0 审计 |
| RefCOCO family | <https://www.tensorflow.org/datasets/catalog/ref_coco> | 官方 split 可见，适合 P0/P1 审计 |
| ObjectNet | <https://objectnet.dev/download.html> | OOD evaluation source，不是默认训练源 |
| Video-MME | <https://huggingface.co/datasets/siyrus/BToks-Video-MME> | test-only eval mirror |
| MVBench | <https://huggingface.co/datasets/siyrus/BToks-MVBench> | benchmark package / eval mirror |
| EgoSchema | <https://huggingface.co/datasets/siyrus/BToks-EgoSchema> | test / subset-test eval mirror |
| MomentSeeker | <https://huggingface.co/datasets/siyrus/BToks-MomentSeeker> | test-only eval mirror |
| OpenGVLab/VideoChat2-IT | <https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT> | gated annotation source，官方说明约 1.9M JSON annotations；只提供 JSON，媒体需按源数据说明另取 |
| VideoChat2-IT-clean | <https://huggingface.co/datasets/byminji/VideoChat2-IT-clean> | 社区视频子集清洗 manifest；本地实测 16 个 `train.json` 合计 877,489 条样本，和上游逐子集表格一致，但上游 README 总数字段写成 874,869；其中 media-backed 子集已通过 `videochat2_it_train` 正式接入 195,584 条训练样本，剩余子集继续按源数据补审 license / provenance / media |
| VATEX | <https://huggingface.co/datasets/siyrus/BToks-VATEX>、<https://huggingface.co/datasets/lmms-lab/VATEX> | eval 入口仍以 test 为主；训练侧使用本地官方 VATEX train JSON 和视频归档，`vatex_train` 已完成正式 `data/frames.lance` 根迁移并进入 v2 |
| RefCOCO HF media mirror | <https://huggingface.co/datasets/xche32/refcpco> | 抽样确认保留 `refcoco/images/train2014/COCO_train2014_...` key；应先下载 `refcoco.tar.gz` 到本地，再作为 conversion 媒体源 |
| ViDoSeek page | <https://huggingface.co/datasets/siyrus/BToks-ViDoSeek-page-fixed> | corpus / queries 为 test，qrels train 不能单独构成完整训练源 |
| MMLongBench page | <https://huggingface.co/datasets/siyrus/BToks-MMLongBench-page-fixed> | test-only eval mirror |

## Image 子集

| 子集 | task_type | 状态 | 覆盖或下载入口 | Source kind | split / 媒体线索 | 后续动作 |
|------|-----------|------|----------------|-------------|-------------------|----------|
| `VOC2007` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `N24News` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `SUN397` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `ImageNet-1K` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `HatefulMemes` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `OK-VQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `A-OKVQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `DocVQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `InfographicsVQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `ChartQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `Visual7W` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `VisDial` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `CIRR` | composed_image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `VisualNews_t2i` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `VisualNews_i2t` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `MSCOCO_t2i` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `MSCOCO_i2t` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `NIGHTS` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `WebQA` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `MSCOCO` | image_captioning | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split，image side table | 已覆盖，跳过 |
| `ImageNet-A` | image_classification | `audit_blocked` | ImageNet-A / ImageNet-1K proxy | `official_upstream` | OOD eval set；可考虑 ImageNet-1K 训练 proxy | P2，仅作为 proxy 方向讨论 |
| `ImageNet-R` | image_classification | `audit_blocked` | ImageNet-R / ImageNet-1K proxy | `official_upstream` | OOD eval set；可考虑 ImageNet-1K 训练 proxy | P2，仅作为 proxy 方向讨论 |
| `ObjectNet` | image_classification | `eval_only` | ObjectNet official download | `official_upstream` | OOD evaluation source，license / access 需确认 | P2，不进默认训练 |
| `Country211` | image_classification | `implemented` | `country211_train` | `hf_official_or_author` | default train split excludes eval boundary | 已覆盖，跳过 |
| `Place365` | image_classification | `implemented` | `place365_train` | `official_upstream` | official train split | 已覆盖，跳过 |
| `ScienceQA` | visual_question_answering | `implemented` | `scienceqa_train` | `hf_official_or_author` | official train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `GQA` | visual_question_answering | `implemented` | `gqa_train` | `official_upstream` | official balanced train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `TextVQA` | visual_question_answering | `implemented` | `textvqa_train` | `official_upstream` | official train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `VizWiz` | visual_question_answering | `implemented` | `vizwiz_train` / VizWiz VQA official | `official_upstream` | `official_train_answerable_without_mmeb_v2_eval` 为 14,012 行，图片表为 24,842 行 | 已覆盖，跳过 |
| `Wiki-SS-NQ` | image_retrieval | `implemented` | `wikissnq_train` | `hf_official_or_author` | official train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `FashionIQ` | composed_image_retrieval | `implemented` | `mbeir_train` / `FashionIQ_train` | `hf_official_or_author` | M-BEIR task 7 | 已覆盖，跳过 |
| `OVEN` | image_retrieval | `implemented` | `mbeir_train` / OVEN train views | `hf_official_or_author` | M-BEIR task 6 / 8 | 已覆盖，跳过 |
| `EDIS` | image_retrieval | `implemented` | `mbeir_train` / `EDIS_train` | `hf_official_or_author` | M-BEIR task 2 | 已覆盖，跳过 |
| `Visual7W-Pointing` | visual_grounding | `implemented` | `visual7w_pointing_train` | `official_upstream` | official train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `RefCOCO` | visual_grounding | `implemented` | `refcoco_train` | `hf_metadata_plus_local_refcoco_tar_media` | train / validation / testA / testB；默认训练 split 38,181 行 | 已覆盖，跳过 |
| `RefCOCO-Matching` | image_retrieval | `implemented` | `refcoco_train` matching view | `hf_metadata_plus_local_refcoco_tar_media` | 与 `RefCOCO` 共用默认 split 和 image side table；query 和 positive 侧都保留 referring expression prompt | 已覆盖，进入 v2 enabled 配置 |

## Video 子集

| 子集 | task_type | 状态 | 覆盖或下载入口 | Source kind | split / 媒体线索 | 后续动作 |
|------|-----------|------|----------------|-------------|-------------------|----------|
| `K700` | video_classification | `implemented` | `kinetics700_train` | `official_upstream` | invalid-filtered official train；正式根已有 `data/frames.lance` 和 `data/videos.lance`，默认 split 为 536,489 行 | 已完成 frame-unit 重转、正式 root 替换和取样 smoke，并进入 v2 enabled 配置 |
| `UCF101` | video_classification | `implemented` | `ucf101_train` | `official_upstream` | official train splits | 已覆盖，跳过 |
| `HMDB51` | video_classification | `implemented` | `hmdb51_train` | `official_upstream` | official train splits | 已覆盖，跳过 |
| `SmthSmthV2` | video_classification | `implemented` | `ssv2_train` | `official_upstream` | official train / validation | 已覆盖，跳过 |
| `Breakfast` | video_classification | `implemented` | `breakfast_train` | `official_upstream` | archive train, non-`cam01` filter | 已覆盖，跳过 |
| `Video-MME` | video_question_answering | `eval_only` | `siyrus/BToks-Video-MME` | `hf_eval_mirror` | test only，约 2700 rows | P2，只能另找 proxy |
| `MVBench` | video_question_answering | `eval_only` | `siyrus/BToks-MVBench`；proxy 候选为 `OpenGVLab/VideoChat2-IT-clean` 的 `clevrer_*`、`next_qa`、`ssv2`、`k710` 等拆分子集 | `hf_eval_mirror` / `community_cleaned_manifest` | benchmark 本体不训练；proxy 子集需逐项排除已接入 runtime 重复，且 `clevrer_*` / `tgif_*` 需额外审计 MVBench eval-family overlap | P0/P1，先处理 proxy 子集，不接 benchmark 本体 |
| `NExTQA` | video_question_answering | `implemented` | `nextqa_train` | `hf_official_or_author` | `official_train_without_mmeb_v2_eval` 为 34,132 行，视频表为 3,870 行 | 已覆盖，跳过 |
| `EgoSchema` | video_question_answering | `eval_only` | `siyrus/BToks-EgoSchema`；proxy 候选为 `OpenGVLab/VideoChat2-IT-clean` 的 `ego_qa` | `hf_eval_mirror` / `community_cleaned_manifest` | EgoSchema 本体是 test / subset-test；`ego_qa` 仅 7,797 条有效样本，不需要接入 Ego4D full，但仍需按 Ego4D clip id 审计 eval-family overlap | P1，按 `ego_qa` 小子集审计 |
| `ActivityNetQA` | video_question_answering | `covered_by_existing_family` | `activitynet_captions_train` | `hf_official_or_author` / `local_only` | 有同家族 caption / segment source，不是同名 QA train；本轮已采用 segment-level retrieval 训练视图 | 已覆盖为同家族训练 proxy，后续只做质量审计 |
| `MSR-VTT` | text_video_retrieval | `implemented` | `msrvtt_train` | `official_upstream` | train 9k without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `MSVD` | text_video_retrieval | `implemented` | `msvd_train` | `official_upstream` | train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `DiDeMo` | text_video_retrieval | `implemented` | `didemo_train` | `official_upstream` | train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `VATEX` | text_video_retrieval | `implemented` | `vatex_train` / official VATEX train JSON and video archive | `official_upstream` / `local_only` | 正式根已迁移为 frame-unit；`official_train_without_mmeb_v2_eval` 为 22,105 行，`data/frames.lance` 为 22,105 行，坏媒体排除 3,886 行 | 已覆盖，进入 v2 enabled 配置 |
| `YouCook2` | text_video_retrieval | `implemented` | `youcook2_train` | `community_mirror` / `local_only` | `official_train_without_mmeb_v2_eval` 为 10,330 行，视频表为 1,332 行 | 已覆盖，跳过 |
| `MomentSeeker` | text_video_retrieval | `eval_only` | `siyrus/BToks-MomentSeeker` | `hf_eval_mirror` | test only，约 1602 rows | P2，不进默认训练 |
| `QVHighlight` | video_moment_retrieval | `implemented` | `qvhighlight_train` | `official_upstream` | official train without MMEB-V2 eval overlap | 已覆盖，跳过 |
| `Charades-STA` | video_moment_retrieval | `implemented` | `charades_sta_train` | `official_upstream` | official train without MMEB-V2 eval overlap | 已覆盖，跳过 |

## VisDoc 子集

| 子集 | task_type | 状态 | 覆盖或下载入口 | Source kind | split / 媒体线索 | 后续动作 |
|------|-----------|------|----------------|-------------|-------------------|----------|
| `ViDoRe_arxivqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_docvqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_infovqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_tabfquad` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_tatdqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_shiftproject` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_syntheticDocQA_artificial_intelligence` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_syntheticDocQA_energy` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_syntheticDocQA_government_reports` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_syntheticDocQA_healthcare_industry` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_esg_reports_human_labeled_v2` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_biomedical_lectures_v2_multilingual` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_economics_reports_v2_multilingual` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoRe_esg_reports_v2_multilingual` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `VisRAG_ArxivQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `VisRAG_ChartQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `VisRAG_MP-DocVQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `VisRAG_SlideVQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `VisRAG_InfoVQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `VisRAG_PlotQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | 已覆盖，跳过 |
| `ViDoSeek-page` | document_retrieval | `eval_only` | `siyrus/BToks-ViDoSeek-page-fixed` | `hf_eval_mirror` | corpus / queries 为 test；qrels train 不能单独构成完整训练源 | P1，确认是否有官方 train corpus / query |
| `ViDoSeek-doc` | document_retrieval | `audit_blocked` | MMEB-V2 local eval assembly / ViDoSeek | `local_only` | 当前只确认 eval assembly | P1，先找官方来源 |
| `MMLongBench-page` | document_retrieval | `eval_only` | `siyrus/BToks-MMLongBench-page-fixed` | `hf_eval_mirror` | test-only eval mirror | P1/P2，确认是否有 page-level train 或仅 eval |
| `MMLongBench-doc` | document_retrieval | `implemented` | `mmlongbench_doc_train` | `hf_official_or_author` / `local_only` | default split 仅 4 条 non-overlap rows | 已覆盖但高风险，后续只做质量审计 |

## 新增候选优先级

| 优先级 | 对象 | 进入条件 | 原因 |
|--------|------|----------|------|
| 下一批正式 P0 | `SNLI-VE`、`MSRVTT-QA`、`MSVD-QA`，以及 `OpenGVLab/VideoChat2-IT-clean` 尚无本地媒体的剩余子集 | 先完成 source / license / split / media 依赖 / MMEB-V2 overlap 审计，再进入 conversion/runtime/config；`VideoChat2-IT` 已按 media-backed 子集拆分接入，剩余子集不得合并成一个无媒体 enabled entry | 任务语义清楚，已有本地源或可复用正式视频 root；`VideoChat2-IT` 已提供一批 `MVBench` / `EgoSchema` 方向训练 proxy |
| 下一批正式 P1 | `ActivityNetQA`、`SOP` / Stanford Online Products、`DeepFashion In-Shop`、`LaSCo` | 先补 annotation、媒体根目录、任务语义、license 和 overlap key；通过真实 root smoke 后再启用 | 价值明确，但 source 依赖、重叠边界或训练语义还需要单独审计 |
| 保留候选 | `mmE5 synthetic` 抽样版、`Kinetics-400`、`TVQA`、`iVQA`、`Diving48`、`FineGym`、`Epic-Kitchens-100`、`RefCOCO+`、`RefCOCOg` | 只作为 v3 / 后续 Stage 1 或专项任务候选，不进入当前正式队列 | 有潜在价值，但当前不应阻塞下一批正式接入 |
| 移出正式候选 | `ViDoSeek-page`、`ViDoSeek-doc`、`MMLongBench-page`、`Video-MME`、`MVBench` benchmark 本体、`EgoSchema` benchmark 本体、`MomentSeeker`、`ImageNet-A`、`ImageNet-R`、`ObjectNet` | 保留为 eval-only、blocked 或 proxy 讨论；不得作为默认训练数据直接接入 | 当前证据指向 test-only / eval mirror / OOD benchmark，不能构成正式训练源；移除的是 benchmark 本体，不是移除 `VideoChat2-IT` 这类 proxy |

## 单数据集审计模板

每个 `train_candidate` 或 P1 条目进入实现前，都需要补齐以下信息：

1. 上游 source、license / usage policy、下载协议和是否允许再分发。
2. raw annotation schema、split 名称、row count、媒体文件数量和缺失媒体处理策略。
3. overlap key：image/video/document id、question id、caption id、segment timestamp 或 bbox id。
4. 默认训练 split 名称，以及如何排除 MMEB-V2 eval overlap。
5. `TrainSample` 语义：classification、VQA、retrieval、moment retrieval、grounding 或 document retrieval。
6. 是否应作为独立 runtime、并入已有 family，或只作为 proxy 训练源。
7. 为什么不能直接使用 MMEB-V2 eval mirror 或 test-only mirror 作为训练数据。
