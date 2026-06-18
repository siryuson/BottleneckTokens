# MMEB-V2 Training Source Survey

This page records whether each of the 78 MMEB-V2 evaluation subsets has a usable training source, and what must be audited before a future single-dataset rewrite. It does not download data, add conversion/runtime code, or change default training configs.

## Summary

- The MMEB-V2 benchmark index contains 78 subsets: 36 image, 18 video, and 24 visdoc subsets.
- Not every evaluation subset has a one-to-one public training set. `TIGER-Lab/MMEB-V2` is an eval frames / parquet entry, and `TIGER-Lab/MMEB_Raw_Video` is a raw-video entry. Neither is a unified training source.
- Current training coverage mainly comes from 20 image subsets in `TIGER-Lab/MMEB-train`, implemented standalone image/video runtimes, M-BEIR, ViDoRe, VisRAG, and LLaVA-Hound. `NExTQA`, `YouCook2`, `ActivityNet Captions`, `VizWiz`, `RefCOCO`, `RefCOCO-Matching`, and `VATEX` have moved from this survey's candidate pool to production training runtimes.
- The remaining second-round objects to audit, regenerate, or keep as placeholders are `ActivityNetQA`, `ViDoSeek-*`, and `MMLongBench-page`; the `K700` frame-unit rerun is complete and enabled in v2.
- `Video-MME`, `MVBench`, `EgoSchema`, and `MomentSeeker` are currently confirmed as benchmark / test mirrors and should not enter default training configs directly; the `MVBench` / `EgoSchema` directions can still use split subsets from `OpenGVLab/VideoChat2-IT-clean` as training proxy candidates, instead of training on the benchmarks themselves.

## Status Terms

This page uses only the candidate statuses defined by `expand-training-dataset-catalog`. Finer judgments, such as same-family proxy training sources, test mirrors, or the absence of a same-name public train split, belong in coverage entries, source kind, split / media notes, and next actions instead of new status values.

| Status | Meaning |
|--------|---------|
| `implemented` | The current repo already has a production training entry, standalone runtime, or production eval support |
| `covered_by_existing_family` | Covered by a family-level training source, without a same-name training subset |
| `design_only` | Only a design placeholder exists; there is no loader, registry, or local Lance sample yet |
| `pending_download` | The source has not been downloaded locally yet |
| `downloaded_pending_audit` | Local source data exists, but schema / split / overlap audit is not complete |
| `audit_blocked` | Annotation, media package, provenance, or license still needs confirmation |
| `train_candidate` | A train split or training-corpus entry is visible and can be audited |
| `formal_candidate` | Included in the next formal-admission queue, but still requires source, license, split, media, overlap, and real-root smoke checks before enablement |
| `eval_only` | The visible entry is a benchmark, validation, test mirror, or OOD eval set |
| `pretrain_only` | Better suited to large-scale pretraining, not the current finetuning path |
| `defer` | Deferred from this training expansion |

| Source kind | Meaning |
|-------------|---------|
| `official_upstream` | Official website or paper project release |
| `hf_official_or_author` | Official or author-maintained Hugging Face entry |
| `hf_eval_mirror` | Hugging Face evaluation mirror |
| `community_mirror` | Community mirror that needs extra provenance auditing |
| `local_only` | Visible in this repo or local audit notes; external entry still needs confirmation |
| `community_cleaned_manifest` | Community-cleaned or repackaged annotation / manifest that needs extra provenance, license, and media-policy auditing |

## External Evidence

| Object | Entry | Conclusion |
|--------|-------|------------|
| VLM2Vec official repo | <https://github.com/TIGER-AI-Lab/VLM2Vec> | Official training configs mainly list MMEB-train, ViDoRe, VisRAG, and LLaVA-Hound |
| Official training config | <https://raw.githubusercontent.com/TIGER-AI-Lab/VLM2Vec/main/experiments/public/train/train_alltasks.yaml> | Confirms 20 MMEB-train image subsets and several family-level training sources |
| MMEB-V2 eval | <https://huggingface.co/datasets/TIGER-Lab/MMEB-V2> | Evaluation frames / parquet, not a training entry |
| MMEB raw video | <https://huggingface.co/datasets/TIGER-Lab/MMEB_Raw_Video> | Raw video media entry, not training annotations |
| VizWiz VQA | <https://vizwiz.org/tasks-and-datasets/vqa/> | Official VQA train / validation / test entry, suitable for P0 audit |
| RefCOCO family | <https://www.tensorflow.org/datasets/catalog/ref_coco> | Official splits are visible, suitable for P0/P1 audit |
| ObjectNet | <https://objectnet.dev/download.html> | OOD evaluation source, not a default training source |
| Video-MME | <https://huggingface.co/datasets/siyrus/BToks-Video-MME> | Test-only eval mirror |
| MVBench | <https://huggingface.co/datasets/siyrus/BToks-MVBench> | Benchmark package / eval mirror |
| EgoSchema | <https://huggingface.co/datasets/siyrus/BToks-EgoSchema> | Test / subset-test eval mirror |
| MomentSeeker | <https://huggingface.co/datasets/siyrus/BToks-MomentSeeker> | Test-only eval mirror |
| OpenGVLab/VideoChat2-IT | <https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT> | Gated annotation source; the card reports about 1.9M JSON annotations and provides JSON only, while media must be retrieved from source datasets |
| VideoChat2-IT-clean | <https://huggingface.co/datasets/byminji/VideoChat2-IT-clean> | Community cleaned video-subset manifest; the local 16 `train.json` files contain 877,489 samples in total, matching the upstream per-subset table, while the upstream README total field says 874,869; media-backed subsets are now formally wired through `videochat2_it_train` with 195,584 training samples, while remaining subsets still need source-level license / provenance / media audits |
| VATEX | <https://huggingface.co/datasets/siyrus/BToks-VATEX>, <https://huggingface.co/datasets/lmms-lab/VATEX> | Eval entries still mainly expose test data; training uses the local official VATEX train JSON and video archive. `vatex_train` now has a production `data/frames.lance` root and is enabled in v2 |
| RefCOCO HF media mirror | <https://huggingface.co/datasets/xche32/refcpco> | Sampling confirms original `refcoco/images/train2014/COCO_train2014_...` keys; download `refcoco.tar.gz` locally first, then use it as the conversion media source |
| ViDoSeek page | <https://huggingface.co/datasets/siyrus/BToks-ViDoSeek-page-fixed> | corpus / queries are test; qrels train alone is not a complete training source |
| MMLongBench page | <https://huggingface.co/datasets/siyrus/BToks-MMLongBench-page-fixed> | Test-only eval mirror |

## Image Subsets

| Subset | task_type | Status | Coverage or Download Entry | Source kind | Split / Media Notes | Next Action |
|--------|-----------|--------|----------------------------|-------------|---------------------|-------------|
| `VOC2007` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `N24News` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `SUN397` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `ImageNet-1K` | image_classification | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `HatefulMemes` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `OK-VQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `A-OKVQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `DocVQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `InfographicsVQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `ChartQA` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `Visual7W` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `VisDial` | visual_question_answering | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `CIRR` | composed_image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `VisualNews_t2i` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `VisualNews_i2t` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `MSCOCO_t2i` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `MSCOCO_i2t` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `NIGHTS` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `WebQA` | image_retrieval | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `MSCOCO` | image_captioning | `implemented` | `TIGER-Lab/MMEB-train` / `mmeb_train` | `hf_official_or_author` | `original` split, image side table | Covered, skip |
| `ImageNet-A` | image_classification | `audit_blocked` | ImageNet-A / ImageNet-1K proxy | `official_upstream` | OOD eval set; ImageNet-1K can be discussed as a proxy | P2, proxy-only discussion |
| `ImageNet-R` | image_classification | `audit_blocked` | ImageNet-R / ImageNet-1K proxy | `official_upstream` | OOD eval set; ImageNet-1K can be discussed as a proxy | P2, proxy-only discussion |
| `ObjectNet` | image_classification | `eval_only` | ObjectNet official download | `official_upstream` | OOD evaluation source; license / access needs confirmation | P2, do not enter default training |
| `Country211` | image_classification | `implemented` | `country211_train` | `hf_official_or_author` | default train split excludes eval boundary | Covered, skip |
| `Place365` | image_classification | `implemented` | `place365_train` | `official_upstream` | official train split | Covered, skip |
| `ScienceQA` | visual_question_answering | `implemented` | `scienceqa_train` | `hf_official_or_author` | official train without MMEB-V2 eval overlap | Covered, skip |
| `GQA` | visual_question_answering | `implemented` | `gqa_train` | `official_upstream` | official balanced train without MMEB-V2 eval overlap | Covered, skip |
| `TextVQA` | visual_question_answering | `implemented` | `textvqa_train` | `official_upstream` | official train without MMEB-V2 eval overlap | Covered, skip |
| `VizWiz` | visual_question_answering | `implemented` | `vizwiz_train` / VizWiz VQA official | `official_upstream` | `official_train_answerable_without_mmeb_v2_eval` has 14,012 rows, and the image table has 24,842 rows | Covered, skip |
| `Wiki-SS-NQ` | image_retrieval | `implemented` | `wikissnq_train` | `hf_official_or_author` | official train without MMEB-V2 eval overlap | Covered, skip |
| `FashionIQ` | composed_image_retrieval | `implemented` | `mbeir_train` / `FashionIQ_train` | `hf_official_or_author` | M-BEIR task 7 | Covered, skip |
| `OVEN` | image_retrieval | `implemented` | `mbeir_train` / OVEN train views | `hf_official_or_author` | M-BEIR task 6 / 8 | Covered, skip |
| `EDIS` | image_retrieval | `implemented` | `mbeir_train` / `EDIS_train` | `hf_official_or_author` | M-BEIR task 2 | Covered, skip |
| `Visual7W-Pointing` | visual_grounding | `implemented` | `visual7w_pointing_train` | `official_upstream` | official train without MMEB-V2 eval overlap | Covered, skip |
| `RefCOCO` | visual_grounding | `implemented` | `refcoco_train` | `hf_metadata_plus_local_refcoco_tar_media` | train / validation / testA / testB; default training split has 38,181 rows | Covered, skip |
| `RefCOCO-Matching` | image_retrieval | `implemented` | `refcoco_train` matching view | `hf_metadata_plus_local_refcoco_tar_media` | Shares the `RefCOCO` default split and image side table; both query and positive sides keep the referring-expression prompt | Covered, enabled in v2 |

## Video Subsets

| Subset | task_type | Status | Coverage or Download Entry | Source kind | Split / Media Notes | Next Action |
|--------|-----------|--------|----------------------------|-------------|---------------------|-------------|
| `K700` | video_classification | `implemented` | `kinetics700_train` | `official_upstream` | invalid-filtered official train; the production root now has `data/frames.lance` and `data/videos.lance`, with 536,489 default split rows | Frame-unit rerun, production-root replacement, and sampling smoke are complete; enabled in v2 |
| `UCF101` | video_classification | `implemented` | `ucf101_train` | `official_upstream` | official train splits | Covered, skip |
| `HMDB51` | video_classification | `implemented` | `hmdb51_train` | `official_upstream` | official train splits | Covered, skip |
| `SmthSmthV2` | video_classification | `implemented` | `ssv2_train` | `official_upstream` | official train / validation | Covered, skip |
| `Breakfast` | video_classification | `implemented` | `breakfast_train` | `official_upstream` | archive train, non-`cam01` filter | Covered, skip |
| `Video-MME` | video_question_answering | `eval_only` | `siyrus/BToks-Video-MME` | `hf_eval_mirror` | test only, about 2700 rows | P2, proxy source only |
| `MVBench` | video_question_answering | `eval_only` | `siyrus/BToks-MVBench`; proxy candidates are split `OpenGVLab/VideoChat2-IT-clean` subsets such as `clevrer_*`, `next_qa`, `ssv2`, and `k710` | `hf_eval_mirror` / `community_cleaned_manifest` | Do not train on the benchmark body; proxy subsets need per-subset duplicate removal against existing runtimes, and `clevrer_*` / `tgif_*` also need MVBench eval-family overlap audit | P0/P1, process proxy subsets first and keep the benchmark body out |
| `NExTQA` | video_question_answering | `implemented` | `nextqa_train` | `hf_official_or_author` | `official_train_without_mmeb_v2_eval` has 34,132 rows, and the video table has 3,870 rows | Covered, skip |
| `EgoSchema` | video_question_answering | `eval_only` | `siyrus/BToks-EgoSchema`; proxy candidate is `OpenGVLab/VideoChat2-IT-clean` `ego_qa` | `hf_eval_mirror` / `community_cleaned_manifest` | EgoSchema itself is test / subset-test; `ego_qa` has only 7,797 valid samples and does not require adding full Ego4D, but still needs Ego4D clip-id eval-family overlap audit | P1, audit the small `ego_qa` subset |
| `ActivityNetQA` | video_question_answering | `covered_by_existing_family` | `activitynet_captions_train` | `hf_official_or_author` / `local_only` | Related caption / segment source exists, but it is not a same-name QA train set; this round uses a segment-level retrieval training view | Covered as a same-family training proxy; keep for quality audit only |
| `MSR-VTT` | text_video_retrieval | `implemented` | `msrvtt_train` | `official_upstream` | train 9k without MMEB-V2 eval overlap | Covered, skip |
| `MSVD` | text_video_retrieval | `implemented` | `msvd_train` | `official_upstream` | train without MMEB-V2 eval overlap | Covered, skip |
| `DiDeMo` | text_video_retrieval | `implemented` | `didemo_train` | `official_upstream` | train without MMEB-V2 eval overlap | Covered, skip |
| `VATEX` | text_video_retrieval | `implemented` | `vatex_train` / official VATEX train JSON and video archive | `official_upstream` / `local_only` | Production root has been migrated to frame units; `official_train_without_mmeb_v2_eval` has 22,105 rows, `data/frames.lance` has 22,105 rows, and 3,886 bad-media rows are excluded | Covered, enabled in v2 |
| `YouCook2` | text_video_retrieval | `implemented` | `youcook2_train` | `community_mirror` / `local_only` | `official_train_without_mmeb_v2_eval` has 10,330 rows, and the video table has 1,332 rows | Covered, skip |
| `MomentSeeker` | text_video_retrieval | `eval_only` | `siyrus/BToks-MomentSeeker` | `hf_eval_mirror` | test only, about 1602 rows | P2, do not enter default training |
| `QVHighlight` | video_moment_retrieval | `implemented` | `qvhighlight_train` | `official_upstream` | official train without MMEB-V2 eval overlap | Covered, skip |
| `Charades-STA` | video_moment_retrieval | `implemented` | `charades_sta_train` | `official_upstream` | official train without MMEB-V2 eval overlap | Covered, skip |

## VisDoc Subsets

| Subset | task_type | Status | Coverage or Download Entry | Source kind | Split / Media Notes | Next Action |
|--------|-----------|--------|----------------------------|-------------|---------------------|-------------|
| `ViDoRe_arxivqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_docvqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_infovqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_tabfquad` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_tatdqa` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_shiftproject` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_syntheticDocQA_artificial_intelligence` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_syntheticDocQA_energy` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_syntheticDocQA_government_reports` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_syntheticDocQA_healthcare_industry` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_esg_reports_human_labeled_v2` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_biomedical_lectures_v2_multilingual` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_economics_reports_v2_multilingual` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoRe_esg_reports_v2_multilingual` | document_retrieval | `covered_by_existing_family` | `vidore/colpali_train_set` / `vidore_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `VisRAG_ArxivQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `VisRAG_ChartQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `VisRAG_MP-DocVQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `VisRAG_SlideVQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `VisRAG_InfoVQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `VisRAG_PlotQA` | document_retrieval | `covered_by_existing_family` | `openbmb/VisRAG-Ret-Train-In-domain-data` / `visrag_train` | `hf_official_or_author` | family-level train | Covered, skip |
| `ViDoSeek-page` | document_retrieval | `eval_only` | `siyrus/BToks-ViDoSeek-page-fixed` | `hf_eval_mirror` | corpus / queries are test; qrels train alone is not a complete training source | P1, confirm whether an official train corpus / query set exists |
| `ViDoSeek-doc` | document_retrieval | `audit_blocked` | MMEB-V2 local eval assembly / ViDoSeek | `local_only` | Only eval assembly is confirmed | P1, find official source first |
| `MMLongBench-page` | document_retrieval | `eval_only` | `siyrus/BToks-MMLongBench-page-fixed` | `hf_eval_mirror` | test-only eval mirror | P1/P2, confirm whether page-level train exists or this is eval-only |
| `MMLongBench-doc` | document_retrieval | `implemented` | `mmlongbench_doc_train` | `hf_official_or_author` / `local_only` | default split has only 4 non-overlap rows | Covered but high risk; keep for quality audit only |

## New Candidate Priority

| Priority | Objects | Entry Condition | Reason |
|----------|---------|-----------------|--------|
| Next formal P0 | `SNLI-VE`, `MSRVTT-QA`, `MSVD-QA`, and the remaining no-local-media split `OpenGVLab/VideoChat2-IT-clean` subsets | Complete source / license / split / media dependency / MMEB-V2 overlap audits before conversion, runtime, and config work; the media-backed `VideoChat2-IT` subsets are already split and wired, and remaining subsets must not be merged into one media-less enabled entry | Their task semantics are clear, and they either have a local source candidate or can reuse an existing production video root; `VideoChat2-IT` now provides one media-backed proxy batch for `MVBench` / `EgoSchema` directions |
| Next formal P1 | `ActivityNetQA`, `SOP` / Stanford Online Products, `DeepFashion In-Shop`, `LaSCo` | Confirm annotations, media roots, task semantics, licenses, and overlap keys; enable only after real-root smoke passes | They have clear value, but source dependencies, overlap boundaries, or training semantics still need focused audits |
| Keep as candidates | sampled `mmE5 synthetic`, `Kinetics-400`, `TVQA`, `iVQA`, `Diving48`, `FineGym`, `Epic-Kitchens-100`, `RefCOCO+`, `RefCOCOg` | Keep them for v3 / future Stage 1 or dedicated tasks, not the current formal queue | They may be useful, but they should not block the next formal-admission batch |
| Remove from formal candidates | `ViDoSeek-page`, `ViDoSeek-doc`, `MMLongBench-page`, `Video-MME`, the `MVBench` benchmark body, the `EgoSchema` benchmark body, `MomentSeeker`, `ImageNet-A`, `ImageNet-R`, `ObjectNet` | Keep as eval-only, blocked, or proxy-discussion items; do not wire them as default training data | Current evidence points to test-only mirrors, eval mirrors, or OOD benchmarks, not formal training sources; this removes the benchmark bodies, not proxy sources such as `VideoChat2-IT` |

## Single-Dataset Audit Template

Each `train_candidate` or P1 entry needs these checks before implementation:

1. Upstream source, license / usage policy, download terms, and redistribution constraints.
2. Raw annotation schema, split names, row count, media-file count, and missing-media strategy.
3. Overlap key: image/video/document id, question id, caption id, segment timestamp, or bbox id.
4. Default training split name and how MMEB-V2 eval overlap will be excluded.
5. `TrainSample` semantics: classification, VQA, retrieval, moment retrieval, grounding, or document retrieval.
6. Whether the dataset should become a standalone runtime, join an existing family, or remain a proxy training source.
7. Why the MMEB-V2 eval mirror or a test-only mirror cannot be used directly as training data.
