# Pending Training Datasets

This document reviews data sources that appeared in the Qwen3vl archive experiments (the expanded public training source survey) and separates them into two groups:

- `archive references already covered today`: the current BToks repository already has a production training loader / registry for the underlying data source, while the archive may still differ in pairing view, source split, or sampling policy;
- `truly pending extra data`: datasets used in the archive that are still not part of the current production training registry.

The main purpose of this page is the second group, because those are the candidates that may improve training performance while still requiring split / test-overlap audits before adoption.

> Current policy: expanded training datasets move forward as one rewrite task per dataset. Each task first confirms the raw schema, origin parser behavior, old implementation fixes, and target `TrainSample` output, then rewrites conversion/runtime; after replacement, the matching legacy script, runtime, and docs entry are removed.

> Training families already covered in the current repository: `MMEB`, `ViDoRe`, `VisRAG`, `LLaVA-Hound`, standalone video classification / retrieval / moment-retrieval runtimes, standalone OOD / VQA runtimes, `M-BEIR`, and `Visual7W-Pointing`. The second-round full training config is `configs/datasets/expanded_train_datasets_v2.yaml`; it inherits v1 and appends the currently sampleable `VizWiz`, `RefCOCO`, `RefCOCO-Matching`, `VATEX`, and `K700` entries. `K700` has completed its frame-unit rerun, production-root replacement, and sampling smoke, so it is no longer a pending frame-unit rerun item. Covered families should not be treated as pending families anymore.

> Default expanded-training weights: after moving `K700` out of v1, `configs/datasets/expanded_train_datasets_v1.yaml` uses a conservative `image / video / visdoc ≈ 46.3 / 17.2 / 31.8` mix, with entries inside each modality assigned by `samples^0.78`. `alpha=0.78` keeps every enabled entry at a natural rounded weight of at least `0.1`, so no extra minimum-weight floor is needed. The modality mix is a target range, not an exact integer constraint. GQA defaults to the official balanced train split, so the current default sample pool is about 6.5M rows and the expected `all_exhausted` epoch is about 14.1M rows, below the current acceptable 15M cap; the 14.3M-row GQA `train_all` split remains available as an explicit full-source view.

> Legacy-boundary conclusion: this training rewrite no longer uses `render.py`, `ReplaceDefaultTransform`, `materialize_training_view`, or old MMEB-V2 eval parser internal `render_*` / `_materialize_*` names to construct training samples. `render.py` and `vlm2emb.data.contracts` have both been removed; MMEB-V2 parser text assembly now uses `compose_*` / `format_*` names, and item builders now use `_build_*` names.

## 2026-04-20 Unified Inventory Policy

This section is the initial inventory for `expand-training-dataset-catalog`. It places the old pending notes, design-only cards, current production configs, archive parser clues, `lmms-lab` scan candidates, and the Module C recommendation list into one status vocabulary. The goal is not to implement every dataset in one pass; it is to create a stable entry point for later one-dataset rewrite tasks.

### Status and coverage vocabulary

| Field | Value | Meaning |
|-------|-------|---------|
| `status` | `implemented` | A production training loader / registry or production eval path exists today |
| `status` | `covered_by_existing_family` | Covered by `MMEB`, `MMEB-V2` eval, `ViDoRe`, `VisRAG`, or `LLaVA-Hound`; do not implement again |
| `status` | `design_only` | Design card only; no production loader / registry / local Lance sample yet |
| `status` | `pending_download` | Not downloaded locally yet |
| `status` | `downloaded_pending_audit` | Local source data exists, but schema / split / overlap audit is not complete |
| `status` | `audit_blocked` | Known risk must be resolved first: split, source, license, or sample coverage |
| `status` | `train_candidate` | Audit passed or risk is low enough to enter a one-dataset rewrite |
| `status` | `eval_only` | Mainly useful for evaluation, not a training candidate |
| `status` | `pretrain_only` | Suitable for large-scale pretraining, outside the current fine-tuning data path |
| `status` | `defer` | Deferred until task definition or local resources are clearer |
| `coverage` | `none` | No current coverage |
| `coverage` | `train_implemented` | A production training implementation exists |
| `coverage` | `eval_implemented` | A production eval implementation exists |
| `coverage` | `design_card_only` | Design card only |
| `coverage` | `same_source_different_view` | Same upstream source, different training view |
| `coverage` | `mirror_or_duplicate` | Mirror or duplicate source |

### Sources inspected

| Source | Extracted content | Meaning for later tasks |
|--------|-------------------|-------------------------|
| This old pending page | Current pending, downloaded, pending-audit, archive-view differences, and `lmms-lab` scan candidates | Initial candidate pool for one-dataset tasks |
| `docs/zh/datasets/train/ood/` and `docs/en/datasets/train/ood/` | Design cards and old loader clues for `Place365`, `Country211`, `ScienceQA`, `GQA`, `TextVQA`, `Wiki-SS-NQ`, and `MMLongBench-doc` | Separate `design_only`, legacy runtime, and not-yet-implemented objects |
| `docs/zh/datasets/train/mbeir/` and `docs/en/datasets/train/mbeir/` | Target schema and M-BEIR task ids for `OVEN`, `EDIS`, and `FashionIQ` | Source/task mapping entry point for later M-BEIR rewrites |
| `docs/zh/datasets/train/visual-grounding/` and `docs/en/datasets/train/visual-grounding/` | Legacy Lance schema, split, and runtime clues for `visual7W_pointing_train` | Old behavior reference for the later visual-grounding rewrite |
| `configs/datasets/mmeb_train.yaml` | Current production training path for `MMEB`, `ViDoRe`, `VisRAG`, and `LLaVA-Hound` | Confirms which recommended datasets are already covered by basic families |
| `configs/datasets/expanded_train_datasets_v1.yaml` | Archive-era expanded training recipe, current loadable training entries, and a few empty `type` placeholders waiting for download and audit; old `lance_*` / text-format entries are only historical cleanup inventory | Empty placeholders preserve future wiring positions, and production loaders should skip objects whose `type` is empty |
| the public training source survey | Origin parser filenames and training-view clues | Inspect one by one during each dataset rewrite; do not bulk-port old code |
| Module C recommendation list | Full recommended universe for Classification, Caption / Image-Text Pairs, Retrieval, VQA, and Visual Grounding | Assign train/eval/pretrain/defer status per dataset; appearing in the list does not imply implementation |

### Current production coverage boundary

| Family | Covered objects | Current status / coverage |
|--------|-----------------|--------|
| `MMEB` | `ImageNet_1K`, `N24News`, `HatefulMemes`, `VOC2007`, `SUN397`, `OK-VQA`, `A-OKVQA`, `DocVQA`, `InfographicsVQA`, `ChartQA`, `Visual7W`, `VisDial`, `CIRR`, `VisualNews_t2i`, `VisualNews_i2t`, `MSCOCO_t2i`, `MSCOCO_i2t`, `NIGHTS`, `WebQA`, `MSCOCO` | `implemented`; coverage = `train_implemented` |
| `MMEB-V2 eval` | `ImageNet-A`, `ImageNet-R`, `ObjectNet`, `Country211`, `Place365`, `ScienceQA`, `GQA`, `TextVQA`, `VizWiz`, `FashionIQ`, `Wiki-SS-NQ`, `OVEN`, `EDIS`, `RefCOCO`, `Visual7W-Pointing`, `UCF101`, `HMDB51`, `MSR-VTT`, `DiDeMo`, and other eval parsers | `eval_only`; coverage = `eval_implemented`; training still needs a separate decision |
| `ViDoRe` | Full `colpali_train_set` view and `arxiv_qa`, `pdf`, `docvqa`, `tatdqa`, `Infographic-VQA` source views | `implemented`; coverage = `same_source_different_view` |
| `ViDoRe RAG` | `vidore_rag_train` source-specific RAG view over the same `colpali_train_set` source | `implemented`; coverage = `same_source_different_view` |
| `VisRAG` | Full `visrag-indomain` view and `InfoVQA`, `ArxivQA`, `PlotQA`, `SlideVQA`, `MP-DocVQA`, `ChartQA` source views | `implemented`; coverage = `same_source_different_view` |
| `LLaVA-Hound` | `video-caption-300k`, `video-caption-300k-video`, `video-qa-240k` | `implemented`; coverage = `train_implemented` |
| `SmthSmthV2` | `ssv2_train` has frame-unit runtime / converter support; the formal root now contains `data/videos.lance` + `data/frames.lance`, and the default training path does not raw-decode videos | `implemented`; coverage = `train_implemented`; media status = `frame_unit_formal` |
| `HMDB51` | `hmdb51_train` has frame-unit runtime / converter support; the formal root now contains `data/videos.lance` + `data/frames.lance`, and the default training path does not raw-decode videos | `implemented`; coverage = `train_implemented`; media status = `frame_unit_formal` |
| `UCF101` | `ucf101_train` has frame-unit runtime / converter support; the formal root now contains `data/videos.lance` + `data/frames.lance`, and the default training path does not raw-decode videos | `implemented`; coverage = `train_implemented`; media status = `frame_unit_formal` |

### Basic Family Review

| Object | Current implementation boundary | Handling in the expansion catalog |
|--------|---------------------------------|-----------------------------------|
| `MMEB-train` | `mmeb_train` uses raw-preserving parquet plus `data/images/<subset>.lance` image tables. The runtime inherits from `SampleLanceDataset`; the default transform emits `TrainSample` directly, and a custom `transform` replaces the default function instead of going through a legacy formatting path. | Covered classification, retrieval, VQA, caption, and grounding subsets are marked as `covered_by_existing_family`; coverage may be recorded as `train_implemented`, and no duplicate training family is created. |
| `MMEB-V2 eval` | `mmeb_v2` is an eval boundary backed by eval loaders and parser registry. Parser-internal text assembly uses `compose_*` / `format_*` names, and is not the training runtime path. | `ImageNet-A/R`, `ObjectNet`, `Country211`, `FashionIQ eval`, and similar non-enabled objects are handled as `eval_only` or placeholders; `RefCOCO` and `RefCOCO-Matching` are now covered by `refcoco_train`. |
| `ViDoRe` | `vidore_train` reads the `data/train.lance` full view. `vidore_rag_train` expresses the archive RAG view. Both use dataset-owned transforms, with no old eval/schema helper as the training entry. | Keep full-view and five source-view coverage. Mark the RAG view as `same_source_different_view`; do not treat these sources as new upstream datasets. |
| `VisRAG` | `visrag_train` is described separately from ViDoRe, reads `data/train.lance`, injects source-specific training prompts, and emits `TrainSample`; old visdoc helpers are not training entries. | Keep full-view and six source-view coverage. Source-specific configs only represent same-source sampling members. |
| `LLaVA-Hound` | `llavahound_train` uses instruction tables under `data/video_instruction/<subset>.lance` and the frame table `data/train_300k.lance`. The frame table fields are `video_id`, `frame_idx`, and `image`; runtime joins frames by the raw `video` field and emits `TrainSample`. | The expansion catalog keeps only the current implementation; after legacy script cleanup, there is no second video-instruction training path that users could mistake as equivalent. |

### Design-only card inventory

| Family | Datasets | Current status / coverage | Legacy code clue | Next step |
|--------|----------|-------------------|------------------|-----------|
| OOD | `Country211`, `Place365`, `ScienceQA`, `GQA`, `TextVQA` | `implemented` / `train_implemented` | Migrated to one-dataset conversion/runtime | Keep row count, schema, and eval-overlap notes current |
| OOD | `Wiki-SS-NQ` | `implemented` / `train_implemented` | Migrated to `wikissnq_train`; full conversion, join coverage, and runtime sampling are complete | Default training split is `official_train_without_mmeb_v2_eval` |
| OOD | `MMLongBench-doc` | `implemented` / `train_implemented` | 473 of the 1091 raw rows reproduce the archive single-page training view; 469 of those overlap MMEB-V2 eval queries | The default train split keeps only the 4 non-overlapping rows; use `official_train` only for archive reproduction |
| M-BEIR | `OVEN`, `EDIS`, `FashionIQ`, `Fashion200K`, `INFOSEEK` | `implemented` / `train_implemented` | Migrated to `mbeir_train`; full conversion is complete | Keep reviewing the MMEB-V2 eval exclusion boundary |
| Visual Grounding | `visual7W_pointing_train` | `implemented` / `train_implemented` | Migrated to `visual7w_pointing_train`; no legacy runtime / text-format path remains | Keep `ScreenSpot-*` as later separate tasks |

### Module C recommendation split

| Category | Covered or do not reimplement | Need one-dataset decision / rewrite | Not in current fine-tuning path |
|----------|-------------------------------|-------------------------------------|---------------------------------|
| Classification | `ImageNet-1K`, `SUN397`, `VOC2007`, `N24News`, and `HatefulMemes` are covered by `MMEB`; `ImageNet-A/R` and `ObjectNet` are mostly eval; `SmthSmthV2`, `HMDB51`, `UCF101`, `Kinetics-700`, `Breakfast`, `Place365`, and `Country211` now have training runtimes | None | None |
| Caption / Image-Text Pairs | `MSCOCO` is covered by `MMEB`; `LLaVA-Hound` is implemented | `Flickr30K`, `SBU Captions`, `WebVid-2.5M/10M` need purpose decisions first | `CC3M`, `CC12M`, `LAION-400M/5B` are `pretrain_only` / `defer` for now |
| Retrieval | `MSCOCO`, `VisualNews`, `VisDial`, `CIRR`, `WebQA`, and `NIGHTS` are covered by `MMEB`; `ViDoRe`, `VisRAG`, `MSR-VTT`, `MSVD`, `DiDeMo`, `Charades-STA`, `QVHighlights`, `ActivityNet Captions`, `YouCook2`, `M-BEIR`, and `Wiki-SS-NQ` are implemented | None | `FashionIQ eval` remains eval-only |
| VQA | `OK-VQA`, `A-OKVQA`, `DocVQA`, `InfographicVQA`, `ChartQA`, `Visual7W` are covered by `MMEB` or document-retrieval families; `ScienceQA`, `GQA`, `TextVQA`, `VizWiz`, `NExTQA`, and `MMLongBench-doc` are implemented | None | None |
| Visual Grounding | `MSCOCO` and some `RefCOCO`-like objects already have eval / MMEB-V2 clues; `visual7W_pointing_train` is implemented; `RefCOCO` and `RefCOCO-Matching` now have local conversion/runtime/config/tests | `ScreenSpot-v2`, `ScreenSpot-Pro` | `RefCOCO+ / RefCOCOg` do not enter v2 enabled scope yet |

### Long-term Candidate Pool

The following candidates do not enter the current default training rewrite. They are only entry points for later one-dataset tasks. Each one still needs official source, license, schema, split, and local path confirmation before conversion.

| Dataset | Purpose | Local Path Status | Current Handling |
|---------|---------|-------------------|------------------|
| `LaSCo` | compositional / language-guided retrieval candidate | no clear directory found under `./data/converted` | `pending_download`; confirm official source and license before downloading |
| `SNLI-VE` | image+text -> label | found `./data/converted` | later audit train/dev/test, image source, and license |
| `NLVR2` | multi-image visual reasoning | no clear local directory found | `pending_download`; confirm official source and split before downloading |
| `VCR` | image+text QA / rationale | no clear local directory found | `pending_download`; confirm usage terms first |
| `SOP` / Stanford Online Products | image retrieval | found `./data/converted` | later audit class / product splits and retrieval target |
| `DeepFashion In-Shop` | image retrieval | found `./data/converted` | later audit train/gallery/query splits and overlap with FashionIQ / Fashion200K |
| `Flickr30K Entities` | grounding / phrase localization | no clear local directory found | `defer`; confirm the training target before default training adoption |
| `Visual Genome` | grounding / region / relation | no clear local directory found | `pending_download`; define the exact task view first |
| `Docmatix` | document / visdoc | no clear local directory found | `pending_download`; confirm license and page media schema |

Large-scale or synthetic sources do not block the current fine-tuning rewrite:

| Dataset | Current Strategy | Local Path Status |
|---------|------------------|-------------------|
| `LAION`, `DataComp`, `Recap-DataComp` | `pretrain_only`; requires separate sampling, filtering, license, and safety policy | only `./data/converted` is present locally; it is not full LAION / DataComp |
| `CC3M`, `CC12M`, `WebVid`, `HowTo100M`, `InternVid` | `pretrain_only` or large-scale video pretraining strategy | only Unite-derived `WebVid-CoVR` and `InternVid-FLT` were found locally; they are not the official full sources |
| `MegaPairs`, `MagicLens`, `mmE5 synthetic` | synthetic / derived-data strategy; generation and dedup rules need separate review | no clear local directory found |

### One-dataset rewrite template

Every dataset that enters implementation uses this sequence:

1. Inspect the raw source schema, split, samples, and media fields.
2. Inspect how the origin parser builds query, positive, negative, media input, and instruction.
3. Inspect old repository fixes; keep only fixes that are still necessary and explainable.
4. Define all split-view sources, row counts, whether they point to the same real records, and whether the default training split must exclude MMEB-V2 eval samples.
5. Define the target `TrainSample` output, including text format, visual-token position, newline rules, optional-negative semantics, and metadata retention.
6. Implement raw-preserving conversion in `src/vlm2emb/data/conversion/<dataset>.py`; the script wrapper only parses arguments.
7. Implement the dataset-owned default transform, and make config defaults match runtime defaults.
8. Run conversion smoke, runtime sampling, bad-row-neighborhood checks, config loading, and overlap checks.
9. Remove the superseded script, runtime, helper, docs entry, and tests for that dataset.

### Initial Rewrite Lessons And Future Rules

The first video-classification migrations exposed issues that are likely to recur in more complex datasets:

- Official raw data, VLM2Vec metadata, archive train views, and old Lance roots may not be the same view. Docs must separately record unique media counts, split row counts, and archive train row counts.
- Multiple split views may point to the same real records. New conversion should write each split as a primary readable table such as `data/<split>.lance`, while shared images, videos, frames, or records live in side tables joined by stable keys.
- The default training entry must not silently mutate the original split. If MMEB-V2 test samples need to be removed, conversion should create a clearly named training split such as `train_without_mmeb_v2_eval` or a dataset-specific equivalent.
- MMEB-V2 is a Tiger-Lab curated composite benchmark. It may directly use upstream test splits or manually selected split variants, so related training candidates require sample-level overlap audits.
- If no reliable record, media, or query key can exclude MMEB-V2 eval samples, the dataset should remain `downloaded_pending_audit` or `audit_blocked` and must not enter the default training mixture.
- Manifest, dataset-info, and contract ids from old scripts are not runtime requirements. New conversion should preserve raw fields first and move only large media into Lance side tables.
- Bad rows, missing positives, or narrow data-quality issues should be fixed or removed in conversion with a narrow scope. Runtime should not silently skip training-semantic errors.
- Large media conversion should expose `num_workers`, batch size, Lance file size, and index-creation parameters from the beginning to avoid full-rerun rework.

### Legacy reference cleanup inventory

| Type | Current hits | Handling in this round | Later owner / deletion condition |
|------|--------------|------------------------|----------------------------------|
| Legacy MMEB eval conversion scripts write `manifest.json` / use `vlm2emb.data.contracts` | Legacy `scripts/convert/mmeb_conversion_common.py`, `scripts/convert/mmeb_registry.py`, `scripts/convert/check_media_text_conflicts.py`, and `scripts/convert/validate_mmeb_conversion.py` have been removed; the maintained entrypoints are `src/vlm2emb/data/conversion/mmeb_v2/` and `scripts/convert/eval/mmeb_v2.py` | New training conversion no longer writes manifest / README / dataset info; MMEB-V2 eval conversion uses the maintained conversion package | `cleanup-vlm2emb-data`; `vlm2emb.data.contracts` has been removed |
| Legacy training runtimes use `render`, `ReplaceDefaultTransform`, `TrainLanceDataset` | `image_classification.py`, `image_retrieval.py`, and `video_moment_retrieval.py` are deleted; `visual7w_pointing.py` is rewritten; `TrainLanceDataset` is no longer exported; the old `render.py` compatibility wrapper has been removed | Training runtime now uses `SampleLanceDataset` + dataset-owned transform; this rewrite no longer uses `render.py`, `ReplaceDefaultTransform`, or `TrainLanceDataset` | Old training wrappers were removed by `cleanup-vlm2emb-data`; MMEB-V2 parser-internal old `render_*` names now use `compose_*` / `format_*` |
| Legacy config still references `lance_*` and old text-format params | Expanded data section in `configs/datasets/expanded_train_datasets_v1.yaml` | Treat as archive-era behavior samples and cleanup targets, not as new defaults | `cleanup-vlm2emb-data`; keep only archive notes or delete after production configs and examples stop referencing legacy registry names or old text-format params |
| Legacy docs still describe text-format overrides | Current training-data docs no longer expose old text-format config entries; the `render.py` compatibility wrapper has been removed; MMEB-V2 parser internals now use `compose_*` / `format_*` names | Do not describe `render.py` as a training entry; MMEB-V2 eval parser text assembly remains parser-owned transform internals | Cleaned by `cleanup-vlm2emb-data` |
| `materialize_training_view` | No exact symbol hit in the current training path | Nothing to migrate for that symbol; keep preventing reintroduction | `cleanup-vlm2emb-data`; supported code and tests must not reintroduce runtime dependencies |
| `vlm2emb.data.contracts` | Legacy MMEB eval conversion, legacy data-intake workflow, and contract/catalog test references are removed | Do not add new training dependencies; no catalog / manifest / stable-id compatibility package remains | Removed by `cleanup-vlm2emb-data` |
| MMEB-V2 eval parser internal old `render_*` / `_materialize_*` naming | No old internal names remain under `src/vlm2emb/data/datasets/mmeb_v2/parsers/` | Parser-owned transform entrypoints stay stable; internal text assembly uses `compose_*` / `format_*`, and item builders use `_build_*` | Cleaned by `cleanup-vlm2emb-data`; keep eval output stability tests |

## Current Conclusion And Future Trigger

### 1. Record now, onboard later

At this stage the truly pending extra datasets on this page should be treated as:

- `pending download`: to be downloaded locally by the user later;
- `pending audit`: to be audited only after local samples are available;
- `pending decision`: to be admitted into the training mixture only after the audit passes.

### 2. Unified audit checklist for later local inspection

After the datasets are downloaded locally, each dataset should go through the same checks:

1. whether the raw split is explicit, and whether training truly consumes `train` rather than mixing in `val` / `test`;
2. whether there is sample-level overlap with current evaluation data, not just shared upstream source;
3. whether the archive-era parser directly references evaluation artifacts such as `vlm2vec_eval`, `MMEB_Test_Instruct`, or `MMEB-V2`;
4. whether multiple split views point to the same real records, and whether conversion should use split views as primary readable tables;
5. whether a dedicated training split that excludes MMEB-V2 eval samples is needed, and which record/media/query overlap key is available;
6. whether the dataset needs extra provenance, license, citation, or sampling-policy fields;
7. which task archetype it should map to, instead of expanding the family abstraction further.

### 3. Archive Download And Reading Policy

Future `tar`, `tar.gz`, `tgz`, and split-tar datasets should preserve the archive boundary by default instead of being fully unpacked into file trees. Conversion and runtime audits should first build an archive member index that records `archive_path`, `member_path`, stable media keys, split / label / query metadata, and then read member contents on demand.

For large `tar.gz` / `tgz` files, direct compressed-stream reading is not mandatory. To improve parallelism and random access, the compressed tar can first be normalized once into an uncompressed `.tar`; sampling and conversion then operate on the `.tar` plus its member index. This still preserves the archive layout and avoids creating a large number of small files.

A one-dataset task may unpack into a file tree only when the official instructions require an unpacked directory, an uncompressed tar is still not performant enough, standard tools cannot reliably read members, or the user explicitly requests unpacking. If unpacking is used, the task must document the reason, target layout, whether the original archive is kept, and how small-file overhead is controlled.

### 4. v1 Video Media Status Recheck

The 2026-04-28 recheck against `configs/datasets/expanded_train_datasets_v1.yaml` and the production data roots separates implementation status from media-table status: runtime / converter support does not mean the current production root has already been migrated to frame units. On 2026-04-29, `K700` was moved out of the v1 default expanded config and into the v2 pending frame-unit rerun scope, so it no longer blocks running v1 first. On 2026-05-08, `K700` completed the frame-unit rerun, production-root replacement, and v2 config enablement.

| Dataset | Runtime | Current production media status | Current rows or media count | Conclusion |
|---------|---------|---------------------------------|-----------------------------|------------|
| `video_caption_300k` / `video_caption_300k-video` / `video_qa_240k` | `llavahound_train` | Separate frame table `data/train_300k.lance` | 8,402,603 frame rows | Outside the frame-unit migration scope |
| `SmthSmthV2` | `ssv2_train` | `data/frames.lance` + `data/videos.lance` | 193,690 frame units | Production root has been migrated to frame units; the default path does not raw-decode video |
| `HMDB51` | `hmdb51_train` | `data/frames.lance` + `data/videos.lance` | 6,436 frame units | Staging result has been promoted to the formal root; the default path does not raw-decode video |
| `UCF101` | `ucf101_train` | `data/frames.lance` + `data/videos.lance` | 13,320 frame units | Staging result has been promoted to the formal root; the default path does not raw-decode video |
| `MSR-VTT` | `msrvtt_train` | `data/frames.lance` + `data/videos.lance` | 10,000 frame units | Production root has been migrated to frame units |
| `MSVD` | `msvd_train` | `data/frames.lance` + `data/videos.lance` | 1,970 frame units | Production root has been migrated to frame units |
| `DiDeMo` | `didemo_train` | `data/frames.lance` + `data/videos.lance` | 9,399 frame units | Production root has been migrated to frame units |
| `YouCook2` | `youcook2_train` | `data/frames.lance` + `data/videos.lance` | 10,330 frame units | Production root has been migrated to frame units |
| `ActivityNet-Captions` | `activitynet_captions_train` | `data/frames.lance` + `data/videos.lance` | 71,787 frame units / 14,926 videos | Production root has been migrated to segment frame units |
| `K700` | `kinetics700_train` | `data/frames.lance` frame-unit layout + `data/videos.lance` audit view | 536,489 videos | Moved out of v1 and enabled in v2; the production root has been replaced by the frame-unit root |
| `Breakfast` | `breakfast_train` | `data/frames.lance` + `data/videos.lance` | 1,989 frame units | Production root has been migrated to full-video frame units |
| `Charades-STA` | `charades_sta_train` | `data/frames.lance` + `data/videos.lance` | 18,444 frame units / 6,672 videos | Production root has been migrated to mixed-view frame units |
| `QVHighlights` | `qvhighlight_train` | `data/frames.lance` + `data/videos.lance` | 25,715 frame units / 10,148 videos | Production root has been migrated to mixed-view frame units |
| `NExTQA` | `nextqa_train` | `data/frames.lance` + `data/videos.lance` | 3,870 frame units | Production root has been migrated to full-video frame units |

### 5. Current risk conclusions

- `Breakfast`: local sample-level verification is complete and `breakfast_train` is wired. The archive directory-walk parser and `cam01` exclusion behavior are replaced by explicit split tables; the `cereal/cereals` and `salad/salat` naming differences are kept as reviewable alias fixes. The training runtime emits the canonical `label` as the positive text first, while `raw_label` remains metadata for audit and missing-label fallback. `SmthSmthV2`, `HMDB51`, `UCF101`, `Kinetics-700`, `Place365`, and `Country211` have been checked with the same rule; no source-label versus training-label split was found outside `Breakfast`.
- `SmthSmthV2`, `HMDB51`, and `UCF101`: local sample-level verification is complete and `ssv2_train`, `hmdb51_train`, and `ucf101_train` are wired. The default training path uses the VLM2Vec train view, while official splits are preserved as optional train/test views. Under the current policy, default training configs should not decode source videos at runtime; they should fail when `data/frames.lance` is missing, and only explicit raw-video compatibility configs may use the legacy fallback. All three frame-unit roots have been promoted to formal paths.
- `MSR-VTT`, `MSVD`, `DiDeMo`, `Charades-STA`, `QVHighlights`: local sample-level verification is complete, standalone runtimes are wired, and production roots have been migrated to frame units. `MSR-VTT`, `MSVD`, `DiDeMo`, `Charades-STA`, and `QVHighlights` have 10,000, 1,970, 9,399, 18,444, and 25,715 rows in `data/frames.lance`, respectively. `MSR-VTT`, `MSVD`, `DiDeMo`, and `Charades-STA` keep default train splits with the same row counts as their official train views; `QVHighlights` currently uses the expanded community split-video directory, and every annotated train/val/test video is present.
- `NExTQA`: `nextqa_train` conversion/runtime/config/tests are wired, and the production root has been migrated to frame units. The production root is under `./data/converted`; both `official_train` and `official_train_without_mmeb_v2_eval` have 34,132 rows, and both `data/frames.lance` and `data/videos.lance` have 3,870 rows. Overlap with the 1,000 MMEB-V2 `NExTQA` eval videos is 0. Runtime preserves the archive multiple-choice query template, emits the correct answer text as the positive, and keeps the negative side as empty text plus empty media.
- `YouCook2`: `youcook2_train` conversion/runtime/config/tests are now wired, and the production frame-unit root has been migrated. The converted root is under `./data/converted`; both `official_train` and `official_train_without_mmeb_v2_eval` have 10,330 rows, and `data/frames.lance` has 10,330 rows. Overlap with MMEB-V2 `YouCook2` eval youtube ids / segment stems is 0. Runtime uses the original train-view caption as a text-to-video query, and the positive comes from a segment frame unit restored from upstream row-level `binary_frames`; it no longer uses the full video as the positive. Each segment keeps up to 8 frames.
- `ActivityNet Captions`: the project now uses segment-level text-to-video retrieval semantics, and `activitynet_captions_train` conversion/runtime/config/tests are wired. The production root has been migrated to frame units. It is under `./data/converted`; `official_train` has 37,421 rows, `official_val1` has 17,505 rows, `official_val2` has 17,031 rows, `official_train_without_mmeb_v2_eval` has 37,421 rows, `data/frames.lance` has 71,787 rows, and `data/videos.lance` has 14,926 rows. Overlap with the 614 MMEB-V2 `ActivityNetQA` eval video ids is 0. Each positive segment keeps up to 8 frames.
- `VizWiz`: second-round `vizwiz_train` conversion/runtime/config/tests are wired. The converted root is under `./data/converted`; the default `official_train_answerable_without_mmeb_v2_eval` split has 14,012 rows, and `data/images.lance` has 24,842 rows. Question/answer overlap with MMEB-V2 eval removes 979 rows. Runtime uses image+question as the query and a confidence-aware majority-voted selected answer as the positive.
- `VATEX`: second-round `vatex_train` conversion/runtime/config/tests are wired, and the production root has been migrated to frame units. The production root is under `./data/converted`; the default `official_train_without_mmeb_v2_eval` split has 22,105 rows, `data/frames.lance` has 22,105 rows, `data/videos.lance` has 22,105 rows, 3,886 bad-media rows are recorded in `data/exclusions/bad_media.lance`, and video-id overlap with MMEB-V2 `VATEX` eval is 0. The runtime defaults to `data/frames.lance`, each full-video frame unit keeps up to 64 frames, and `VATEX` is now enabled in `expanded_train_datasets_v2.yaml`.
- `RefCOCO` / `RefCOCO-Matching`: second-round `refcoco_train` conversion/runtime/config/tests are wired. Metadata comes from `./data/converted`, and media comes from the local `./data/converted`; the converted root is under `./data/converted`. The default `official_train_without_mmeb_v2_eval` split has 38,181 rows, and `data/images.lance` has 19,994 rows. Conversion excludes 4,223 MMEB-V2 eval overlaps by `ann_id`. `RefCOCO` uses full image plus referring expression as the query and the bbox crop image as the positive. `RefCOCO-Matching` reuses the same root and split, while preserving the referring-expression prompt on the positive side.
- `OpenGVLab/VideoChat2-IT`: the media-backed subsets are now formally wired through `videochat2_it_train`, with the production root under `./data/converted`. Start from the video-only `byminji/VideoChat2-IT-clean` manifest; the local 16 `train.json` files contain 877,489 samples in total, matching the upstream per-subset table, while the upstream README total field says 874,869. The v2 enabled split currently includes only rows that resolve to local production media roots from `next_qa`, `youcook2`, `ssv2`, `k710`, and `videochatgpt`, expanding to 195,584 training samples. Missing-media `youcook2` / `k710` source rows are recorded in `data/exclusions/bad_media.lance`; the remaining no-local-media subsets stay audit-only.
- `Place365`: local sample-level verification is complete and `place365_train` is wired. The old directory-walk parser is replaced by explicit split tables.

### 6. Current trigger state

These prerequisites are now mostly satisfied:

- dataset governance and provenance foundations are in place;
- task-archetype abstractions have stabilized;
- dataset-owned adapter boundaries have largely converged;
- raw schema / quality / overlap audits are available for newly downloaded data.

Therefore, **the user can now start downloading the candidate datasets listed here locally** and then move into formal local auditing.

Recommended priority order:

1. Completed in this round and now included in the default training config: `NExTQA`, `YouCook2`, `ActivityNet Captions`
2. Stronger potential coupling with current eval datasets and requiring continued source-unit review: `OVEN_it2it_train`, `EDIS_train`, `FashionIQ_train`, `visual7W_pointing_train`
3. Long-term candidates whose official source or local path is still unresolved: `LaSCo`, `NLVR2`, `VCR`, `Visual Genome`, `Docmatix`
4. Continue in the second dataset expansion round: `ViDoSeek-page`, `ViDoSeek-doc`, `MMLongBench-page`
5. Split video-instruction proxy source: `OpenGVLab/VideoChat2-IT-clean`
6. Large-scale pretraining or synthetic sources: `LAION`, `DataComp`, `CC3M/CC12M`, `WebVid`, `HowTo100M`, `InternVid`

## Pending Summary Table

| Category | Configurations | Task Type |
|----------|----------------|-----------|
| Video Classification | 4 | video_classification |
| Video Retrieval | 8 | video_retrieval |
| Video Moment Retrieval | 2 | video_moment_retrieval |
| Video QA | 2 | video_qa |
| OOD Image Datasets | 6 | various |
| OOD Document Datasets | 1 | document_understanding |
| M-BEIR Datasets | 3 | multimodal_retrieval |
| Visual Grounding | 1 | visual_grounding |

---

## Archive References Already Covered Today

### ViDoRe RAG Subsets

ViDoRe is not a pending family in the current repository. BToks already supports the full `colpali_train_set` and the 5 source-level subset configs below through `vidore_train`.

These entries are not new upstream data sources. They come from archive-era experiments that split the same `vidore/colpali_train_set` training set by `source`.

- **HuggingFace**: [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set)
- **Paper**: ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449))

| Config Key | Subset Name | Samples | Task |
|------------|-------------|---------|------|
| colpali_train_set_arxiv_qa | arxiv_qa | 9,949 | document_retrieval |
| colpali_train_set_pdf | pdf | 45,718 | document_retrieval |
| colpali_train_set_docvqa | docvqa | 39,302 | document_retrieval |
| colpali_train_set_tatdqa | tatdqa | 13,188 | document_retrieval |
| colpali_train_set_Infographic-VQA | Infographic-VQA | 10,038 | document_retrieval |

Note:

- The current project already supports the full `colpali_train_set` and these 5 source-specific subset configs.
- These 5 configs are still `source`-level splits of the same training dataset, not additional HuggingFace datasets.
- The archive-specific information worth preserving is not "new data", but a "different training view": the subset configs used the `vidore_rag_single` pairing style, where the query side becomes a text retrieval instruction and the positive side becomes the document image. That differs from the current documented full-dataset `vidore` pairing.
- This group should therefore be understood as "already-covered upstream data + archive pairing-view differences", not as new pending training data.
- Current implementation path:
  - `vidore_train` keeps the full-view semantics;
  - `vidore_rag_train` expresses the archive `vidore_rag_single` derived training view directly, not as a `source_filter` on top of `vidore_train`.
- 07-02 implementation contract:
  - recommended artifact / dataset naming: `vidore-rag-<source>` or an equivalent standalone standardized-subset name;
  - recommended loader / view naming: a dedicated training-view name such as `vidore_rag_train`;
  - prohibited shortcut: do not use `vidore_train + source_filter` as if it were semantically equivalent to archive `vidore_rag_single`.
- 2026-03-31 local audit findings:
  - raw root: `./data/converted`
  - converted root: `./data/converted`
  - raw `train` rows: `118,195`
  - raw and converted artifacts both expose the same sources: `Infographic-VQA`, `arxiv_qa`, `docvqa`, `pdf`, `tatdqa`
  - all 5 archive `vidore_rag_single` sources are covered, with no missing or extra sources
  - sampled raw rows confirm that the upstream storage is still the `query + image + answer (+ prompt / options / page / answer_type)` view, not the archive `rag_single` text-query-to-image-positive view
  - the converted root is the standardized training root; future subset / rag-view materialization should continue from this layer

---

### VisRAG Subsets

VisRAG is also not a pending family in the current repository. BToks already supports the full `visrag-indomain` and the 6 source-level subset configs below through `visrag_train`.

These entries are also not new upstream data sources. They come from archive-era experiments that split the same `openbmb/VisRAG-Ret-Train-In-domain-data` training set by `source`.

- **HuggingFace**: [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data)
- **Paper**: VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594))

| Config Key | Subset Name | Samples | Task |
|------------|-------------|---------|------|
| visrag-indomain_InfoVQA | InfoVQA | 17,664 | document_retrieval |
| visrag-indomain_ArxivQA | ArxivQA | 25,856 | document_retrieval |
| visrag-indomain_PlotQA | PlotQA | 56,192 | document_retrieval |
| visrag-indomain_SlideVQA | SlideVQA | 8,192 | document_retrieval |
| visrag-indomain_MP-DocVQA | MP-DocVQA | 10,624 | document_retrieval |
| visrag-indomain_ChartQA | ChartQA | 4,224 | document_retrieval |

Note:

- The current project already supports the full `visrag-indomain` and these 6 source-specific subset configs.
- These 6 configs are still `source`-level splits of the same training dataset, not additional HuggingFace datasets.
- Unlike ViDoRe, archive `visrag_single` stays broadly aligned with the full `visrag` pairing semantics. The main change is that the monolithic mixture is split into independently sampled, independently weighted source-specific subsets.
- This group should therefore be understood as "already-covered upstream data + archive mixture-view differences", not as new pending training corpora.
- Current recommended path:
  - first run a local subset audit to confirm coverage of the 6 archive sources and any eval/source-unit risk;
  - then write those source subsets as standalone artifacts / standalone mixture members;
  - keep `source_filter` only as a later optimization candidate after the audit is complete.
- 07-02 implementation contract:
  - recommended artifact naming: `visrag-indomain-<source>` or an equivalent standalone standardized-subset name;
  - training configs should express `InfoVQA / ArxivQA / PlotQA / SlideVQA / MP-DocVQA / ChartQA` as separate mixture members with explicit weights;
  - only evaluate `source_filter_candidate` later if pre-split artifacts prove too costly to store or maintain.
- 2026-03-31 local audit findings:
  - raw root: `./data/converted`
  - converted root: `./data/converted`
  - raw `train` rows: `122,752`
  - raw and converted artifacts both expose the same sources: `ArxivQA`, `ChartQA`, `InfoVQA`, `MP-DocVQA`, `PlotQA`, `SlideVQA`
  - all 6 archive `visrag_single` sources are covered, with no missing or extra sources
  - sampled raw rows confirm that the upstream storage is already semantically aligned with archive `visrag_single`; the remaining difference is source-specific subset materialization and independent weighting
  - the converted root is the standardized training root; source subsets should be written from this layer

---

## Truly Pending Extra Data

### Video Classification

| Config Key | Parser | Samples | Confirmed Source | Paper |
|------------|--------|---------|------------------|-------|
| K700 | Kinetics-700 | ~536,499 (full) | [siyrus/BToks-Kinetics-700](https://huggingface.co/datasets/siyrus/BToks-Kinetics-700) | A Short Note on the Kinetics-700-2020 Human Action Dataset ([arXiv:2010.10864](https://arxiv.org/abs/2010.10864)) |
| Breakfast | Breakfast | 1,556 | [siyrus/BToks-Breakfast](https://huggingface.co/datasets/siyrus/BToks-Breakfast) | The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities (CVPR 2014) |

All hosted on MMEB-V2. Task type: video_classification.

---

### Video Retrieval

| Config Key | Parser | Samples | HuggingFace | Paper |
|------------|--------|---------|-------------|-------|
| MSR-VTT | msrvtt | 9,000; config weight now folds archive `rep1-4` oversampling | [friedrichor/MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT) | MSR-VTT: A Large Video Description Dataset for Bridging Video and Language (CVPR 2016) |
| MSVD | msvd | 1,200; config weight now folds archive `rep1-4` oversampling | [friedrichor/MSVD](https://huggingface.co/datasets/friedrichor/MSVD) | Collecting Highly Parallel Data for Paraphrase Evaluation (ACL 2011) |
| DiDeMo | didemo | 8,395 | [friedrichor/DiDeMo](https://huggingface.co/datasets/friedrichor/DiDeMo) | Localizing Moments in Video with Natural Language ([arXiv:1708.01641](https://arxiv.org/abs/1708.01641)) |

Note: the archive MSR-VTT and MSVD recipe used rep1-4 copies (5 total entries each) to oversample smaller video retrieval datasets. Those copies use the same parser, source data, and transform, so they are not separate training views. The current config no longer keeps repeated entries; it folds the same sampling intent into the main entry `weight` and accepts the small epoch-composition difference from archive slot-based accounting.

---

### Video Moment Retrieval

| Config Key | Parser | Samples | HuggingFace | Paper |
|------------|--------|---------|-------------|-------|
| Charades-STA | Charades-STA_train | ~12,408 | [siyrus/BToks-Charades-STA](https://huggingface.co/datasets/siyrus/BToks-Charades-STA) | TALL: Temporal Activity Localization via Language Query ([arXiv:1705.02101](https://arxiv.org/abs/1705.02101)) |
| QVHighlights | qvhighlight_train | ~8,000+ | [siyrus/BToks-QVHighlight](https://huggingface.co/datasets/siyrus/BToks-QVHighlight) | QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries ([arXiv:2107.09609](https://arxiv.org/abs/2107.09609)) |

Both hosted on MMEB-V2. Task type: video_moment_retrieval.

---

### Video QA

| Config Key | Parser | Samples | Confirmed Source | Paper |
|------------|--------|---------|------------------|-------|
| NEXTQA | nextqa | 34,132 | [Shuaiii/NExT-QA](https://huggingface.co/datasets/Shuaiii/NExT-QA) `nextqa_train.json` plus [rhymes-ai/NeXTVideo](https://huggingface.co/datasets/rhymes-ai/NeXTVideo) `NExTVideo.zip`; official [doc-doc/NExT-QA](https://github.com/doc-doc/NExT-QA) remains the provenance reference | NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions ([arXiv:2105.08276](https://arxiv.org/abs/2105.08276)) |
`NExTQA` source confirmation: the archive parser needs a multiple-choice training CSV with `video`, `question`, `answer`, `qid`, `type`, `a0` through `a4`, and `video_path`. The current implementation uses `Shuaiii/NExT-QA` `nextqa_train.json` as the primary table, with fields `video`, `frame_count`, `width`, `height`, `question`, `answer`, `qid`, `type`, `options`, and `video_path`; conversion preserves the `options` list and no longer writes derived `a0` through `a4` columns, while runtime dynamically expands `options` into `(A)` through `(E)` when building the query. `rhymes-ai/NeXTVideo` `NExTVideo.zip` covers all 3,870 training video paths and has 1,570 extra videos outside the train split. Note that `rhymes-ai/NeXTVideo/train.jsonl` rewrites answers as option letters, so it is not used as the raw annotation for reproducing the archive train parser.

---

### OOD Image Datasets

Out-of-distribution image datasets for evaluating generalization.

| Config Key | Parser | Samples | HuggingFace | Paper |
|------------|--------|---------|-------------|-------|
| Place365 | Place365_train | 226,397 | [zichengsaber/Places365](https://huggingface.co/datasets/zichengsaber/Places365) or local | Places: A 10 Million Image Database for Scene Recognition (TPAMI 2017) |
| Country211 | country211_train | 31,650 | [clip-benchmark/wds_country211](https://huggingface.co/datasets/clip-benchmark/wds_country211) | CLIP: Learning Transferable Visual Models From Natural Language Supervision ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) |
| ScienceQA | scienceQA_train | 12,726 | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (NeurIPS 2022) |
| GQA | GQA_train | 70,000 | [lmms-lab/GQA](https://huggingface.co/datasets/lmms-lab/GQA) or local | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering ([arXiv:1902.09506](https://arxiv.org/abs/1902.09506)) |
| TextVQA | textvqa_train | 34,602 | [facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) | Towards VQA Models That Can Read ([arXiv:1904.08920](https://arxiv.org/abs/1904.08920)) |
| Wiki-SS-NQ | wikissnq_train | 15,293 | Custom (Wiki-SS-NQ) | Wiki-SS-NQ: Wikipedia Screenshot Natural Questions |

---

### OOD Document Datasets

| Config Key | Parser | Samples | HuggingFace | Paper |
|------------|--------|---------|-------------|-------|
| MMLongBench-doc | MMLongBench_doc_train | ~1,091 | [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) | MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations |

Task type: document_understanding.

---

### M-BEIR Datasets

Multi-modal benchmark datasets for universal information retrieval.

- **HuggingFace**: [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)
- **Paper**: One Embedder, Any Task: Instruction-Finetuned Text Embeddings / UniIR: Training and Benchmarking Universal Multimodal Information Retrievers ([arXiv:2311.17136](https://arxiv.org/abs/2311.17136))

| Config Key | Parser | Task ID | Samples | Original Paper |
|------------|--------|---------|---------|----------------|
| oven_it2it_train | mbeir | 8 | 154,460 | Open-domain Visual Entity Recognition (arXiv:2302.11154) |
| EDIS_train | mbeir | 2 | 25,917 | EDIS: Entity-Driven Image Search over Multimodal Web Content (ECIR 2022) |
| FashionIQ_train | mbeir | 7 | 16,176 | Fashion IQ: A New Dataset Towards Retrieving Images by Relative Natural Language Feedback (arXiv:1905.12794) |

Task type: multimodal_retrieval.

---

### Visual Grounding

| Config Key | Parser | Samples | HuggingFace | Paper |
|------------|--------|---------|-------------|-------|
| visual7W_pointing_train | visual7W_pointing_train | 186,188 | [ai2-jackh/visual7w-pointing](https://huggingface.co/datasets/ai2-jackh/visual7w-pointing) or local | Visual7W: Grounded Question Answering in Images ([arXiv:1511.03416](https://arxiv.org/abs/1511.03416)) |

Task type: visual_grounding.

---

## Notes for Adaptation

When adapting these datasets to BToks:

1. **One-dataset rewrite**: New datasets should no longer start by reusing legacy family loaders / contract design. Each dataset first confirms raw schema, origin parser behavior, and target `TrainSample` output, then implements a dataset-owned transform.

2. **Lance Schema**: Choose single-table, dual-table join, or video frame + instruction dual-table layouts from the real raw schema. Conversion is raw-preserving by default; only narrow audited data fixes are exceptions.

3. **Conversion entrypoint**: Business logic lives in `src/vlm2emb/data/conversion/<dataset>.py`; `scripts/convert/train/convert_<dataset>_to_lance.py` only parses arguments, logs, and calls package functions.

4. **Legacy cleanup**: After the new conversion/runtime for a dataset passes validation, remove the matching legacy conversion script, runtime, helper, tests, and docs entry.

5. **Task Types**: Map each dataset to the task semantics required by the current training runtime (document_retrieval, video_classification, video_retrieval, etc.), and explicitly label it as `implemented`, `design_only`, `eval_only`, `pretrain_only`, or `defer`.

---

## Future Download And Audit List

This list is meant for a future user-side local download pass. At the current stage, not every dataset needs a fully confirmed official HuggingFace page; unresolved sources should simply be flagged for manual lookup before download.

### A. Datasets actually enabled in the final archive training config and worth auditing first

Notes:

- This table records objects that appeared in the archive final config or were audited in this round. Some are now migrated into the current production training registry; they remain here to record migration conclusions, row counts, source risks, and legacy-entry cleanup status.
- The "static risk" column records risks inferred from archive parsers, docs, and configs. When an item has completed local sample-level verification, the next-action column records that result directly.

| Dataset | Task Archetype | Current State | Static Risk | HF Status | Next Action |
|---------|----------------|---------------|-------------|-----------|-------------|
| HMDB51 | video classification | implemented / promoted to formal data / frame-unit migrated | low, training path explicitly reads `train` | known | `hmdb51_train` is wired and the formal root is under `./data/converted`; `vlm2vec_train` has 10,710 rows, `data/frames.lance` has 6,436 rows, and the default path does not raw-decode videos |
| UCF101 | video classification | implemented / promoted to formal data / frame-unit migrated | low, training path explicitly reads `train` | known | `ucf101_train` is wired and the formal root is under `./data/converted`; `vlm2vec_train` has 28,747 rows, `data/frames.lance` has 13,320 rows, and the default path does not raw-decode videos |
| MSR-VTT | video retrieval | implemented / promoted to formal data / frame-unit migrated | low, training path explicitly reads `train_9k/train`; local audit found zero video_id overlap between `train_9k` and eval `test_1k` | known | `msrvtt_train` is wired and the formal root is under `./data/converted`; the default split is `train_9k_without_mmeb_v2_eval`, currently with the same 9000 rows as `train_9k`, and `data/frames.lance` has 10,000 rows |
| MSVD | video retrieval | implemented / promoted to formal data / frame-unit migrated | low, training path explicitly reads `train`; local audit found zero video_id overlap between `train` and eval `test` | known | `msvd_train` is wired and the formal root is under `./data/converted`; the default split is `train_without_mmeb_v2_eval`, currently with the same 1200 rows as `train`, and `data/frames.lance` has 1,970 rows |
| DiDeMo | video retrieval | implemented / promoted to formal data / frame-unit migrated | low, training path explicitly reads `train`; local audit found zero video path overlap between `train` and eval `test` | known | `didemo_train` is wired and the formal root is under `./data/converted`; the default split is `train_without_mmeb_v2_eval`, currently with the same 8395 rows as `train`, and `data/frames.lance` has 9,399 rows |
| VATEX | video retrieval | implemented / promoted to formal data / frame-unit migrated / v2 enabled | medium, the official training JSON and video package are available; the old raw-video root is backed up and the formal root has been replaced by a frame-unit root; 3,886 bad-media rows are excluded | known | `vatex_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval` with 22,105 rows, `data/frames.lance` has 22,105 rows, and `data/videos.lance` has 22,105 rows |
| K700 | video classification | implemented / promoted to formal data / frame-unit migrated / v2 enabled | medium, large dataset; the runtime defaults to `train_without_mmeb_v2_eval` and keeps an invalid-filtered `official_train` audit view | known | The current production root is under `./data/converted`; `data/frames.lance`, `data/videos.lance`, `official_train.lance`, and `train_without_mmeb_v2_eval.lance` each have 536,489 rows. Historical audit confirms raw `train.csv` deduplicates to 536,499 `video_path` values, and the 10 unusable original tar members are recorded in `data/exclusions/invalid_videos.lance` and `data/exclusions/bad_media.lance` |
| Breakfast | video classification | implemented / promoted to formal data / frame-unit migrated | medium, the origin parser's `cam01` filter is reproduced; `cereal/cereals` and `salad/salat` naming differences are handled as explicit aliases | known | `breakfast_train` is wired, runtime positives prefer the canonical `label`, and `raw_label` remains an audit field; the formal root is under `./data/converted`, and `data/frames.lance` has 1,989 rows |
| Charades-STA | video moment retrieval | implemented / promoted to formal data / frame-unit migrated | low, training path explicitly reads `Charades_sta_train.txt`; the default training split preserves `official_train` and adds an eval-excluded view | known | `charades_sta_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval`, currently with the same 12408 rows as `official_train`, and `data/frames.lance` has 18,444 rows |
| QVHighlights | video moment retrieval | implemented / promoted to formal data / frame-unit migrated | medium, the local official-named tar `qvhilights_videos.tar.gz` is truncated and fails with EOF, so it is not used as a source; this round uses the expanded directory from the community HF split-video source `yeliudev/VideoMind-Dataset`, with zero missing videos for train/val/test annotations, but byte-level equivalence with the official tar is still unproven | partially known | `qvhighlight_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval`, currently with 7218 rows, and `data/frames.lance` has 25,715 rows; future archive-direct support can audit the local three-part `yeliudev` archive before considering the broken tar |
| ActivityNet Captions | video segment retrieval | implemented / promoted to formal data / frame-unit migrated | medium, the archive has no train parser; this round confirms segment-level text-to-video retrieval semantics, and train split overlap with MMEB-V2 `ActivityNetQA` eval video ids is 0 | known | `activitynet_captions_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval` with 37,421 rows, `data/frames.lance` has 71,787 rows, and `data/videos.lance` has 14,926 rows |
| NExTQA | video QA | implemented / promoted to formal data / frame-unit migrated | medium, the archive train parser uses a multiple-choice `train.csv`; the current `Shuaiii/NExT-QA` annotations and `rhymes-ai/NeXTVideo` package cover the train split, and overlap with MMEB-V2 `NExTQA` eval video ids is 0 | known | `nextqa_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval` with 34,132 rows, and `data/frames.lance` has 3,870 rows |
| Place365 | image classification | implemented / promoted to formal data | medium, the training parser was directory-walk based; current conversion uses explicit split tables | partially known | `place365_train` is wired and the formal root is under `./data/converted`; the default split is `official_train` |
| Country211 | image classification | implemented / promoted to formal data | low, train/eval docs already point to explicit split structure | known | `country211_train` is wired; the default split excludes the MMEB-V2 eval boundary |
| ScienceQA | image QA | implemented / promoted to formal data | low, explicit upstream splits exist | known | `scienceqa_train` is wired; the default training split has 6218 rows |
| GQA | image QA | implemented / promoted to formal data | medium, multiple upstream mirrors/versions may exist; image-aware overlap audit is complete, with zero intersection between train image IDs and MMEB-V2 eval image IDs and zero intersection for `imageId + question` keys | partially known | `gqa_train` is wired; the default split is `official_train_balanced_without_mmeb_v2_eval` with 943000 rows; the 14,305,356-row `official_train_all` remains available as an explicit full-source view; text-only question / answer overlap is high but occurs on different images, so it is not used as the sample-level leakage key |
| TextVQA | image QA | implemented / promoted to formal data | low, explicit upstream splits exist | known | `textvqa_train` is wired; the default training split has 34496 rows after excluding 106 rows by query/answer key |
| Wiki-SS-NQ | image retrieval | implemented / promoted to formal data | medium, training queries overlap MMEB-V2 eval queries, so the default split excludes them by query text; the image join key uses the screenshot filename stem | known | `wikissnq_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval`, currently with 332518 rows; `images.lance` has 1267874 rows and covers all positive docids |
| MMLongBench-doc | visdoc OOD / document QA | implemented / promoted to formal data | high, 469 of the 473 archive-usable single-page rows exactly overlap MMEB-V2 eval queries; the default split can only keep 4 non-overlapping rows | known | `mmlongbench_doc_train` is wired and the formal root is under `./data/converted`; primary tables preserve raw parquet fields and rendered pages are stored in `data/pages.lance` |
| OVEN_it2it_train | multimodal retrieval | implemented / promoted to formal data | medium, comes from M-BEIR and still needs source-unit boundary review against eval `OVEN` | known | `mbeir_train` is wired; the default split is `official_train_without_mmeb_v2_eval`, currently with 143051 rows |
| EDIS_train | multimodal retrieval | implemented / promoted to formal data | medium, comes from M-BEIR while eval also contains EDIS | known | `mbeir_train` is wired; the default split is `official_train_without_mmeb_v2_eval`, currently with 22676 rows |
| FashionIQ_train | multimodal retrieval | implemented / promoted to formal data | medium, comes from M-BEIR while eval also contains FashionIQ | known | `mbeir_train` is wired; the default split is `official_train_without_mmeb_v2_eval`, currently with 10755 rows |
| visual7W_pointing_train | visual grounding | implemented / promoted to formal data | medium, continue reviewing its sample boundary against eval `Visual7W-Pointing` | partially known | `visual7w_pointing_train` is wired and the formal root is under `./data/converted`; the default split is `official_train_without_mmeb_v2_eval`, currently with 93813 rows |

### B. Candidate datasets that appeared in archive experiments but did not survive into the final recipe

| Dataset | Task Archetype | Current State | HF Status | Next Action |
|---------|----------------|---------------|-----------|-------------|
| OVEN_it2t_train | multimodal retrieval | implemented / promoted to formal data | known | `mbeir_train` is wired; the default split has 132399 rows |
| INFOSEEK_it2t | multimodal retrieval | implemented / promoted to formal data | known | `mbeir_train` is wired; the default split has 135613 rows |
| INFOSEEK_it2it | multimodal retrieval | implemented / promoted to formal data | known | `mbeir_train` is wired; the default split has 134330 rows |
| Fashion200K_t2i | multimodal retrieval | implemented / promoted to formal data | known | `mbeir_train` is wired; the default split has 14123 rows |
| Fashion200K_i2t | multimodal retrieval | implemented / promoted to formal data | known | `mbeir_train` is wired; the default split has 12558 rows |

### C. Archive training-view differences (not new upstream data)

The following items are still worth referencing in future training design, but they should not be counted as "new pending datasets":

- `vidore_rag_single`: a different pairing view over the same `vidore/colpali_train_set` source, emphasizing text-to-document-image retrieval.
- `visrag_single`: a source-specific mixture view over the same `openbmb/VisRAG-Ret-Train-In-domain-data` source.
- `MSR-VTT_rep1-4` and `MSVD_rep1-4`: repeated sampling strategy for small video retrieval datasets, not new upstream datasets; the current config folds them into the main entry weights.
- `YouCook2`: do not use the `lmms-lab/YouCook2` eval mirror as a training source; training should use only the audited original `YouCook2` train view.

### D. `lmms-lab` Hugging Face scan candidates (2026-04-03)

Scan scope:

- the currently public 166 datasets under `https://huggingface.co/lmms-lab/datasets`
- only candidates close to the current framework task space were retained: `video classification / retrieval / QA / moment retrieval`, `image QA / retrieval / visual grounding`, `visdoc / document QA`, and `multimodal retrieval`
- we explicitly excluded pure benchmark mirrors, text-only exam sets, audio-first datasets, and mirrors that are already covered by the current pending list

Recommended new candidates to evaluate first:

| Dataset | Suggested task mapping | Current judgment | Source | Notes |
|---------|------------------------|------------------|--------|-------|
| ScreenSpot-v2 | image visual grounding / UI grounding | worth evaluating first | [lmms-lab/ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2) | the HF repo clearly exposes a `train` split; it belongs to the same broad grounding family as `visual7W_pointing_train`, but the target domain is GUI screens, so we should decide whether to add a dedicated screen-grounding subtype |
| ScreenSpot-Pro | image visual grounding / UI grounding | worth evaluating first | [lmms-lab/ScreenSpot-Pro](https://huggingface.co/datasets/lmms-lab/ScreenSpot-Pro) | the HF repo clearly exposes a `train` split; it looks like a stronger same-domain variant of `ScreenSpot-v2`, so they should be evaluated together and either one retained or both sampled hierarchically |
| LLaVA-Video-178K | video QA / video-text instruction | defer: move to a future video-instruction family design | [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | local source is downloaded under `./data/converted`, with processed QA/caption JSON files and many video tarballs; about 1.2T, so it should not use the current retrieval/classification rewrite template |
| EgoIT-99K | multimodal QA / video-text instruction | defer: move to a future egocentric-instruction family design | [lmms-lab/EgoIT-99K](https://huggingface.co/datasets/lmms-lab/EgoIT-99K) | local source is downloaded under `./data/converted`, mixing image/video/audio/segments plus source-specific parquet/JSON; choose the target subtask before conversion |

Not recommended right now as “new training datasets”:

- `ActivityNetQA`: `lmms-lab/ActivityNetQA` currently exposes only `test` parquet plus video zip files, so it behaves like an eval mirror rather than a training source.
- `charades_sta`: `lmms-lab/charades_sta` currently exposes only `test` parquet, and we already have a more complete official `Charades_v1.zip + Charades-STA annotations` path.
- `RefCOCO / RefCOCOplus / RefCOCOg`: the current HF repos expose only `val/test`, not `train`, so they are better treated as eval references.
- `MP-DocVQA`: the current HF repo exposes only `val/test`, and the framework already has a `visrag-indomain_MP-DocVQA` path.
- `VideoSearch`: the repo only exposes milestone-style `test` parquets and no train split.
- `EMMA`: the public content is test-oriented and looks closer to an evaluation set than a train source.
- `FVQA`: it does contain `train/test` parquet files, but the current modality tags look text-heavy and the repo does not surface a clear raw image pack, so the payoff is less clear than `ScreenSpot-*`.

Already covered or highly redundant with the current list:

- `GQA`, `textvqa`, `ScienceQA`, `NExTQA`, and `YouCook2`: equivalent train/candidate objects already exist in the current pending list, so the `lmms-lab` mirrors should not open new download items.
- `VATEX`: the current local `lmms-lab` copy only confirms eval/test mirror evidence and cannot be used as a training source. Training uses the official training JSON and matching video package from `qingy2024/VaTeX`; conversion/runtime code exists, and the production root has been rerun to write `data/frames.lance` and enter the v2 enabled config.
- `DocVQA`: `lmms-lab/DocVQA` does expose `train/validation/test`, but the framework already contains an `MMEB` training version of `DocVQA`, so the incremental value is low unless we later want source comparison or replacement.
- `EgoLife`: although it contains large-scale raw videos, its primary label is `video-text-to-text` with Chinese-centric supervision, so it looks more like a future standalone initiative than a direct drop-in to the current mixture.

Current local progress (2026-04-20):

- `lmms-lab` local mirrors already present: `GQA`, `NExTQA`, `VATEX`, `YouCook2`
- Newly completed `lmms-lab` downloads:
  - `ScreenSpot-v2`
  - `ScreenSpot-Pro`
  - `DocVQA`
  - `textvqa`
  - `ScienceQA`
  - `EgoIT-99K`
  - `LLaVA-Video-178K`
- These downloads are rooted at `./data/converted`; no download tmux session was still running during this pass
- Final status for large conversions started during this change:
  - `M-BEIR` formal root is `./data/converted`, about 7.9G, with runtime sampling complete
  - `Kinetics-700` formal root is `./data/converted`, about 687G, with 536489 rows in the default training split
  - `QVHighlights` formal root is `./data/converted`, about 110G, with 7218 rows in the default training split
  - `siyrus/BToks-Breakfast` formal root is `./data/converted`, about 3.8G, with runtime sampling complete
- Downloads added after that pass and now source-audited:
  - `NExTQA` annotations are under `./data/converted`, and the video zip is under `./data/converted`; all 3,870 training videos are covered by the zip, with zero MMEB-V2 eval video-id overlap
  - `YouCook2` train parquet is under `./data/converted`, and raw split-tar videos are under `./data/converted`; the parquet differs from the official train annotation by only one video / seven segments, has zero MMEB-V2 eval youtube-id / segment-stem overlap, and the split tar covers all 1,332 video members required by the parquet
  - `ActivityNet Captions` is under `./data/converted`; the 10,009 train video ids have zero overlap with MMEB-V2 `ActivityNetQA` eval video ids, and the split tar covers all 14,926 unique video ids across train, val1, and val2
