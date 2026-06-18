# 待适配训练数据集

本文档梳理了 Qwen3vl 归档实验（the expanded public training source survey）里出现过的数据来源，并将它们区分为两类：

- `archive 参考但当前已覆盖`：当前 BToks 已经有正式训练 loader / registry，只是 archive 曾使用过不同的配对视图、source 拆分或采样方式；
- `真正待适配的额外数据`：archive 使用过、但当前 BToks 还没有正式纳入训练 registry 的候选数据集。

本页的重点是第二类，因为它们才是真正可能帮助提升训练性能、同时又需要做 split / test overlap 审计的新增候选。

> 当前策略：扩展训练数据集按“每个数据集一个 rewrite task”推进。每个 task 先确认 raw schema、origin parser 行为、旧实现修正和目标 `TrainSample` 输出，再重写 conversion/runtime；替换完成后删除对应旧脚本、旧 runtime 和旧文档入口。

> 当前仓库已正式覆盖的训练家族：`MMEB`、`ViDoRe`、`VisRAG`、`LLaVA-Hound`、视频分类 / 检索 / 时刻检索独立 runtime、OOD / VQA 独立 runtime、`M-BEIR`、`Visual7W-Pointing`。第二轮完整训练配置位于 `configs/datasets/expanded_train_datasets_v2.yaml`，它继承 v1 并追加当前可真实取样的 `VizWiz`、`RefCOCO`、`RefCOCO-Matching`、`VATEX` 和 `K700`。`K700` 已完成 frame-unit 重转、正式 root 替换和取样 smoke，不再作为 pending frame-unit rerun 项。已覆盖家族本身不应再被视为 pending family。

> 默认扩展训练权重：`configs/datasets/expanded_train_datasets_v1.yaml` 在移出 `K700` 后使用较保守的 `image / video / visdoc ≈ 46.3 / 17.2 / 31.8`，类内按 `samples^0.78` 分配。`alpha=0.78` 让所有已启用 entry 自然保留至少 `0.1` 权重，不需要额外最小权重兜底。这里的大类比例只是目标范围。GQA 默认使用官方 balanced train split，当前默认样本池约 6.5M 行，在 `all_exhausted` sampler 下预计单 epoch 约 14.1M 行，低于当前可接受的 15M 上限；14.3M 行的 GQA `train_all` 仍作为显式 full-source view 保留。

> 旧边界结论：本轮训练 rewrite 不再使用 `render.py`、`ReplaceDefaultTransform`、`materialize_training_view` 或 MMEB-V2 eval parser 内部旧 `render_*` / `_materialize_*` 命名来构造训练样本。`render.py` 与 `vlm2emb.data.contracts` 均已删除；MMEB-V2 parser 内部文本装配已改为 `compose_*` / `format_*` 命名，item 构造函数已改为 `_build_*` 命名。

## 2026-04-20 统一盘点口径

本节是 `expand-training-dataset-catalog` 的初始盘点结果。它把旧 pending 文档、设计占位卡片、当前正式训练配置、archive parser 线索、`lmms-lab` 扫描候选和模块 C 推荐清单放到同一套状态词表里。这里的目标不是一次性实现所有数据集，而是为后续逐数据集 rewrite 建立稳定入口。

### 状态与覆盖词表

| 字段 | 取值 | 含义 |
|------|------|------|
| `status` | `implemented` | 当前已有正式训练 loader / registry 或正式 eval 支持 |
| `status` | `covered_by_existing_family` | 已由 `MMEB`、`MMEB-V2` eval、`ViDoRe`、`VisRAG` 或 `LLaVA-Hound` 覆盖，不应重复实现 |
| `status` | `design_only` | 只有设计卡片，暂无正式 loader / registry / 本地 Lance 样例 |
| `status` | `pending_download` | 尚未本地下载，后续需用户下载 |
| `status` | `downloaded_pending_audit` | 本地源数据已存在，但尚未完成 schema / split / overlap 审计 |
| `status` | `audit_blocked` | 已有明确风险，需先解决 split、source、license 或样本覆盖问题 |
| `status` | `train_candidate` | 审计通过或风险足够低，可进入单数据集 rewrite |
| `status` | `eval_only` | 主要适合评测，不作为训练候选 |
| `status` | `pretrain_only` | 适合大规模预训练，不进入当前微调训练数据主线 |
| `status` | `defer` | 暂缓，等待更明确的任务定义或本地资源 |
| `coverage` | `none` | 当前没有覆盖 |
| `coverage` | `train_implemented` | 已有正式训练实现 |
| `coverage` | `eval_implemented` | 已有正式评测实现 |
| `coverage` | `design_card_only` | 只有设计卡片 |
| `coverage` | `same_source_different_view` | 同源不同训练视图 |
| `coverage` | `mirror_or_duplicate` | 镜像或重复来源 |

### 已读取的信息来源

| 来源 | 已提取内容 | 对后续 task 的意义 |
|------|------------|-------------------|
| 本页旧 pending 表 | 现有 pending、已下载、待审计、archive 训练视图差异和 `lmms-lab` 扫描候选 | 作为逐数据集 task 的初始候选池 |
| `docs/zh/datasets/train/ood/` 与 `docs/en/datasets/train/ood/` | `Place365`、`Country211`、`ScienceQA`、`GQA`、`TextVQA`、`Wiki-SS-NQ`、`MMLongBench-doc` 的设计卡片和旧 loader 线索 | 区分 `design_only`、已存在旧 runtime 和真正未实现对象 |
| `docs/zh/datasets/train/mbeir/` 与 `docs/en/datasets/train/mbeir/` | `OVEN`、`EDIS`、`FashionIQ` 的目标 schema 和 M-BEIR task id | 后续 M-BEIR rewrite 的 source/task 映射入口 |
| `docs/zh/datasets/train/visual-grounding/` 与 `docs/en/datasets/train/visual-grounding/` | `visual7W_pointing_train` 的旧 Lance schema、split 和 runtime 线索 | 后续 visual grounding rewrite 的旧行为参照 |
| `configs/datasets/mmeb_train.yaml` | 当前正式 `MMEB`、`ViDoRe`、`VisRAG`、`LLaVA-Hound` 训练主路径 | 用于确认哪些推荐数据集已经由基础家族覆盖 |
| `configs/datasets/expanded_train_datasets_v1.yaml` | archive-era 扩展训练配方、当前可加载训练 entry，以及少量等待下载和审计的空 `type` 占位；旧 `lance_*` / text-format 入口只作为历史清理对象记录 | 空占位用于保留未来接入位置，正式训练加载方应跳过 `type` 为空的对象 |
| the public training source survey | origin parser 文件名和训练视图线索 | 每个数据集 rewrite 时逐个深入，不在本轮一次性复写旧代码 |
| 模块 C 推荐清单 | Classification、Caption / Image-Text Pairs、Retrieval、VQA、Visual Grounding 全量推荐对象 | 用于给每个推荐对象标注 train/eval/pretrain/defer，不等于全部实现 |

### 当前正式覆盖边界

| 家族 | 覆盖对象 | 当前 status / coverage |
|------|----------|------|
| `MMEB` | `ImageNet_1K`、`N24News`、`HatefulMemes`、`VOC2007`、`SUN397`、`OK-VQA`、`A-OKVQA`、`DocVQA`、`InfographicsVQA`、`ChartQA`、`Visual7W`、`VisDial`、`CIRR`、`VisualNews_t2i`、`VisualNews_i2t`、`MSCOCO_t2i`、`MSCOCO_i2t`、`NIGHTS`、`WebQA`、`MSCOCO` | `implemented`；coverage = `train_implemented` |
| `MMEB-V2 eval` | `ImageNet-A`、`ImageNet-R`、`ObjectNet`、`Country211`、`Place365`、`ScienceQA`、`GQA`、`TextVQA`、`VizWiz`、`FashionIQ`、`Wiki-SS-NQ`、`OVEN`、`EDIS`、`RefCOCO`、`Visual7W-Pointing`、`UCF101`、`HMDB51`、`MSR-VTT`、`DiDeMo` 等 eval parser | `eval_only`；coverage = `eval_implemented`，是否训练需另行判定 |
| `ViDoRe` | `colpali_train_set` full view 与 `arxiv_qa`、`pdf`、`docvqa`、`tatdqa`、`Infographic-VQA` source view | `implemented`；coverage = `same_source_different_view` |
| `ViDoRe RAG` | 同一 `colpali_train_set` 上的 `vidore_rag_train` source-specific RAG view | `implemented`；coverage = `same_source_different_view` |
| `VisRAG` | `visrag-indomain` full view 与 `InfoVQA`、`ArxivQA`、`PlotQA`、`SlideVQA`、`MP-DocVQA`、`ChartQA` source view | `implemented`；coverage = `same_source_different_view` |
| `LLaVA-Hound` | `video-caption-300k`、`video-caption-300k-video`、`video-qa-240k` | `implemented`；coverage = `train_implemented` |
| `SmthSmthV2` | `ssv2_train` 已实现 frame-unit runtime / converter；正式根已写入 `data/videos.lance` + `data/frames.lance`，默认训练路径不 raw decode | `implemented`；coverage = `train_implemented`；media 状态 = `frame_unit_formal` |
| `HMDB51` | `hmdb51_train` 已实现 frame-unit runtime / converter；正式根已写入 `data/videos.lance` + `data/frames.lance`，默认训练路径不 raw decode | `implemented`；coverage = `train_implemented`；media 状态 = `frame_unit_formal` |
| `UCF101` | `ucf101_train` 已实现 frame-unit runtime / converter；正式根已写入 `data/videos.lance` + `data/frames.lance`，默认训练路径不 raw decode | `implemented`；coverage = `train_implemented`；media 状态 = `frame_unit_formal` |

### 基础家族复核结论

| 对象 | 当前实现边界 | 对扩展目录的处理 |
|------|--------------|------------------|
| `MMEB-train` | `mmeb_train` 使用 raw-preserving parquet + `data/images/<subset>.lance` 图像表；runtime 继承 `SampleLanceDataset`，默认 transform 直接输出 `TrainSample`，自定义 `transform` 会替换默认函数，不再经过 legacy formatting path | 已覆盖的 classification、retrieval、VQA、caption / grounding 子集标记为 `covered_by_existing_family`；coverage 可记录为 `train_implemented`，不创建重复训练 family |
| `MMEB-V2 eval` | `mmeb_v2` 是 eval 边界，使用 eval loader 和 parser registry；parser 内部文本装配使用 `compose_*` / `format_*` 命名，不作为训练 runtime 主路径 | `ImageNet-A/R`、`ObjectNet`、`Country211`、`FashionIQ eval` 等未启用对象按 `eval_only` 或 placeholder 处理；`RefCOCO` 和 `RefCOCO-Matching` 已进入 `refcoco_train` |
| `ViDoRe` | `vidore_train` 读取 `data/train.lance` full view；`vidore_rag_train` 表达 archive RAG view；两者都是 dataset-owned transform，没有旧 eval/schema helper 作为训练入口 | 保留 full view 与 5 个 source view 的覆盖关系，RAG view 标记为 `same_source_different_view`，不再把这些 source 当作新上游数据源 |
| `VisRAG` | `visrag_train` 独立于 ViDoRe 描述，读取 `data/train.lance`，按 `source` 注入训练提示并输出 `TrainSample`；旧 visdoc helper 不再作为训练入口 | 保留 full view 与 6 个 source view 的覆盖关系，source-specific 配置只代表同源不同采样成员 |
| `LLaVA-Hound` | `llavahound_train` 使用 instruction 表 `data/video_instruction/<subset>.lance` 和帧表 `data/train_300k.lance`；帧表字段为 `video_id`、`frame_idx`、`image`，runtime 以原始 `video` 字段 join 帧表并输出 `TrainSample` | 扩展目录只保留当前实现；旧脚本清理后不再保留另一条可被误认为等价的视频 instruction 训练入口 |

### 设计占位卡片盘点

| 家族 | 数据集 | 当前 status / coverage | 旧代码线索 | 下一步 |
|------|--------|--------------|------------|--------|
| OOD | `Country211`、`Place365`、`ScienceQA`、`GQA`、`TextVQA` | `implemented` / `train_implemented` | 已迁移到单数据集 conversion/runtime | 持续维护 row count、schema 和 eval overlap 记录 |
| OOD | `Wiki-SS-NQ` | `implemented` / `train_implemented` | 已迁移到 `wikissnq_train`；全量转换、join 覆盖和 runtime 抽样已完成 | 默认训练 split 为 `official_train_without_mmeb_v2_eval` |
| OOD | `MMLongBench-doc` | `implemented` / `train_implemented` | 1091 原始行中 473 行可复现 archive 单页训练视图；其中 469 行与 MMEB-V2 eval query 重叠 | 默认训练 split 仅保留 4 条非重叠样本，archive 复现时才使用 `official_train` |
| M-BEIR | `OVEN`、`EDIS`、`FashionIQ`、`Fashion200K`、`INFOSEEK` | `implemented` / `train_implemented` | 已迁移到 `mbeir_train`；全量转换已完成 | 持续复核与 MMEB-V2 eval 的排除边界 |
| 视觉定位 | `visual7W_pointing_train` | `implemented` / `train_implemented` | 已迁移到 `visual7w_pointing_train`，不再使用旧 runtime / text-format path | 继续把 `ScreenSpot-*` 作为后续单独 task |

### 模块 C 推荐清单拆分

| 类别 | 已覆盖或不重复实现 | 待单数据集判定 / rewrite | 当前不进入微调训练主线 |
|------|--------------------|--------------------------|------------------------|
| Classification | `ImageNet-1K`、`SUN397`、`VOC2007`、`N24News`、`HatefulMemes` 已由 `MMEB` 覆盖；`ImageNet-A/R`、`ObjectNet` 更偏 eval；`SmthSmthV2`、`HMDB51`、`UCF101`、`Kinetics-700`、`Breakfast`、`Place365`、`Country211` 已有训练 runtime | 无 | 无 |
| Caption / Image-Text Pairs | `MSCOCO` 已由 `MMEB` 覆盖；`LLaVA-Hound` 已实现 | `Flickr30K`、`SBU Captions`、`WebVid-2.5M/10M` 需先判定用途 | `CC3M`、`CC12M`、`LAION-400M/5B` 暂按 `pretrain_only` / `defer` |
| Retrieval | `MSCOCO`、`VisualNews`、`VisDial`、`CIRR`、`WebQA`、`NIGHTS` 已由 `MMEB` 覆盖；`ViDoRe`、`VisRAG`、`MSR-VTT`、`MSVD`、`DiDeMo`、`Charades-STA`、`QVHighlights`、`ActivityNet Captions`、`YouCook2`、`M-BEIR`、`Wiki-SS-NQ` 已实现 | 无 | `FashionIQ eval` 保持 eval-only |
| VQA | `OK-VQA`、`A-OKVQA`、`DocVQA`、`InfographicVQA`、`ChartQA`、`Visual7W` 已由 `MMEB` 或文档检索家族覆盖；`ScienceQA`、`GQA`、`TextVQA`、`VizWiz`、`NExTQA`、`MMLongBench-doc` 已实现 | 无 | 无 |
| Visual Grounding | `MSCOCO` 和部分 `RefCOCO` 类对象已有 eval / MMEB-V2 线索；`visual7W_pointing_train` 已实现；`RefCOCO` 与 `RefCOCO-Matching` 已完成本地 conversion/runtime/config/tests | `ScreenSpot-v2`、`ScreenSpot-Pro` | `RefCOCO+ / RefCOCOg` 当前不进入 v2 enabled scope |

### 长期候选池

以下候选不进入当前默认训练 rewrite。它们只作为后续单数据集任务的入口，必须先确认 official source、license、schema、split 和本地路径，再决定是否转换。

| 数据集 | 用途 | 本地路径状态 | 当前处理 |
|--------|------|--------------|----------|
| `LaSCo` | 组合 / 语言引导检索候选 | 未在 `./data/converted` 找到明确目录 | `pending_download`，下载前需确认官方源和 license |
| `SNLI-VE` | image+text -> label | 已见 `./data/converted` | 后续可单独审计 train/dev/test、图像来源和 license |
| `NLVR2` | 多图视觉推理 | 未在本地找到明确目录 | `pending_download`，下载前需确认官方源和 split |
| `VCR` | image+text QA / rationale | 未在本地找到明确目录 | `pending_download`，且需先确认使用条款 |
| `SOP` / Stanford Online Products | image retrieval | 已见 `./data/converted` | 后续可审计 class / product split 与检索目标 |
| `DeepFashion In-Shop` | image retrieval | 已见 `./data/converted` | 后续可审计 train/gallery/query split 与 FashionIQ / Fashion200K overlap |
| `Flickr30K Entities` | grounding / phrase localization | 未在本地找到明确目录 | `defer`，需先确认训练目标而不是直接进入默认训练 |
| `Visual Genome` | grounding / region / relation | 未在本地找到明确目录 | `pending_download`，需明确只用哪个任务视图 |
| `Docmatix` | document / visdoc | 未在本地找到明确目录 | `pending_download`，需确认 license 和页面媒体 schema |

超大或合成来源不阻塞当前微调 rewrite：

| 数据集 | 当前策略 | 本地路径状态 |
|--------|----------|--------------|
| `LAION`、`DataComp`、`Recap-DataComp` | `pretrain_only`，需要单独采样、过滤、license 和安全策略 | 本地只见 `./data/converted`，不是完整 LAION / DataComp |
| `CC3M`、`CC12M`、`WebVid`、`HowTo100M`、`InternVid` | `pretrain_only` 或大规模视频预训练策略 | 本地只见 Unite 派生的 `WebVid-CoVR` 和 `InternVid-FLT`，不等于官方全量源 |
| `MegaPairs`、`MagicLens`、`mmE5 synthetic` | 合成 / 派生训练策略，需单独评估生成方式和去重 | 未在本地找到明确目录 |

### 单数据集 rewrite 模板

每个进入实现范围的数据集都按以下顺序处理：

1. 读取 raw source 的真实 schema、split、样本和媒体字段。
2. 调查 origin parser 如何构造 query、positive、negative、媒体输入和 instruction。
3. 调查仓库旧实现曾经额外修正的问题，只保留仍然必要且能解释清楚的修正。
4. 明确所有 split view 的来源、row count、是否指向同一实际 record，以及默认训练 split 是否需要排除 MMEB-V2 eval 样本。
5. 确认目标 `TrainSample` 输出，包括 text 格式、visual token 位置、换行规则、optional negative 语义和 metadata 保留方式。
6. 在 `src/vlm2emb/data/conversion/<dataset>.py` 写 raw-preserving conversion，脚本 wrapper 只处理参数。
7. 在 dataset runtime 中实现默认 transform，并让 config 可表达默认值。
8. 做转换 smoke、runtime 抽样、坏行附近样本验证、配置加载和 overlap 检查。
9. 删除该数据集被替换的旧脚本、旧 runtime、旧 helper、旧 docs 入口和旧测试。

### 首批迁移经验与后续规则

首批视频分类迁移暴露出一些会在更复杂数据中反复出现的问题：

- official raw、VLM2Vec metadata、archive train view 和旧 Lance root 可能不是同一视图；文档必须分别记录 unique media 数、split row 数和 archive train row 数。
- 多组 split 可能指向同一实际 record。新转换应将每个 split 写成主读表 `data/<split>.lance`，共享图像、视频、帧或 record 放入 side table，通过稳定 join key 关联。
- 默认训练入口不应静默改写原始 split。如果需要剔除 MMEB-V2 测试样本，应新增明确命名的训练 split，例如 `train_without_mmeb_v2_eval` 或数据集等价命名。
- MMEB-V2 是 Tiger-Lab 手动整理的复合 benchmark，可能直接使用上游 test split，也可能使用特定整理 split；所有相关训练候选都需要做样本级 overlap 审计。
- 如果无法定义可靠的 record key、media key 或 query key 来剔除 MMEB-V2 eval 样本，该数据集应保持 `downloaded_pending_audit` 或 `audit_blocked`，不能进入默认训练 mixture。
- 旧脚本中的 manifest / dataset info / contract id 不是 runtime 必需字段；新转换优先保留原始字段，只将大媒体写入 Lance side table。
- 坏行、缺失 positive 或局部数据质量问题应优先在 conversion 中做窄范围修复或剔除，runtime 不应静默跳过训练语义错误。
- 大规模媒体转换从一开始就应暴露 `num_workers`、batch size、Lance 文件大小和索引创建参数，避免后续全量转换时返工。

### 旧引用清理清单

| 类型 | 当前命中 | 本轮处理策略 | 后续 owner / 删除条件 |
|------|----------|--------------|-----------------------|
| 旧 MMEB eval conversion 脚本写 `manifest.json` / 使用 `vlm2emb.data.contracts` | 旧 `scripts/convert/mmeb_conversion_common.py`、`scripts/convert/mmeb_registry.py`、`scripts/convert/check_media_text_conflicts.py`、`scripts/convert/validate_mmeb_conversion.py` 已移除；当前维护入口是 `src/vlm2emb/data/conversion/mmeb_v2/` 与 `scripts/convert/eval/mmeb_v2.py` | 新训练 conversion 不再生成 manifest / README / dataset info；MMEB-V2 eval conversion 走维护中的 conversion package | `cleanup-vlm2emb-data`；`vlm2emb.data.contracts` 已删除 |
| 旧训练 runtime 使用 `render`、`ReplaceDefaultTransform`、`TrainLanceDataset` | `image_classification.py`、`image_retrieval.py`、`video_moment_retrieval.py` 已删除；`visual7w_pointing.py` 已重写；`TrainLanceDataset` 不再导出；旧 `render.py` 兼容 wrapper 已删除 | 训练路径统一为 `SampleLanceDataset` + dataset-owned transform；本轮训练 rewrite 不再使用 `render.py`、`ReplaceDefaultTransform` 或 `TrainLanceDataset` | 已由 `cleanup-vlm2emb-data` 删除旧训练 wrapper；MMEB-V2 parser 内部旧 `render_*` 命名已改为 `compose_*` / `format_*` |
| 旧配置仍引用 `lance_*` 和旧 text-format 参数 | `configs/datasets/expanded_train_datasets_v1.yaml` 的扩展数据段 | 作为 archive-era 行为样本和清理对象，不作为新默认配置 | `cleanup-vlm2emb-data`；当生产配置和示例命令不再引用旧注册名或旧 text-format 参数后，只保留归档说明或删除 |
| 旧文档仍写旧配置入口 | 当前训练数据文档未发现旧 text-format 配置入口；`render.py` 兼容 wrapper 已删除；MMEB-V2 parser 内部旧命名已收敛为 `compose_*` / `format_*` | 不把 `render.py` 写成训练入口；MMEB-V2 eval parser 文本装配只作为 parser-owned transform 内部实现 | 已由 `cleanup-vlm2emb-data` 清理 |
| `materialize_training_view` | 当前训练路径未命中精确符号 | 无需迁移该符号；继续防止重新引入 | `cleanup-vlm2emb-data`；受支持代码和测试中不得重新出现运行时依赖 |
| `vlm2emb.data.contracts` | 旧 MMEB eval conversion 脚本、旧数据接入 workflow 和 contract/catalog 测试引用已删除 | 训练数据 rewrite 不新增依赖；不再保留 catalog / manifest / stable id 兼容包 | 已由 `cleanup-vlm2emb-data` 删除 |
| MMEB-V2 eval parser 内部旧 `render_*` / `_materialize_*` 命名 | `src/vlm2emb/data/datasets/mmeb_v2/parsers/` 已无旧内部命名 | parser-owned transform 入口保持不变，内部文本装配函数使用 `compose_*` / `format_*`，item 构造函数使用 `_build_*` | 已由 `cleanup-vlm2emb-data` 清理；继续保留 eval 输出稳定性测试 |

## 当前结论与后续触发点

### 1. 当前只记录计划，不直接扩充训练集

本页中真正待适配的额外数据目前应被视为：

- `待下载`：未来由用户下载到本地；
- `待审计`：下载完成后再做样本级分析；
- `待决策`：审计通过后才决定是否加入训练 mixture。

### 2. 后续本地审计的统一检查项

当数据集下载到本地后，需要统一完成以下检查：

1. 原始 split 是否清晰，训练侧是否真的只使用 `train`，而不是混入 `val` / `test`
2. 与当前 eval 数据是否存在样本级重叠，而不只是“来源相同但 split 不同”
3. archive-era parser 是否直接引用了 `vlm2vec_eval`、`MMEB_Test_Instruct`、`MMEB-V2` 等评测产物路径
4. 多组 split 是否指向同一实际 record；若是，转换产物是否需要将 split view 作为主读表
5. 是否需要建立剔除 MMEB-V2 eval 样本的专用训练 split，以及可用的 record/media/query overlap key 是什么
6. 数据是否需要额外的 provenance、license、citation、sampling policy 字段
7. 该数据集应映射到哪一种 task archetype，而不是继续扩展 family 抽象

### 3. 归档文件下载与读取策略

后续所有 `tar`、`tar.gz`、`tgz` 和分卷 tar 数据默认保持归档边界，不全量解包成文件树。转换和 runtime 审计优先建立 archive member index，记录 `archive_path`、`member_path`、stable media key、split / label / query metadata，再按需读取 member 内容。

对大规模 `tar.gz` / `tgz`，不强制直接读压缩流。为了并行效率和可随机访问性，可以先把压缩 tar 一次性规范化为未压缩 `.tar`，再基于 `.tar` 建 index 和执行抽样 / 转换。这个过程仍然保留归档形态，不生成大量小文件。

只有在官方说明明确要求目录结构、未压缩 tar 仍无法满足训练性能、标准工具无法可靠读取 member，或用户明确要求时，单数据集 task 才允许解包成文件树。若发生解包，必须记录原因、目标 layout、是否保留原始归档，以及如何避免大量小文件影响训练和校验。

### 4. v1 视频媒体状态重新核对

2026-04-28 按 `configs/datasets/expanded_train_datasets_v1.yaml` 和正式数据根重新核对后，v1 视频数据集的“实现状态”和“媒体表状态”必须分开记录：runtime / converter 已实现不代表当前正式根已经完成 frame-unit 迁移。2026-04-29 决定将 `K700` 从 v1 默认扩展配置移出，放入 v2 待 frame-unit 重转范围，因此它不再阻塞 v1 先跑。2026-05-08，`K700` 已完成 frame-unit 重转、正式 root 替换和 v2 配置接入。

| 数据集 | runtime | 当前正式根媒体状态 | 当前行数或媒体数 | 结论 |
|--------|---------|--------------------|------------------|------|
| `video_caption_300k` / `video_caption_300k-video` / `video_qa_240k` | `llavahound_train` | 独立帧表 `data/train_300k.lance` | 8,402,603 帧行 | 不属于 frame-unit 迁移范围 |
| `SmthSmthV2` | `ssv2_train` | `data/frames.lance` + `data/videos.lance` | 193,690 frame units | 正式根已完成 frame-unit 迁移；默认路径不 raw decode |
| `HMDB51` | `hmdb51_train` | `data/frames.lance` + `data/videos.lance` | 6,436 frame units | staging 结果已 promote 到正式根；默认路径不 raw decode |
| `UCF101` | `ucf101_train` | `data/frames.lance` + `data/videos.lance` | 13,320 frame units | staging 结果已 promote 到正式根；默认路径不 raw decode |
| `MSR-VTT` | `msrvtt_train` | `data/frames.lance` + `data/videos.lance` | 10,000 frame units | 正式根已完成 frame-unit 迁移 |
| `MSVD` | `msvd_train` | `data/frames.lance` + `data/videos.lance` | 1,970 frame units | 正式根已完成 frame-unit 迁移 |
| `DiDeMo` | `didemo_train` | `data/frames.lance` + `data/videos.lance` | 9,399 frame units | 正式根已完成 frame-unit 迁移 |
| `YouCook2` | `youcook2_train` | `data/frames.lance` + `data/videos.lance` | 10,330 frame units | 正式根已完成 frame-unit 迁移 |
| `ActivityNet-Captions` | `activitynet_captions_train` | `data/frames.lance` + `data/videos.lance` | 71,787 frame units / 14,926 videos | 正式根已完成 segment frame-unit 迁移 |
| `K700` | `kinetics700_train` | `data/frames.lance` frame-unit layout + `data/videos.lance` audit view | 536,489 videos | 已移出 v1并在 v2 接入；正式根已替换为 frame-unit root |
| `Breakfast` | `breakfast_train` | `data/frames.lance` + `data/videos.lance` | 1,989 frame units | 正式根已完成 full-video frame-unit 迁移 |
| `Charades-STA` | `charades_sta_train` | `data/frames.lance` + `data/videos.lance` | 18,444 frame units / 6,672 videos | 正式根已完成 mixed-view frame-unit 迁移 |
| `QVHighlights` | `qvhighlight_train` | `data/frames.lance` + `data/videos.lance` | 25,715 frame units / 10,148 videos | 正式根已完成 mixed-view frame-unit 迁移 |
| `NExTQA` | `nextqa_train` | `data/frames.lance` + `data/videos.lance` | 3,870 frame units | 正式根已完成 full-video frame-unit 迁移 |

### 5. 当前已确认的风险结论

- `Breakfast`：已完成本地样本级核验并接入 `breakfast_train`。archive 训练 parser 的目录遍历和 `cam01` 排除行为已经被显式 split 表替代；`cereal/cereals` 与 `salad/salat` 命名差异作为可 review 的 alias 修正保留。训练 runtime 的 positive 优先使用 canonical `label`，`raw_label` 只用于 metadata 审计和缺失兜底。`SmthSmthV2`、`HMDB51`、`UCF101`、`Kinetics-700`、`Place365` 和 `Country211` 已按同一规则检查，除 `Breakfast` 外没有发现源标签与训练标签分叉。
- `SmthSmthV2`、`HMDB51`、`UCF101`：已完成本地样本级核验并接入 `ssv2_train`、`hmdb51_train`、`ucf101_train`；训练侧默认使用 VLM2Vec train 视图，official split 作为可选训练/测试视图保留。当前原则下，默认训练配置不应现场解码原始视频，缺少 `data/frames.lance` 时应失败，只有显式 raw-video 兼容配置可以使用 legacy fallback。三者的 frame-unit 根均已 promote 到正式路径。
- `MSR-VTT`、`MSVD`、`DiDeMo`、`Charades-STA`、`QVHighlights`：已完成本地样本级核验并接入独立 runtime，且已完成 frame-unit 正式根迁移。`MSR-VTT`、`MSVD`、`DiDeMo`、`Charades-STA`、`QVHighlights` 的 `data/frames.lance` 分别为 10,000、1,970、9,399、18,444、25,715 行。`MSR-VTT`、`MSVD`、`DiDeMo` 和 `Charades-STA` 的默认训练 split 与对应 official train 行数一致；`QVHighlights` 当前使用已展开的社区分卷视频目录，annotation 覆盖的视频缺失数为 0。
- `NExTQA`：已接入 `nextqa_train` conversion/runtime/config/tests，并完成 frame-unit 正式根迁移。正式根目录位于 `./data/converted`，`official_train` 和 `official_train_without_mmeb_v2_eval` 均为 34,132 行，`data/frames.lance` 与 `data/videos.lance` 均为 3,870 行；与 MMEB-V2 `NExTQA` eval 的 1,000 个视频重叠为 0。runtime 还原 archive 多选 query 模板，positive 为正确答案文本，negative 为空文本和空媒体。
- `YouCook2`：已接入 `youcook2_train` conversion/runtime/config/tests，并完成 frame-unit 正式根迁移。转换产物位于 `./data/converted`，`official_train` 和 `official_train_without_mmeb_v2_eval` 均为 10,330 行，`data/frames.lance` 为 10,330 行；与 MMEB-V2 `YouCook2` eval 的 youtube id / segment stem 重叠为 0。runtime 使用原始 train view 的 caption 做 text-to-video query，positive 来自上游 row-level `binary_frames` 还原出的片段 frame unit，不再使用完整视频作为 positive；每个片段最多保留 8 帧。
- `ActivityNet Captions`：已确认采用 segment-level text-to-video retrieval 语义，并接入 `activitynet_captions_train` conversion/runtime/config/tests，且已完成 frame-unit 正式根迁移。正式根目录位于 `./data/converted`，`official_train` 为 37,421 行，`official_val1` 为 17,505 行，`official_val2` 为 17,031 行，`official_train_without_mmeb_v2_eval` 为 37,421 行，`data/frames.lance` 为 71,787 行，`data/videos.lance` 为 14,926 行；与 MMEB-V2 `ActivityNetQA` eval 的 614 个 video ids 重叠为 0。每个 positive segment 最多保留 8 帧。
- `VizWiz`：第二轮已接入 `vizwiz_train` conversion/runtime/config/tests。转换产物位于 `./data/converted`，默认训练 split `official_train_answerable_without_mmeb_v2_eval` 为 14,012 行，`data/images.lance` 为 24,842 行；MMEB-V2 eval question/answer overlap 排除 979 行。runtime 使用图像+问题作为 query，positive 使用 confidence-aware majority voting 得到的 selected answer。
- `VATEX`：第二轮已接入 `vatex_train` conversion/runtime/config/tests，并完成 frame-unit 正式根迁移。正式根目录位于 `./data/converted`，默认训练 split `official_train_without_mmeb_v2_eval` 为 22,105 行，`data/frames.lance` 为 22,105 行，`data/videos.lance` 为 22,105 行，坏媒体排除 3,886 行并写入 `data/exclusions/bad_media.lance`，与 MMEB-V2 `VATEX` eval 的 video id overlap 为 0。runtime 默认读取 `data/frames.lance`，每个完整视频最多保留 64 帧，`VATEX` 已进入 `expanded_train_datasets_v2.yaml`。
- `RefCOCO` / `RefCOCO-Matching`：第二轮已接入 `refcoco_train` conversion/runtime/config/tests。metadata 来源为 `./data/converted`，媒体来源为本地 `./data/converted`；转换产物位于 `./data/converted`。默认训练 split `official_train_without_mmeb_v2_eval` 为 38,181 行，`data/images.lance` 为 19,994 行；按 `ann_id` 排除 MMEB-V2 eval overlap 4,223 行。`RefCOCO` runtime 使用全图+referring expression 作为 query，positive 为 bbox crop 图像；`RefCOCO-Matching` 复用同一根目录和 split，并在 positive 侧保留 referring expression prompt。
- `OpenGVLab/VideoChat2-IT`：已按 media-backed 子集完成正式接入，入口为 `videochat2_it_train`，正式根位于 `./data/converted`。优先使用 `byminji/VideoChat2-IT-clean` 视频清洗清单；本地实测 16 个 `train.json` 合计 877,489 条样本，和上游逐子集表格一致，但上游 README 总数字段写成 874,869。v2 enabled split 当前只纳入能解析到本地正式媒体 root 的 `next_qa`、`youcook2`、`ssv2`、`k710` 和 `videochatgpt`，展开后为 195,584 条训练样本；缺失媒体的 `youcook2` / `k710` 源行已写入 `data/exclusions/bad_media.lance`，其它无本地媒体的子集仍保留 audit-only。
- `Place365`：已完成本地样本级核验并接入 `place365_train`。旧目录遍历式 parser 已替换为显式 split 表。

### 6. 当前触发状态

以下前置条件现已基本满足：

- 数据治理与 provenance 基础能力已经可用
- task archetype 抽象已经稳定
- dataset-owned adapter 边界已经基本收敛
- 已可对新下载数据运行 raw schema / quality / overlap 审计

因此，**现在可以开始按本页清单下载候选数据集到本地**，然后进入正式本地审计。

建议优先顺序：

1. 本轮已完成并进入默认训练配置：`NExTQA`、`YouCook2`、`ActivityNet Captions`
2. 与现有 eval 潜在耦合较强且需要持续 source-unit 复核：`OVEN_it2it_train`、`EDIS_train`、`FashionIQ_train`、`visual7W_pointing_train`
3. 长期候选池中尚未确认官方源或本地路径的对象：`LaSCo`、`NLVR2`、`VCR`、`Visual Genome`、`Docmatix`
4. 第二轮数据集扩展继续处理：`ViDoSeek-page`、`ViDoSeek-doc`、`MMLongBench-page`
5. 按子集拆分的 video instruction proxy：`OpenGVLab/VideoChat2-IT-clean`
6. 超大预训练 / 合成来源：`LAION`、`DataComp`、`CC3M/CC12M`、`WebVid`、`HowTo100M`、`InternVid`

## 待适配汇总表

| 类别 | 配置数 | 任务类型 |
|------|--------|----------|
| 视频分类 | 4 | 视频分类 |
| 视频检索 | 8 | 视频检索 |
| 视频时刻检索 | 2 | 视频时刻检索 |
| 视频问答 | 2 | 视频问答 |
| OOD 图像数据集 | 6 | 多种 |
| OOD 文档数据集 | 1 | 文档理解 |
| M-BEIR 数据集 | 3 | 多模态检索 |
| 视觉定位 | 1 | 视觉定位 |

---

## archive 参考但当前已覆盖的数据

### ViDoRe 文档检索子集

ViDoRe 在当前仓库中不是 pending family。当前 BToks 已经通过 `vidore_train` 正式支持完整的 `colpali_train_set`，以及下列 5 个按 `source` 切分的子集配置。

这些条目不是新的上游数据源，而是 archive-era 实验在同一个 `vidore/colpali_train_set` 训练集内部按 `source` 拆出来的子集配置。

- **HuggingFace**: [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set)
- **论文**: ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449))

| 配置键 | 子集名称 | 样本数 | 任务 |
|--------|----------|--------|------|
| colpali_train_set_arxiv_qa | arxiv_qa | 9,949 | 文档检索 |
| colpali_train_set_pdf | pdf | 45,718 | 文档检索 |
| colpali_train_set_docvqa | docvqa | 39,302 | 文档检索 |
| colpali_train_set_tatdqa | tatdqa | 13,188 | 文档检索 |
| colpali_train_set_Infographic-VQA | Infographic-VQA | 10,038 | 文档检索 |

注意：

- 当前项目已正式支持完整的 `colpali_train_set` 与这 5 个 source-specific 子集。
- 这 5 个配置本质上是同一训练集的 `source` 级拆分，不代表引入了新的 HuggingFace 数据源。
- archive 中额外值得保留的信息不是“新数据”，而是“新训练视图”：这些子集曾使用 `vidore_rag_single` 风格配对，query 侧改为纯文本检索指令，positive 侧改为文档图像；这与当前正式文档中记录的原始 `vidore` 配对方式不同。
- 因此，这一组内容应理解为“当前已覆盖的数据源 + archive 训练视图差异”，而不是待适配的新训练数据。
- 当前实现路径：
  - `vidore_train` 保留完整 full view 语义；
  - `vidore_rag_train` 单独表达 archive `vidore_rag_single` 派生训练视图，不通过 `vidore_train` 的 `source_filter` 实现。
- 07-02 实现合同：
  - 推荐 artifact / dataset naming：`vidore-rag-<source>` 或等价的独立标准化训练子集命名；
  - 推荐 loader / view naming：`vidore_rag_train` 一类的独立训练视图名字；
  - 禁止事项：不要把 `vidore_train + source_filter` 当作 archive `vidore_rag_single` 的语义等价实现。
- 2026-03-31 本地审计结果：
  - raw root：`./data/converted`
  - converted root：`./data/converted`
  - raw `train` 总行数：`118,195`
  - raw / converted 一致发现的 `source`：`Infographic-VQA`、`arxiv_qa`、`docvqa`、`pdf`、`tatdqa`
  - archive 5 个 `vidore_rag_single` 子集全部覆盖，无缺失 source，也没有额外 source
  - raw 抽样结果显示：上游存储是 `query + image + answer (+ prompt / options / page / answer_type)` 视图，而不是 archive `rag_single` 的文本查询到图像正样本视图
  - converted root 目前作为标准化训练根目录使用；后续 subset / rag view 都应从这一层继续写成独立 artifact

---

### VisRAG 文档检索子集

VisRAG 在当前仓库中同样不是 pending family。当前 BToks 已经通过 `visrag_train` 正式支持完整的 `visrag-indomain`，以及下列 6 个按 `source` 切分的子集配置。

这些条目同样不是新的上游数据源，而是 archive-era 实验在同一个 `openbmb/VisRAG-Ret-Train-In-domain-data` 训练集内部按 `source` 拆出来的子集配置。

- **HuggingFace**: [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data)
- **论文**: VisRAG: Vision-based Retrieval-augmented Generation on Multi-modal Documents ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594))

| 配置键 | 子集名称 | 样本数 | 任务 |
|--------|----------|--------|------|
| visrag-indomain_InfoVQA | InfoVQA | 17,664 | 文档检索 |
| visrag-indomain_ArxivQA | ArxivQA | 25,856 | 文档检索 |
| visrag-indomain_PlotQA | PlotQA | 56,192 | 文档检索 |
| visrag-indomain_SlideVQA | SlideVQA | 8,192 | 文档检索 |
| visrag-indomain_MP-DocVQA | MP-DocVQA | 10,624 | 文档检索 |
| visrag-indomain_ChartQA | ChartQA | 4,224 | 文档检索 |

注意：

- 当前项目已正式支持完整的 `visrag-indomain` 与这 6 个 source-specific 子集。
- 这 6 个配置本质上是同一训练集的 `source` 级拆分，不代表引入了新的 HuggingFace 数据源。
- 与 ViDoRe 不同，archive 中 `visrag_single` 与整包 `visrag` 在训练配对语义上基本保持一致；主要变化是把整包 mixture 拆成可独立采样、独立赋权的 source-specific 子集。
- 因此，这一组内容应理解为“当前已覆盖的数据源 + archive mixture 视图差异”，而不是待适配的新训练数据。
- 当前推荐路径：
  - 先通过本地子集审计确认 6 个 archive source 在本地数据中的覆盖情况与 eval/source-unit 风险；
  - 后续实现优先把这些 source 子集写成独立 artifact / 独立 mixture member；
  - `source_filter` 只保留为审计完成后的延后性能优化选项。
- 07-02 实现合同：
  - 推荐 artifact naming：`visrag-indomain-<source>` 或等价的独立标准化训练子集命名；
  - 训练配置应把 `InfoVQA / ArxivQA / PlotQA / SlideVQA / MP-DocVQA / ChartQA` 作为独立 mixture member 赋权；
  - 仅当预拆分 artifact 的存储或维护成本确认不可接受时，才允许后续评估 `source_filter_candidate`。
- 2026-03-31 本地审计结果：
  - raw root：`./data/converted`
  - converted root：`./data/converted`
  - raw `train` 总行数：`122,752`
  - raw / converted 一致发现的 `source`：`ArxivQA`、`ChartQA`、`InfoVQA`、`MP-DocVQA`、`PlotQA`、`SlideVQA`
  - archive 6 个 `visrag_single` 子集全部覆盖，无缺失 source，也没有额外 source
  - raw 抽样结果显示：上游存储已经与 archive `visrag_single` 的训练语义基本对齐，核心差异集中在 source-specific 子集化与独立赋权
  - converted root 目前作为标准化训练根目录使用；source 子集应从这一层继续写成独立 artifact

---

## 真正待适配的额外数据

### 视频分类

| 配置键 | 解析器 | 样本数 | 已确认下载源 | 论文 |
|--------|--------|--------|--------------|------|
| K700 | Kinetics-700 | 约 536,499 (完整) | [siyrus/BToks-Kinetics-700](https://huggingface.co/datasets/siyrus/BToks-Kinetics-700) | A Short Note on the Kinetics-700-2020 Human Action Dataset ([arXiv:2010.10864](https://arxiv.org/abs/2010.10864)) |
| Breakfast | Breakfast | 1,556 | [siyrus/BToks-Breakfast](https://huggingface.co/datasets/siyrus/BToks-Breakfast) | The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities (CVPR 2014) |

以上数据集均托管于 MMEB-V2。任务类型：视频分类。

---

### 视频检索

| 配置键 | 解析器 | 样本数 | HuggingFace | 论文 |
|--------|--------|--------|-------------|------|
| MSR-VTT | msrvtt | 9,000，配置权重已折叠 archive `rep1-4` 重复采样 | [friedrichor/MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT) | MSR-VTT: A Large Video Description Dataset for Bridging Video and Language (CVPR 2016) |
| MSVD | msvd | 1,200，配置权重已折叠 archive `rep1-4` 重复采样 | [friedrichor/MSVD](https://huggingface.co/datasets/friedrichor/MSVD) | Collecting Highly Parallel Data for Paraphrase Evaluation (ACL 2011) |
| DiDeMo | didemo | 8,395 | [friedrichor/DiDeMo](https://huggingface.co/datasets/friedrichor/DiDeMo) | Localizing Moments in Video with Natural Language ([arXiv:1708.01641](https://arxiv.org/abs/1708.01641)) |

注意：archive 中的 MSR-VTT 和 MSVD 使用 rep1-4 副本（共 5 个条目）来对较小的视频检索数据集进行过采样。这些副本使用相同 parser、数据源和 transform，不是新的训练视图；当前配置不再保留重复 entry，而是把重复采样折叠到主 entry 的 `weight` 中，并接受与 archive slot-based epoch 组成之间的小差异。

---

### 视频时刻检索

| 配置键 | 解析器 | 样本数 | HuggingFace | 论文 |
|--------|--------|--------|-------------|------|
| Charades-STA | Charades-STA_train | 约 12,408 | [siyrus/BToks-Charades-STA](https://huggingface.co/datasets/siyrus/BToks-Charades-STA) | TALL: Temporal Activity Localization via Language Query ([arXiv:1705.02101](https://arxiv.org/abs/1705.02101)) |
| QVHighlights | qvhighlight_train | 约 8,000+ | [siyrus/BToks-QVHighlight](https://huggingface.co/datasets/siyrus/BToks-QVHighlight) | QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries ([arXiv:2107.09609](https://arxiv.org/abs/2107.09609)) |

以上数据集均托管于 MMEB-V2。任务类型：视频时刻检索。

---

### 视频问答

| 配置键 | 解析器 | 样本数 | 已确认下载源 | 论文 |
|--------|--------|--------|--------------|------|
| NEXTQA | nextqa | 34,132 | [Shuaiii/NExT-QA](https://huggingface.co/datasets/Shuaiii/NExT-QA) 的 `nextqa_train.json` 加 [rhymes-ai/NeXTVideo](https://huggingface.co/datasets/rhymes-ai/NeXTVideo) 的 `NExTVideo.zip`；官方 [doc-doc/NExT-QA](https://github.com/doc-doc/NExT-QA) 仍作为 provenance 依据 | NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions ([arXiv:2105.08276](https://arxiv.org/abs/2105.08276)) |
`NExTQA` 源确认：archive parser 需要多选训练 CSV，字段包括 `video`、`question`、`answer`、`qid`、`type`、`a0` 到 `a4`、`video_path`。当前实现使用 `Shuaiii/NExT-QA` 的 `nextqa_train.json` 作为主表，字段为 `video`、`frame_count`、`width`、`height`、`question`、`answer`、`qid`、`type`、`options`、`video_path`；conversion 保留 `options` list，不再写入派生的 `a0` 到 `a4` 列，runtime 在构造 query 时从 `options` 动态展开 `(A)` 到 `(E)`。`rhymes-ai/NeXTVideo` 的 `NExTVideo.zip` 覆盖全部 3,870 个训练 video path，且额外 1,570 个视频不属于 train split。注意 `rhymes-ai/NeXTVideo/train.jsonl` 已经把答案改写为选项字母，不适合作为 archive train parser 的原始 annotation。

---

### OOD 图像数据集

用于评估泛化能力的分布外图像数据集。

| 配置键 | 解析器 | 样本数 | HuggingFace | 论文 |
|--------|--------|--------|-------------|------|
| Place365 | Place365_train | 226,397 | [zichengsaber/Places365](https://huggingface.co/datasets/zichengsaber/Places365) 或本地 | Places: A 10 Million Image Database for Scene Recognition (TPAMI 2017) |
| Country211 | country211_train | 31,650 | [clip-benchmark/wds_country211](https://huggingface.co/datasets/clip-benchmark/wds_country211) | CLIP: Learning Transferable Visual Models From Natural Language Supervision ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) |
| ScienceQA | scienceQA_train | 12,726 | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (NeurIPS 2022) |
| GQA | GQA_train | 70,000 | [lmms-lab/GQA](https://huggingface.co/datasets/lmms-lab/GQA) | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering ([arXiv:1902.09506](https://arxiv.org/abs/1902.09506)) |
| TextVQA | textvqa_train | 34,602 | [facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) | Towards VQA Models That Can Read ([arXiv:1904.08920](https://arxiv.org/abs/1904.08920)) |
| Wiki-SS-NQ | wikissnq_train | 15,293 | [Tevatron/wiki-ss-nq](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) | Wiki-SS-NQ: Wikipedia Screenshot Natural Questions |

---

### OOD 文档数据集

| 配置键 | 解析器 | 样本数 | HuggingFace | 论文 |
|--------|--------|--------|-------------|------|
| MMLongBench-doc | MMLongBench_doc_train | 约 1,091 | [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) | MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations |

任务类型：文档理解。

---

### M-BEIR 数据集

用于通用信息检索的多模态基准数据集。

- **HuggingFace**: [TIGER-Lab/M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)
- **论文**: One Embedder, Any Task: Instruction-Finetuned Text Embeddings / UniIR: Training and Benchmarking Universal Multimodal Information Retrievers ([arXiv:2311.17136](https://arxiv.org/abs/2311.17136))

| 配置键 | 解析器 | 任务 ID | 样本数 | 原始论文 |
|--------|--------|---------|--------|----------|
| oven_it2it_train | mbeir | 8 | 154,460 | Open-domain Visual Entity Recognition (arXiv:2302.11154) |
| EDIS_train | mbeir | 2 | 25,917 | EDIS: Entity-Driven Image Search over Multimodal Web Content (ECIR 2022) |
| FashionIQ_train | mbeir | 7 | 16,176 | Fashion IQ: A New Dataset Towards Retrieving Images by Relative Natural Language Feedback (arXiv:1905.12794) |

任务类型：多模态检索。

---

### 视觉定位

| 配置键 | 解析器 | 样本数 | HuggingFace | 论文 |
|--------|--------|--------|-------------|------|
| visual7W_pointing_train | visual7W_pointing_train | 186,188 | [ai2-jackh/visual7w-pointing](https://huggingface.co/datasets/ai2-jackh/visual7w-pointing) 或本地 | Visual7W: Grounded Question Answering in Images ([arXiv:1511.03416](https://arxiv.org/abs/1511.03416)) |

任务类型：视觉定位。

---

## 适配说明

将这些数据集适配到 BToks 时：

1. **逐数据集重写**：新数据集不再优先复用旧 family loader / contract 设计；每个数据集独立确认 raw schema、origin parser 行为和目标 `TrainSample` 输出，再实现 dataset-owned transform。

2. **Lance Schema**：根据真实 raw schema 决定单表、双表 join 或视频帧 + 指令双表 schema。conversion 默认 raw-preserving；只有经过审计确认的窄范围数据修复可以例外。

3. **转换入口**：业务逻辑放在 `src/vlm2emb/data/conversion/<dataset>.py`，`scripts/convert/train/convert_<dataset>_to_lance.py` 只做参数解析、日志和函数调用。

4. **旧代码清理**：每个数据集的新 conversion/runtime 通过验证后，删除对应旧转换脚本、旧 runtime、旧 helper、旧测试和旧文档入口。

5. **任务类型**：将每个数据集映射到当前训练 runtime 所需的任务语义（document_retrieval、video_classification、video_retrieval 等），并在文档中明确 `implemented`、`design_only`、`eval_only`、`pretrain_only` 或 `defer`。

---

## 后续下载与审计清单

以下清单用于未来交给用户下载到本地。当前阶段不要求全部已经有官方 HuggingFace 地址；若地址仍不明确，后续下载前需要人工补齐。

### A. final config 中真实启用过、优先审计的数据集

说明：

- 本表记录 archive final config 中真实启用过、或本轮优先审计过的对象；其中一部分已经迁移到当前正式训练 registry，保留在这里是为了记录迁移结论、row count、source 风险和旧入口清理状态。
- “静态风险”表示基于 archive parser、文档与配置的风险分类；若条目已经完成本地样本级核验，后续动作会直接记录核验结论。

| 数据集 | 任务原型 | 当前状态 | 静态风险 | 地址状态 | 后续动作 |
|--------|----------|----------|----------|----------|----------|
| HMDB51 | video classification | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 低，训练侧显式读 `train` | 已知 | `hmdb51_train` 已接入，正式根目录位于 `./data/converted`；`vlm2vec_train` 为 10,710 行，`data/frames.lance` 为 6,436 行，默认路径不 raw decode |
| UCF101 | video classification | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 低，训练侧显式读 `train` | 已知 | `ucf101_train` 已接入，正式根目录位于 `./data/converted`；`vlm2vec_train` 为 28,747 行，`data/frames.lance` 为 13,320 行，默认路径不 raw decode |
| MSR-VTT | video retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 低，训练侧显式读 `train_9k/train`；本地核对 `train_9k` 与 eval `test_1k` 的 video_id 重叠为 0 | 已知 | `msrvtt_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `train_9k_without_mmeb_v2_eval`，当前行数与 `train_9k` 同为 9000，`data/frames.lance` 为 10,000 行 |
| MSVD | video retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 低，训练侧显式读 `train`；本地核对 `train` 与 eval `test` 的 video_id 重叠为 0 | 已知 | `msvd_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `train_without_mmeb_v2_eval`，当前行数与 `train` 同为 1200，`data/frames.lance` 为 1,970 行 |
| DiDeMo | video retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 低，训练侧显式读 `train`；本地核对 `train` 与 eval `test` 的 video path 重叠为 0 | 已知 | `didemo_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `train_without_mmeb_v2_eval`，当前行数与 `train` 同为 8395，`data/frames.lance` 为 9,399 行 |
| VATEX | video retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit / 已进入 v2 enabled 配置 | 中，官方训练 JSON 和视频包已可用；旧 raw-video 根已备份，正式根已替换为 frame-unit 根；坏媒体排除 3,886 行 | 已知 | `vatex_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 22,105，`data/frames.lance` 为 22,105 行，`data/videos.lance` 为 22,105 行 |
| K700 | video classification | 已实现 / 已提升为正式数据 / 已迁移 frame-unit / 已进入 v2 enabled 配置 | 中，目录规模大；当前 runtime 默认使用 `train_without_mmeb_v2_eval`，并保留 invalid-filtered 的 `official_train` 审计视图 | 已知 | 当前正式根目录位于 `./data/converted`，`data/frames.lance`、`data/videos.lance`、`official_train.lance` 和 `train_without_mmeb_v2_eval.lance` 均为 536,489 行；历史审计确认 raw `train.csv` 去重后为 536,499 个 `video_path`，10 个原始 tar member 没有可用 video stream，并记录到 `data/exclusions/invalid_videos.lance` 与 `data/exclusions/bad_media.lance` |
| Breakfast | video classification | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 中，origin parser 的 `cam01` 过滤已还原；`cereal/cereals` 与 `salad/salat` 命名差异已作为显式 alias 处理 | 已知 | `breakfast_train` 已接入，runtime positive 优先使用 canonical `label`，`raw_label` 保留为审计字段；正式根目录位于 `./data/converted`，`data/frames.lance` 为 1,989 行 |
| Charades-STA | video moment retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 低，训练侧显式读 `Charades_sta_train.txt`；默认训练 split 保留 `official_train` 原始视图并额外提供 eval-excluded 视图 | 已知 | `charades_sta_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数与 `official_train` 同为 12408，`data/frames.lance` 为 18,444 行 |
| QVHighlights | video moment retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 中，本地官方命名 tar `qvhilights_videos.tar.gz` 已确认截断并报 EOF，当前不作为 source；本轮使用 `yeliudev/VideoMind-Dataset` 的社区 HF 分卷视频源展开目录，已验证 train/val/test annotation 对应视频缺失数为 0，但仍未证明与官方 tar 字节级一致 | 部分已知 | `qvhighlight_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 7218，`data/frames.lance` 为 25,715 行；后续如支持 archive 直读，可优先审计本地 `yeliudev` 三段分卷而不是切回损坏 tar |
| ActivityNet Captions | video segment retrieval | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 中，archive 没有训练 parser；本轮已确认使用 segment-level text-to-video retrieval，且 train split 与 MMEB-V2 `ActivityNetQA` eval video id 重叠为 0 | 已知 | `activitynet_captions_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 37,421，`data/frames.lance` 为 71,787 行，`data/videos.lance` 为 14,926 行 |
| NExTQA | video QA | 已实现 / 已提升为正式数据 / 已迁移 frame-unit | 中，archive 训练 parser 使用多选 `train.csv`；当前 `Shuaiii/NExT-QA` annotation 与 `rhymes-ai/NeXTVideo` 视频包已覆盖 train split，且与 MMEB-V2 `NExTQA` eval video id 重叠为 0 | 已知 | `nextqa_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 34,132，`data/frames.lance` 为 3,870 行 |
| Place365 | image classification | 已实现 / 已提升为正式数据 | 中，训练 parser 为目录遍历式；当前实现改为显式 split 表 | 部分已知 | `place365_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train` |
| Country211 | image classification | 已实现 / 已提升为正式数据 | 低，训练与 eval 文档都指向显式 split 体系 | 已知 | `country211_train` 已接入，默认 split 排除 MMEB-V2 eval 边界 |
| ScienceQA | image QA | 已实现 / 已提升为正式数据 | 低，存在显式上游 split | 已知 | `scienceqa_train` 已接入，默认训练 split 为 6218 行 |
| GQA | image QA | 已实现 / 已提升为正式数据 | 中，地址与本地源存在多版本可能；已完成 image-aware overlap audit，train image IDs 与 MMEB-V2 eval image IDs 交集为 0，`imageId + question` 交集也为 0 | 部分已知 | `gqa_train` 已接入，默认训练 split 为 `official_train_balanced_without_mmeb_v2_eval`，当前为 943000 行；14,305,356 行的 `official_train_all` 保留为显式 full-source view；text-only question / answer overlap 很高但发生在不同图像上，不作为样本级泄漏排除键 |
| TextVQA | image QA | 已实现 / 已提升为正式数据 | 低，存在显式上游 split | 已知 | `textvqa_train` 已接入，默认训练 split 为 34496 行，已按 query/answer key 排除 106 行 |
| Wiki-SS-NQ | image retrieval | 已实现 / 已提升为正式数据 | 中，训练 query 与 MMEB-V2 eval query 有重叠，默认 split 已按 query 文本排除；图像 join key 使用 screenshot 文件名 stem | 已知 | `wikissnq_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 332518；`images.lance` 为 1267874 行且覆盖所有 positive docid |
| MMLongBench-doc | visdoc OOD / document QA | 已实现 / 已提升为正式数据 | 高，archive 可用的 473 条单页样本中 469 条与 MMEB-V2 eval query 精确重叠；默认 split 只能使用 4 条非重叠样本 | 已知 | `mmlongbench_doc_train` 已接入，正式根目录位于 `./data/converted`；主表保留原始 parquet 字段，页面图像写入 `data/pages.lance` |
| OVEN_it2it_train | multimodal retrieval | 已实现 / 已提升为正式数据 | 中，来自 M-BEIR，需持续复核与 eval `OVEN` 的 source-unit 边界 | 已知 | `mbeir_train` 已接入，默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 143051 |
| EDIS_train | multimodal retrieval | 已实现 / 已提升为正式数据 | 中，来自 M-BEIR，且 eval 已含 EDIS | 已知 | `mbeir_train` 已接入，默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 22676 |
| FashionIQ_train | multimodal retrieval | 已实现 / 已提升为正式数据 | 中，来自 M-BEIR，且 eval 已含 FashionIQ | 已知 | `mbeir_train` 已接入，默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 10755 |
| visual7W_pointing_train | visual grounding | 已实现 / 已提升为正式数据 | 中，需持续复核与 eval `Visual7W-Pointing` 的样本边界 | 部分已知 | `visual7w_pointing_train` 已接入，正式根目录位于 `./data/converted`；默认 split 为 `official_train_without_mmeb_v2_eval`，当前行数为 93813 |

### B. archive 中出现过、但未保留在最终训练配方中的候选数据集

| 数据集 | 任务原型 | 当前状态 | 地址状态 | 后续动作 |
|--------|----------|----------|----------|----------|
| OVEN_it2t_train | multimodal retrieval | 已实现 / 已提升为正式数据 | 已知 | `mbeir_train` 已接入，默认 split 为 132399 行 |
| INFOSEEK_it2t | multimodal retrieval | 已实现 / 已提升为正式数据 | 已知 | `mbeir_train` 已接入，默认 split 为 135613 行 |
| INFOSEEK_it2it | multimodal retrieval | 已实现 / 已提升为正式数据 | 已知 | `mbeir_train` 已接入，默认 split 为 134330 行 |
| Fashion200K_t2i | multimodal retrieval | 已实现 / 已提升为正式数据 | 已知 | `mbeir_train` 已接入，默认 split 为 14123 行 |
| Fashion200K_i2t | multimodal retrieval | 已实现 / 已提升为正式数据 | 已知 | `mbeir_train` 已接入，默认 split 为 12558 行 |

### C. archive 训练视图差异（非新增数据源）

以下对象值得在后续训练设计时参考，但它们本身不应被算作“新增待适配数据集”：

- `vidore_rag_single`：同一 `vidore/colpali_train_set` 数据源上的不同配对视图，强调 text-to-document-image 检索。
- `visrag_single`：同一 `openbmb/VisRAG-Ret-Train-In-domain-data` 数据源上的 source-specific mixture 视图。
- `MSR-VTT_rep1-4`、`MSVD_rep1-4`：对小规模视频检索数据集的重复采样策略，不是新的上游数据源；当前已折叠为主 entry 的采样权重。
- `YouCook2`：不要使用 `lmms-lab/YouCook2` eval 镜像作为训练源；训练侧只使用已审计通过的原始 `YouCook2` train view。

### D. `lmms-lab` Hugging Face 扫描候选（2026-04-03）

扫描范围：

- `https://huggingface.co/lmms-lab/datasets` 当前公开的 166 个数据集。
- 筛选标准只保留与当前框架任务空间接近的对象：`video classification / retrieval / QA / moment retrieval`、`image QA / retrieval / visual grounding`、`visdoc / document QA`、`multimodal retrieval`。
- 明确排除：纯评测镜像、纯文本题库、音频主模态数据、以及与当前文档已覆盖数据集重复的镜像。

建议优先评估的新增候选：

| 数据集 | 建议映射任务 | 当前判断 | 来源 | 备注 |
|--------|--------------|----------|------|------|
| ScreenSpot-v2 | image visual grounding / UI grounding | 值得优先评估 | [lmms-lab/ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2) | HF 仓中明确含 `train` split；与当前 `visual7W_pointing_train` 同属视觉定位家族，但目标域是 GUI 屏幕，需确认是否要扩展为新的 screen-grounding 子类型 |
| ScreenSpot-Pro | image visual grounding / UI grounding | 值得优先评估 | [lmms-lab/ScreenSpot-Pro](https://huggingface.co/datasets/lmms-lab/ScreenSpot-Pro) | HF 仓中明确含 `train` split；相较 `ScreenSpot-v2` 更像同域增强版，建议和 `ScreenSpot-v2` 成对评估，只保留一个或做层级采样 |
| LLaVA-Video-178K | video QA / video-text instruction | defer：归入后续 video instruction family 设计 | [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | 本地源已下载到 `./data/converted`，包含 processed QA/caption JSON 和大量视频 tar；约 1.2T，不套用当前 retrieval/classification rewrite 模板 |
| EgoIT-99K | multimodal QA / video-text instruction | defer：归入后续 egocentric instruction family 设计 | [lmms-lab/EgoIT-99K](https://huggingface.co/datasets/lmms-lab/EgoIT-99K) | 本地源已下载到 `./data/converted`，包含 image/video/audio/segment 与多源 parquet/JSON；需要先限定目标子任务 |

暂不建议作为“新增训练数据集”纳入：

- `ActivityNetQA`：`lmms-lab/ActivityNetQA` 当前仓里只看到 `test` parquet 和视频 zip，更像评测镜像，不适合直接作为新的训练源。
- `charades_sta`：`lmms-lab/charades_sta` 当前仓里只看到 `test` parquet；而且我们已经有更完整的官方 `Charades_v1.zip + Charades-STA annotations` 方案。
- `RefCOCO / RefCOCOplus / RefCOCOg`：当前 HF 仓里只有 `val/test`，没有 `train`，适合作为 eval 参考，不适合当新的训练源。
- `MP-DocVQA`：当前 HF 仓里只有 `val/test`，而且框架里已经存在 `visrag-indomain_MP-DocVQA` 路径，优先级不高。
- `VideoSearch`：仓里只有各 milestone 的 `test` parquet，没有训练 split。
- `EMMA`：当前公开内容以 `test` 为主，更像评测集而不是训练源。
- `FVQA`：虽然有 `train/test` parquet，但当前 tag 显示主模态偏文本，仓内也没有清晰的原始图像包，接入收益不如 `ScreenSpot-*` 明确。

已被当前清单覆盖或与现有对象高度重复的镜像：

- `GQA`、`textvqa`、`ScienceQA`、`NExTQA`、`YouCook2`：当前文档中已经有对应训练/候选对象，不需要因为 `lmms-lab` 镜像重复开新下载项。
- `VATEX`：当前本地 `lmms-lab` 副本只确认到 eval/test 镜像线索，不能作为训练源。训练侧使用 `qingy2024/VaTeX` 的官方训练 JSON 和匹配视频包；conversion/runtime 代码已存在，正式根目录已重转出 `data/frames.lance` 并进入 v2 enabled 配置。
- `DocVQA`：虽然 `lmms-lab/DocVQA` 同时带 `train/validation/test`，但当前框架已存在 `MMEB` 训练版 `DocVQA`，新增收益有限，除非后续要做 source 对比或替换上游。
- `EgoLife`：虽然含大量原始视频，但主标签是 `video-text-to-text` 且语言以中文为主，当前更像未来单独立项，而不是直接塞进现有训练混合物。

当前本地推进状态（2026-04-20）：

- 已存在 `lmms-lab` 本地副本：`GQA`、`NExTQA`、`VATEX`、`YouCook2`
- 已完成新的 `lmms-lab` 下载：
  - `ScreenSpot-v2`
  - `ScreenSpot-Pro`
  - `DocVQA`
  - `textvqa`
  - `ScienceQA`
  - `EgoIT-99K`
  - `LLaVA-Video-178K`
- 当前这些下载统一落在 `./data/converted`；本轮检查时没有仍在运行的下载 tmux session
- 其他已启动的大数据转换最终状态：
  - `M-BEIR` 正式根目录位于 `./data/converted`，约 7.9G，已完成 runtime 抽样
  - `Kinetics-700` 正式根目录位于 `./data/converted`，约 687G，默认训练 split 为 536489 行
  - `QVHighlights` 正式根目录位于 `./data/converted`，约 110G，默认训练 split 为 7218 行
  - `siyrus/BToks-Breakfast` 正式根目录位于 `./data/converted`，约 3.8G，已完成 runtime 抽样
- 本轮后续补齐下载并完成源头审计：
  - `NExTQA` annotation 位于 `./data/converted`，视频 zip 位于 `./data/converted`；train annotation 的 3,870 个视频全部被 zip 覆盖，且与 MMEB-V2 eval video id 重叠为 0
  - `YouCook2` train parquet 位于 `./data/converted`，raw video 分卷 tar 位于 `./data/converted`；parquet 与官方 train annotation 仅差 1 个视频 / 7 个 segment，与 MMEB-V2 eval 无 youtube id / segment stem 重叠，且分卷 tar 覆盖 parquet 所需 1,332 个视频 member
  - `ActivityNet Captions` 位于 `./data/converted`；train split 的 10,009 个 video ids 与 MMEB-V2 `ActivityNetQA` eval video id 重叠为 0，且分卷 tar 覆盖 train、val1、val2 合计 14,926 个 unique video id
