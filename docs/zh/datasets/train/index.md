# 训练数据集总览

当前训练数据集文档按“已实现家族、独立 runtime、待确认候选”分层维护。已完成 rewrite 的训练入口都直接输出 `TrainSample`，conversion 保留原始表字段，并把图像或视频聚合到共享 Lance 表。

## 当前已实现的训练数据集家族

| 家族 | 子集数 | 存储边界 | Lance Schema |
|------|--------|---------|-------------|
| [MMEB](./mmeb/index.md) | 20 | raw-preserving | sample 表 + image side table |
| [ViDoRe](./vidore/index.md) | 6 | raw-preserving | 单表 |
| [VisRAG](./visrag/index.md) | 7 | raw-preserving | 单表 |
| [LLaVA-Hound](./llavahound/index.md) | 3 | rewritten | 双表（frames + instructions） |
| 视频分类独立 runtime | 5 | 已实现；v1 已启用的视频分类正式根和 v2 `K700` 正式根均已迁移为 frame-unit | split 表 + `data/videos.lance` + `data/frames.lance` |
| 视频检索独立 runtime | 5 | 已实现；v1 已启用的视频检索正式根和第二轮 `VATEX` 正式根均已迁移为 frame-unit | split 表 + `data/videos.lance` + `data/frames.lance` |
| 视频时刻检索独立 runtime | 3 | 已实现；正式根均已迁移为 mixed-view / segment frame-unit | split 表 + `data/videos.lance` + `data/frames.lance` |
| OOD / VQA 独立 runtime | 9 | raw-preserving | split 表 + image/page side table |
| [M-BEIR](./mbeir/index.md) | 8 | raw-preserving | query 表 + candidate 表 + `data/images.lance` |
| [视觉定位](./visual-grounding/index.md) | 1 | raw-preserving | split 表 + `data/images.lance` |

## 设计占位卡片（尚未落地为正式训练实现）

| 家族 | 子集数 | 当前状态 |
|------|--------|----------|
| OOD 未实现候选 | 0 | 当前无已下载且可直接接入的 OOD 设计占位 |
| 长期预训练候选 | 若干 | `LAION`、`CC3M/CC12M`、`WebVid` 等不进入本轮微调 rewrite |
| eval-only 候选 | 若干 | `ImageNet-A/R`、`ObjectNet`、`RefCOCO+ / RefCOCOg eval`、`FashionIQ eval` 等保留 eval 边界 |
| [MMEB-V2 训练源调研](./mmeb-v2-training-source-survey.md) | 78 | 仅记录 eval 子集是否有训练源和后续审计入口，不执行下载或实现 |
| [v2 补充数据下载源 Manifest](./v2-supplement-source-manifest.md) | 若干 | 记录 v3 回看后建议补入 v2 的下载源、媒体依赖、本地目标和后续审计项 |

## 说明

- 扩展训练配置采用版本化层：`configs/datasets/expanded_train_datasets_v1.yaml` 是第一轮扩展基线，`configs/datasets/expanded_train_datasets_v2.yaml` 继承 v1 并追加第二轮已完成数据，因此可以直接作为 v1 + v2 的完整训练配置使用。当前 v2 追加 `VizWiz`、`RefCOCO`、`RefCOCO-Matching`、`VATEX`、`K700` 和 `VideoChat2-IT`。`configs/datasets/expanded_train_datasets_v3.yaml` 继承 v2，并把 Stage 1 训练全集候选列入 `_stage1_candidates`；这些候选只是下载、审计、conversion 和采样策略入口，不是已经可训练的顶层 dataset entry。BToks 对应层分别是 `configs/datasets/btoks_expanded_train_datasets_v1.yaml` 和 `configs/datasets/btoks_expanded_train_datasets_v2.yaml`。v2 / v3 文件不会被现有默认训练 preset 自动加载；需要第二轮扩展数据或 Stage 1 候选清单时，训练 recipe 应显式引用对应版本。
- 本页的“已实现”表示当前仓库已有正式训练注册、conversion/runtime/tests 和对应配置入口，不等价于当前正式数据根已经满足默认训练取样条件。视频数据集还必须有 `data/frames.lance` 或等价的 frame-unit root 映射才能按默认配置训练；缺表时应报错，除非显式启用 raw-video 兼容路径。已注册入口包括：`mmeb_train`、`vidore_train`、`visrag_train`、`llavahound_train`、`ssv2_train`、`hmdb51_train`、`ucf101_train`、`kinetics700_train`、`breakfast_train`、`msrvtt_train`、`msvd_train`、`didemo_train`、`youcook2_train`、`charades_sta_train`、`qvhighlight_train`、`activitynet_captions_train`、`place365_train`、`country211_train`、`scienceqa_train`、`gqa_train`、`textvqa_train`、`vizwiz_train`、`vatex_train`、`nextqa_train`、`mbeir_train`、`wikissnq_train`、`mmlongbench_doc_train`、`refcoco_train`、`visual7w_pointing_train` 和 `videochat2_it_train`。`VATEX`、`K700` 和 `VideoChat2-IT` 均已进入 v2 enabled 配置。
- MMEB-V2 评测子集的训练源扩展方向记录在 [MMEB-V2 训练源调研](./mmeb-v2-training-source-survey.md)，该页只作为后续单数据集审计入口。
- 视频分类、视频检索和 OOD / VQA 的部分独立 runtime 暂不新增单独卡片目录；其审计结论、split 视图和旧入口清理状态记录在 [待适配训练数据集](./pending-datasets.md)。
- 视频训练 runtime 的目标默认路径是读取转换阶段写出的 `data/frames.lance`，不在训练取样阶段 raw decode 原始视频。当前 v1 已启用视频正式根、v2 `VATEX` 和 v2 `K700` 正式根均已具备默认训练所需的 `data/frames.lance`；`LLaVA-Hound` 使用独立帧表 `data/train_300k.lance`。完整视频或已裁好的短视频最多保留 64 帧；明确有时间窗口的 segment / row-level segment 最多保留 8 帧。
- `Charades-STA` 和 `QVHighlights` 是 mixed-view：query 使用完整视频 frame unit，positive 使用目标片段 frame unit。`YouCook2` 使用上游 parquet 中的 row-level `binary_frames` 还原片段 frame unit，不再把完整视频作为 positive。
- 保留下来的设计卡片只表示后续候选或审计边界，不代表可以直接训练。
- 归档实验里更多尚未落地的数据集，仍集中记录在 [待适配训练数据集](./pending-datasets.md)。
