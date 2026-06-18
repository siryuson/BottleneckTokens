# v2 补充数据下载源 Manifest

本文记录 `add-extend-data-v2-datasets` 在 v3 回看后建议补充的数据源入口。本文主要用于下载源和审计入口确认；是否已进入 `configs/datasets/expanded_train_datasets_v2.yaml` enabled 配置，以训练配置和总览文档为准。

## 准入原则

- 优先使用官方或作者维护入口；HF mirror 只作为快速 schema / sample smoke 或缺失官方下载时的补充。
- 下载前必须确认 license / usage policy、split、媒体依赖、row count、缺失媒体策略和 MMEB-V2 eval overlap key。
- 视频 QA 优先复用仓库现有视频 root，只补 QA annotation 和 id mapping，避免重复下载大视频包。
- 合成和 metric-learning 数据只考虑抽样版进入 v2；全量版本留在 v3 Stage 1。

## P0 下载源

| 数据集 | 主入口 | 辅助入口 | 本地目标 / 复用路径 | 下一步审计 |
|--------|--------|----------|---------------------|------------|
| `OpenGVLab/VideoChat2-IT-clean` | [OpenGVLab/VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) | [byminji/VideoChat2-IT-clean](https://huggingface.co/datasets/byminji/VideoChat2-IT-clean) community cleaned manifest | 已接入 `./data/converted` | v2 正式入口只启用可解析到本地正式媒体 root 的 `next_qa`、`youcook2`、`ssv2`、`k710` 和 `videochatgpt`，展开后 195,584 条训练样本；其它子集保留 audit-only |
| `K700` | [CVDF Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset) | [K700 train path](https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt), [train.csv](https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv) | 复用 `./data/converted` | 已完成 frame-unit 正式根替换；`data/frames.lance`、`data/videos.lance` 和默认 split 均为 536,489 行，10 个坏视频已记录到 exclusions |
| `RefCOCO-Matching` | [refer API](https://github.com/lichengunc/refer) | [TFDS ref_coco](https://www.tensorflow.org/datasets/catalog/ref_coco), [COCO download](https://cocodataset.org/#download) | 复用当前 `RefCOCO` source 和 `./data/converted` | 已复用 `RefCOCO` conversion 根构造 matching view，并按 `ann_id` 排除 MMEB-V2 overlap；保留本行作为 provenance 入口 |

## `OpenGVLab/VideoChat2-IT` 子集拆分和重合判断

`OpenGVLab/VideoChat2-IT` 原始仓库是 gated annotation 入口，官方说明总量约 1.9M JSON annotations，仓库本体主要是 JSON，不含完整原始媒体。当前优先采用 `byminji/VideoChat2-IT-clean` 的视频清洗清单；它是社区清洗 manifest，本地实测 16 个 `train.json` 合计 877,489 条样本，和上游逐子集表格一致，但上游 README 总数字段写成 874,869，并已经过滤缺失视频。本轮正式接入不合成一个全量 dataset entry，而是只把可解析到本地正式媒体 root 的子集行写入 `videochat2_it_train` split；无本地媒体的子集仍按下表保留 audit-only。

| 子集 | 来源家族 | 任务视图 | 有效样本 | 与已接入数据的关系 | 处理结论 |
|------|----------|----------|----------|--------------------|----------|
| `textvr` | TextVR | video caption | 39,648 | 当前无明确正式 root 重合 | 保留候选 |
| `youcook2` | YouCook2 | video caption | 8,700 | 与已接入 `youcook2_train` 同家族，部分 row-level media 可复用 | 正式启用可解析的 4,901 行，其余写入 bad media |
| `k710` | Kinetics | video classification | 38,977 | 与 `kinetics700_train` / `K700` 高重合风险，按 clip basename 复用本地 K700 frame units | 正式启用可解析的 29,844 行，其余写入 bad media |
| `ssv2` | Something-Something-V2 | video classification | 40,000 | 与已接入 `ssv2_train` 同家族，media 完整可复用 | 正式启用 40,000 行 |
| `videochat2` | InternVid | video conversation | 9,584 | 当前无明确正式 root 重合 | 保留候选 |
| `videochatgpt` | ActivityNet | video conversation | 13,303 | 与 `ActivityNet-Captions` 共享本地 ActivityNet 全视频来源，已生成本 root 的 full-video frame units | 正式启用，按 QA turn 展开为 86,707 行 |
| `next_qa` | NExT-QA | video reasoning | 34,132 | 与已接入 `nextqa_train` 同家族，media 完整可复用 | 正式启用 34,132 行 |
| `clevrer_qa` | CLEVRER | video reasoning | 40,000 | 当前无训练 root 重合；但与 MVBench eval family 有来源重合风险 | 保留候选，先做 MVBench overlap audit |
| `clevrer_mc` | CLEVRER | video reasoning | 40,000 | 当前无训练 root 重合；但与 MVBench eval family 有来源重合风险 | 保留候选，先做 MVBench overlap audit |
| `ego_qa` | EgoQA | egocentric video QA | 7,797 | 当前无正式 Ego4D / EgoQA root；可作为 EgoSchema proxy，不需要接入 Ego4D full | 保留小规模候选 |
| `tgif_frame_qa` | TGIF | video QA | 39,149 | 当前无训练 root 重合；但与 MVBench eval family 有来源重合风险 | 保留候选，先做 MVBench overlap audit |
| `tgif_transition_qa` | TGIF | video QA | 52,696 | 当前无训练 root 重合；但与 MVBench eval family 有来源重合风险 | 保留候选，先做 MVBench overlap audit |
| `webvid` | WebVid | video caption | 399,740 | 可能与 LLaVA-Hound / ShareGPTVideo 来源有交叉 | 只做 Stage 1 抽样，先 dedup |
| `videochat` | WebVid | video caption | 6,889 | 可能与 LLaVA-Hound / ShareGPTVideo 来源有交叉 | 审计后决定 |
| `videochat1` | WebVid | video conversation | 4,300 | 可能与 LLaVA-Hound / ShareGPTVideo 来源有交叉 | 审计后决定 |
| `webvid_qa` | WebVid | video QA | 99,954 | 可能与 LLaVA-Hound / ShareGPTVideo 来源有交叉 | 只做 Stage 1 抽样，先 dedup |

## P1 下载源

| 数据集 | 主入口 | 辅助入口 | 本地目标 / 复用路径 | 下一步审计 |
|--------|--------|----------|---------------------|------------|
| `LaSCo` | [LaSCo official repo](https://github.com/levymsn/LaSCo) | [COCO images](https://cocodataset.org/#download), [VQA2.0](https://visualqa.org/) | 建议 `./data/converted` | 下载 train / validation query annotations；确认 COCO train2014 / val2014 依赖、license、query-image / target-image id 和 MMEB-V2 overlap key |
| `SNLI-VE` | [SNLI-VE official repo](https://github.com/necla-ml/SNLI-VE) | [HuggingFaceM4/SNLI-VE](https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE) | 已见 `./data/converted` | 复核 train / dev / test split、Flickr30K 图像依赖、label mapping；定义 entailment positive 和 contradiction hard negative |
| `MSRVTT-QA` | [xudejing video-question-answering](https://github.com/xudejing/video-question-answering) | 复用现有 `MSR-VTT` 视频 root | 复用 `./data/converted` | 只下载 QA annotation；确认 MSR-VTT video id mapping、train split 和与 `MSR-VTT` retrieval eval overlap |
| `MSVD-QA` | [xudejing video-question-answering](https://github.com/xudejing/video-question-answering) | [LAVIS MSVD-QA card](https://github.com/salesforce/LAVIS/blob/main/dataset_card/msvd_qa.md), 复用现有 `MSVD` 视频 root | 复用 `./data/converted` | 只下载 QA annotation；确认 `youtube_mapping.txt`、split 和与 `MSVD` eval overlap |
| `ActivityNet-QA` | [MILVLG/activitynet-qa](https://github.com/MILVLG/activitynet-qa) | [ActivityNet download](https://activity-net.org/download.html), 复用现有 `ActivityNet-Captions` root | 先复用 `./data/converted` | 下载 QA JSON；确认 ActivityNet v1.3 video coverage、与现有 frame units 的 id 对齐和 MMEB-V2 `ActivityNetQA` overlap |
| `mmE5 synthetic` 抽样版 | [intfloat/mmE5-synthetic](https://huggingface.co/datasets/intfloat/mmE5-synthetic) | [mmE5 code](https://github.com/haon-chen/mmE5) | 建议 `./data/converted` | 只取抽样版；确认 3 个 subset、93 语言分布、图像包 `LAION_Synthetic.tar.gz`、去重和 MMEB-V2 overlap |

## P2 下载源

| 数据集 | 主入口 | 辅助入口 | 本地目标 / 复用路径 | 下一步审计 |
|--------|--------|----------|---------------------|------------|
| `SOP` / Stanford Online Products | [Stanford CVGL resources](https://cvgl.stanford.edu/resources.html) | [JamieSJS/stanford-online-products](https://huggingface.co/datasets/JamieSJS/stanford-online-products) | 已见 `./data/converted` | 先用 HF mirror 做 schema / split smoke；再确认官方 license、class / product split 和 image-image retrieval target |
| `DeepFashion In-Shop` | [CUHK In-shop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) | [Marqo/deepfashion-inshop](https://huggingface.co/datasets/Marqo/deepfashion-inshop) | 已见 `./data/converted` | 先用 HF mirror 做 smoke；再确认官方 image / bbox / landmark / partition files、train / query / gallery split 和 FashionIQ / Fashion200K overlap |

## 暂不下载到 v2

以下对象留在 v3 Stage 1 或单独审计，不进入本轮 v2 下载队列：`LAION`、`DataComp`、`Recap-DataComp`、`CC3M`、`CC12M`、`WebVid` 全量、`HowTo100M`、`InternVid` 全量、`MagicLens` 全量、`MegaPairs` 全量、`COYO`、`WIT` 全量。`VideoChat2-IT-clean` 中的 `webvid` / `videochat*` 小切片按上面的子集表单独审计。
