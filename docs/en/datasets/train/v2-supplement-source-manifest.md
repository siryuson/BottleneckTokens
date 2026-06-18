# v2 Supplement Source Manifest

This page records download and audit entry points for the post-v3 v2 supplement candidates from `add-extend-data-v2-datasets`. Training configs and overview docs are authoritative for whether an item is already enabled in `configs/datasets/expanded_train_datasets_v2.yaml`.

## Admission Rules

- Prefer official or author-maintained sources. Use HF mirrors for schema inspection, sample smoke, or as fallback.
- Before download, confirm license / usage policy, splits, media dependencies, row counts, missing-media policy, and MMEB-V2 eval overlap keys.
- For VideoQA datasets, prefer reusing existing video roots and only downloading QA annotations plus id mappings.
- Synthetic and metric-learning datasets should only enter v2 as sampled subsets; full sources remain in v3 Stage 1.

## P0 Sources

| Dataset | Primary Source | Auxiliary Source | Local Target / Reuse Path | Next Audit |
|---------|----------------|------------------|---------------------------|------------|
| `OpenGVLab/VideoChat2-IT-clean` | [OpenGVLab/VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) | [byminji/VideoChat2-IT-clean](https://huggingface.co/datasets/byminji/VideoChat2-IT-clean) community cleaned manifest | Wired under `./data/converted` | The v2 production entry enables only rows that resolve to local production media roots from `next_qa`, `youcook2`, `ssv2`, `k710`, and `videochatgpt`, expanding to 195,584 training samples; the remaining subsets stay audit-only |
| `K700` | [CVDF Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset) | [K700 train path](https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt), [train.csv](https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv) | Reuse `./data/converted` | Production frame-unit root replacement is complete; `data/frames.lance`, `data/videos.lance`, and the default split each have 536,489 rows, and the 10 bad videos are recorded under exclusions |
| `RefCOCO-Matching` | [refer API](https://github.com/lichengunc/refer) | [TFDS ref_coco](https://www.tensorflow.org/datasets/catalog/ref_coco), [COCO download](https://cocodataset.org/#download) | Reuse current `RefCOCO` source and `./data/converted` | Matching view now reuses the `RefCOCO` converted root and excludes MMEB-V2 overlap by `ann_id`; keep this row as the provenance entry |

## `OpenGVLab/VideoChat2-IT` Subset Split and Overlap Decision

`OpenGVLab/VideoChat2-IT` is a gated annotation entry. Its card reports about 1.9M JSON annotations, and the repository mainly stores JSON rather than full source media. The first practical source should be the video-only `byminji/VideoChat2-IT-clean` manifest; it is a community cleaned manifest whose local 16 `train.json` files contain 877,489 samples in total, matching the upstream per-subset table, while the upstream README total field says 874,869. The current production wiring does not enable the full manifest as one dataset entry; it writes only rows that resolve to local production media roots into the `videochat2_it_train` split, while no-local-media subsets remain audit-only.

| Subset | Source Family | Task View | Valid Samples | Relation to Existing Data | Decision |
|--------|---------------|-----------|--------------:|---------------------------|----------|
| `textvr` | TextVR | video caption | 39,648 | No known production-root overlap yet | Keep as candidate |
| `youcook2` | YouCook2 | video caption | 8,700 | Same family as the existing `youcook2_train`; part of the row-level media resolves locally | Enable the 4,901 resolved rows; write the rest to bad media |
| `k710` | Kinetics | video classification | 38,977 | High overlap risk with `kinetics700_train` / `K700`; local K700 frame units are reused by clip basename | Enable the 29,844 resolved rows; write the rest to bad media |
| `ssv2` | Something-Something-V2 | video classification | 40,000 | Same family as the existing `ssv2_train`; media resolves locally | Enable 40,000 rows |
| `videochat2` | InternVid | video conversation | 9,584 | No known production-root overlap yet | Keep as candidate |
| `videochatgpt` | ActivityNet | video conversation | 13,303 | Shares the local ActivityNet full-video source with `ActivityNet-Captions`; this root now owns sampled full-video frame units | Enable, expanded by QA turn to 86,707 rows |
| `next_qa` | NExT-QA | video reasoning | 34,132 | Same family as the existing `nextqa_train`; media resolves locally | Enable 34,132 rows |
| `clevrer_qa` | CLEVRER | video reasoning | 40,000 | No known training-root overlap; possible MVBench eval-family source overlap | Keep as candidate after MVBench overlap audit |
| `clevrer_mc` | CLEVRER | video reasoning | 40,000 | No known training-root overlap; possible MVBench eval-family source overlap | Keep as candidate after MVBench overlap audit |
| `ego_qa` | EgoQA | egocentric video QA | 7,797 | No production Ego4D / EgoQA root exists; usable as an EgoSchema proxy without adding Ego4D full | Keep as a small candidate |
| `tgif_frame_qa` | TGIF | video QA | 39,149 | No known training-root overlap; possible MVBench eval-family source overlap | Keep as candidate after MVBench overlap audit |
| `tgif_transition_qa` | TGIF | video QA | 52,696 | No known training-root overlap; possible MVBench eval-family source overlap | Keep as candidate after MVBench overlap audit |
| `webvid` | WebVid | video caption | 399,740 | Possible source overlap with LLaVA-Hound / ShareGPTVideo | Stage 1 sampled only after dedup |
| `videochat` | WebVid | video caption | 6,889 | Possible source overlap with LLaVA-Hound / ShareGPTVideo | Decide after audit |
| `videochat1` | WebVid | video conversation | 4,300 | Possible source overlap with LLaVA-Hound / ShareGPTVideo | Decide after audit |
| `webvid_qa` | WebVid | video QA | 99,954 | Possible source overlap with LLaVA-Hound / ShareGPTVideo | Stage 1 sampled only after dedup |

## P1 Sources

| Dataset | Primary Source | Auxiliary Source | Local Target / Reuse Path | Next Audit |
|---------|----------------|------------------|---------------------------|------------|
| `LaSCo` | [LaSCo official repo](https://github.com/levymsn/LaSCo) | [COCO images](https://cocodataset.org/#download), [VQA2.0](https://visualqa.org/) | Suggested `./data/converted` | Download train / validation query annotations; confirm COCO train2014 / val2014 dependencies, license, query-image / target-image ids, and MMEB-V2 overlap keys |
| `SNLI-VE` | [SNLI-VE official repo](https://github.com/necla-ml/SNLI-VE) | [HuggingFaceM4/SNLI-VE](https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE) | Found `./data/converted` | Recheck train / dev / test splits, Flickr30K image dependency, and label mapping; define entailment positives and contradiction hard negatives |
| `MSRVTT-QA` | [xudejing video-question-answering](https://github.com/xudejing/video-question-answering) | Reuse existing `MSR-VTT` video root | Reuse `./data/converted` | Download QA annotations only; confirm MSR-VTT video id mapping, train split, and overlap with `MSR-VTT` retrieval eval |
| `MSVD-QA` | [xudejing video-question-answering](https://github.com/xudejing/video-question-answering) | [LAVIS MSVD-QA card](https://github.com/salesforce/LAVIS/blob/main/dataset_card/msvd_qa.md), reuse existing `MSVD` video root | Reuse `./data/converted` | Download QA annotations only; confirm `youtube_mapping.txt`, split, and overlap with `MSVD` eval |
| `ActivityNet-QA` | [MILVLG/activitynet-qa](https://github.com/MILVLG/activitynet-qa) | [ActivityNet download](https://activity-net.org/download.html), reuse existing `ActivityNet-Captions` root | First reuse `./data/converted` | Download QA JSON; confirm ActivityNet v1.3 video coverage, alignment with existing frame units, and MMEB-V2 `ActivityNetQA` overlap |
| `mmE5 synthetic` sampled | [intfloat/mmE5-synthetic](https://huggingface.co/datasets/intfloat/mmE5-synthetic) | [mmE5 code](https://github.com/haon-chen/mmE5) | Suggested `./data/converted` | Use sampled subset only; confirm the three subsets, 93-language distribution, image package `LAION_Synthetic.tar.gz`, deduplication, and MMEB-V2 overlap |

## P2 Sources

| Dataset | Primary Source | Auxiliary Source | Local Target / Reuse Path | Next Audit |
|---------|----------------|------------------|---------------------------|------------|
| `SOP` / Stanford Online Products | [Stanford CVGL resources](https://cvgl.stanford.edu/resources.html) | [JamieSJS/stanford-online-products](https://huggingface.co/datasets/JamieSJS/stanford-online-products) | Found `./data/converted` | Use the HF mirror for schema / split smoke first; then confirm official license, class / product splits, and image-image retrieval targets |
| `DeepFashion In-Shop` | [CUHK In-shop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) | [Marqo/deepfashion-inshop](https://huggingface.co/datasets/Marqo/deepfashion-inshop) | Found `./data/converted` | Use the HF mirror for smoke first; then confirm official image / bbox / landmark / partition files, train / query / gallery split, and overlap with FashionIQ / Fashion200K |

## Not Downloaded for v2

These remain in v3 Stage 1 or separate audits, not this v2 download queue: `LAION`, `DataComp`, `Recap-DataComp`, `CC3M`, `CC12M`, full `WebVid`, `HowTo100M`, full `InternVid`, full `MagicLens`, full `MegaPairs`, `COYO`, and full `WIT`. The smaller `webvid` / `videochat*` slices inside `VideoChat2-IT-clean` are audited separately in the subset table above.
