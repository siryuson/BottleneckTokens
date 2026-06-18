# QVHighlights 训练数据集

> 查询驱动的视频时刻检索数据集，用于训练“给定整段视频和文本描述，检索对应片段”的能力。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `moment_retrieval` |
| 用途 | 训练 / held-out 验证 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | QVHighlights |
| 论文 | QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries |
| 许可证 | 参见原始 QVHighlights 说明 |

### 样本规模

- `official_train`: `7,218`
- `official_val`: `1,550`
- `official_test`: `1,542`（无正样本窗口；写出为 held-out 审计 split，不作为默认训练正样本视图）

### 本地源状态

- annotation root: `./data/converted`
- 当前视频 root: `./data/converted`
- 当前转换源来自 `yeliudev/VideoMind-Dataset` 的社区 HF 分卷视频展开目录；`videos/` 有 `12,562` 个 mp4，覆盖 annotation 中 `10,148` 个唯一 `vid`，缺失数为 `0`。
- 本地官方命名 tar `./data/converted` 只有约 `219MB`，`gzip -t` 与 `tar -tzf` 均报 EOF，仅能列出少量视频，因此当前不作为转换源。
- 本地还存在 `yeliudev/VideoMind-Dataset` 的三段 `videos.tar.gz.*` 分卷 archive。后续若增加 archive 直读，应先对这些分卷做完整性校验和 provenance 记录，而不是切回已损坏的官方命名 tar。

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `qid` | string / int | 查询 ID |
| `query` | string | 文本描述 |
| `duration` | float | 视频时长相关元数据 |
| `vid` | string | 视频 ID |
| `relevant_windows` | list[list[float]] | 正样本时刻窗口列表 |

### 样本

**query.text**: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: some military patriots takes us through their safety procedures and measures.\n`
**query.images**: 采样后的整段视频帧列表
**positive.text**: `<|video_pad|>\nUnderstand the content of the provided video.\n`
**positive.images**: 从 `relevant_windows` 中选出的正片段帧列表
**negative.text**: 空

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `qvhighlight_train` |
| 实现文件 | `src/vlm2emb/data/datasets/qvhighlight_train.py` |
| 注册名 | `qvhighlight_train` |
| 转换脚本 | `scripts/convert/train/convert_qvhighlight_to_lance.py` |
| 根目录 | `./data/converted` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/{split}.lance.qid` | string | 查询 ID |
| `data/{split}.lance.vid` | string | 视频 ID |
| `data/{split}.lance.query_frame_unit_key` | string | query 侧完整视频 frame unit key |
| `data/{split}.lance.positive_frame_unit_keys` | list[string] | positive 候选片段 frame unit keys，对应 `relevant_windows` |
| `data/{split}.lance.query` | string | 文本描述 |
| `data/{split}.lance.relevant_windows` | list | 全部正样本窗口列表 |
| `data/frames.lance.frame_unit_key` | string | frame unit join key |
| `data/frames.lance.frame_unit_type` | string | `full_video` 或 `segment` |
| `data/frames.lance.frames` | list[binary] | 预抽样 RGB JPEG 帧 |
| `data/frames.lance.sampled_frame_count` | int | 转换阶段保留的帧数 |
| `data/frames.lance.source_total_num_frames` | int | 原始视频总帧数 |
| `data/frames.lance.clip_total_num_frames` | int | 当前 frame unit 覆盖的帧数 |
| `data/frames.lance.clip_start` / `clip_end` | float | 片段秒级边界；完整视频为 `0` |
| `data/frames.lance.sampled_indices` | list[int] | 相对源视频或片段的采样帧索引 |

### 训练侧映射

- query: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: {query_text}\n`
- positive: `<|video_pad|>\nUnderstand the content of the provided video.\n`
- negative: 空文本（占位）
- `num_frames` 控制每个 video media slot 从预抽帧 frame unit 中二次采样返回的帧数；默认路径不在 runtime raw decode 原始视频。
- query 使用 `full_video` frame unit，转换阶段最多保留 64 帧；positive 从 `positive_frame_unit_keys` 中选择一个 `segment` frame unit，转换阶段每个片段最多保留 8 帧。
- 对 positive media，`clip_total_num_frames` 与 `sampled_indices` 指片段内部帧数和片段内索引；源视频上下文保存在 `source_total_num_frames`、`clip_start` 与 `clip_end`。

### Split 策略

- `official_train_without_mmeb_v2_eval`: 默认训练视图
- `official_train`: 原始训练视图，保留用于审计
- `official_val`: held-out 验证视图
- `official_test`: 原始标注缺少正窗口时不保留为可训练 split；需要审计时保留在排除 / audit 表中

### 本地审计结论

- `MMEB-V2 / QVHighlight` 的 `1081` 条唯一 query 与原始 `official_val` 高度对齐
- 与原始 `official_train` 只有极少 query 重叠
- 当前实现因此保留 `official_train / official_val`，默认不把 `official_val` 混入训练
- 当前转换写出 `data/frames.lance`，其中每个视频有一个完整视频 frame unit，并为每个正样本窗口写出 segment frame unit；训练 split 只保留存在 positive window 的 rows。原始 `official_test` 若缺少 positive window，不作为可训练 split 保留，仅进入 audit / exclusion 记录。
