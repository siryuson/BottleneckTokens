# Runtime Transform 审计

本文档记录对当前训练/评测数据族的运行时 transform 逻辑审计结果。重点核对两类样本：

- raw 存储样本：Lance/Parquet 中的原始文本与媒体字段
- dataset 输出样本：`Dataset.__getitem__()` 或 `build_mmeb_eval_dataset(...)/EvalLanceDataset(transform=...)` 之后的运行时样本

审计目标：

- 确认哪些 runtime transform 是必要的输入规范化
- 确认哪些 runtime transform 已经变成对“已格式化存储”的重复文本变换

## 结论摘要

### 训练数据族

训练侧 runtime transform 目前仍然有必要，原因是 raw 存储并不总是最终模型输入格式：

| 数据族 | 结论 | 主要原因 |
|------|------|------|
| MMEB-train | 保留 | raw query 仍是 `<|image_1|>`，runtime 负责替换成 `<|image_pad|>` |
| ViDoRe | 保留 | raw query 只有自然语言问题，runtime 才补 image token |
| VisRAG | 保留 | raw query 没有 source-specific prompt，positive 也没有目标文本/token |
| LLaVA-Hound | 保留 | raw 指令仍是 conversation/`<video>` 风格，runtime 才转换成检索布局 |

### MMEB eval

`MMEB benchmark` 不应再依赖“共享、无差别”的 eval-time 文本 transform。`strict` 与 `canonical` 都应改为 dataset-owned transform：只在各自数据族显式声明需要时介入，但 `canonical` 的最终行为仍必须遵守旧 canonical 口径。

原因：

- conversion 不应默认固化最终 query/candidate 文本
- `strict` 现在已经转为“原始内容 + 复原 metadata + runtime 复原”
- `canonical` 中历史上已经批准的 visual token / layout / whitespace / family 模板规则仍然有效；Phase 10 要做的是把它们迁到正确层，而不是删除
- `MMEB-V2 eval` image 侧仍然是 composite benchmark materialization：媒体与标注可以分别来自 `MMEB-V2` 和 `MMEB-eval`，但这种 source 组装不等于 runtime text transform

此前最典型的错误是 `MomentSeeker`：

- raw image query 本来是 `<|image_pad|> ...`
- runtime query transform 又按 `video_*` 家族统一补了 `<|video_pad|>`
- 最终变成 `<|video_pad|>\n<|image_pad|> ...`
- 这会在 batch 内留下“裸的 video token”，触发 Qwen2-VL 的 `video tokens != video features`

当前有效口径（叠加 Phase 10 已完成部分）：
- 共享层只保留 validate、materialization 与 common utilities
- 默认 runtime 行为回到各自数据集文件，而不是继续留在共享 family-level formatting 协议中
- token/media mismatch 一律直接 fail-fast
- 无媒体 query 的 legacy token cleanup 仅是历史兼容处理，不是运行时降级
- `strict` 的 image / video / visdoc 已通过 dataset-owned runtime hook 复原原 parser 文本
- `canonical` 的后续工作应以“恢复旧 canonical 完整修复包”为目标；其中唯一新增规则是取消全局强制文末换行，并逐数据集审查是否保留尾换行

下文中的部分 MMEB eval 样本仍记录的是旧存储形态，用来说明为什么需要把默认 runtime 语义收回到数据集文件；它们不应被理解成“旧 canonical 语义已废弃”，而应被当作后续恢复 canonical 最终行为时的审计参照。

## 核查范围

### 训练侧

- `MmebTrainDataset`
- `VidoreTrainDataset`
- `VisragTrainDataset`
- `LlavaHoundTrainDataset`

### 评测侧

- `MMEBBenchmark`
- `RetrievalEvalDataset`
- `MMEB-V2` 的代表子集：
  - `video-tasks/MSR-VTT`
  - `video-tasks/MomentSeeker`
  - `image-tasks/MSCOCO_i2t`
  - `visdoc-tasks/VisRAG_ChartQA`

## 训练数据族样本

### MMEB-train

raw 样本来自 `MMEB-train/data/ImageNet_1K/train.lance`：

```text
qry      = "<|image_1|>\nRepresent the given image for classification\n"
pos_text = "plane, carpenter's plane, woodworking plane"
```

dataset 输出：

```text
query.text    = "<|image_pad|>\nRepresent the given image for classification\n"
query.images  = 1
positive.text = "plane, carpenter's plane, woodworking plane"
positive.images = 0
```

结论：

- 训练侧 runtime 只是把旧 token `<|image_1|>` 规范化为 `<|image_pad|>`
- 该层属于必要规范化，不是重复渲染

相关代码：
`src/vlm2emb/data/datasets/mmeb_train.py`（约第 83 行）

### ViDoRe

raw 样本来自 `vidore_colpali_train_set/data/train.lance`：

```text
query  = "Comparing panels a, b, c, and d, which statement best describes the data variance?"
answer = "D"
```

dataset 输出：

```text
query.text    = "<|image_pad|> Comparing panels a, b, c, and d, which statement best describes the data variance?"
query.images  = 1
positive.text = "D"
positive.images = 0
```

结论：

- raw 存储没有 image token
- runtime 负责把单图 query 转成模型输入格式

相关代码：
`src/vlm2emb/data/datasets/vidore.py`（约第 89 行）

### VisRAG

raw 样本来自 `openbmb_VisRAG-Ret-Train-In-domain-data/data/train.lance`：

```text
query  = "which are the countries with Chinese speaking population other than China according to this infographic?"
source = "InfoVQA"
```

dataset 输出：

```text
query.text =
"This query is related to retrieving an infographic ... according to this infographic?"

positive.text =
"An infographic with structured data, charts, and annotations. <|image_pad|>"
```

结论：

- source-specific prompt 与 positive 侧 image token 都是 runtime 注入
- raw 存储不是最终输入格式，因此该层仍然合理

相关代码：
`src/vlm2emb/data/datasets/visrag.py`（约第 86 行）

### LLaVA-Hound

raw 样本来自 `ShareGPTVideo_train_video_and_instruction/data/video_instruction/train/sft/video_caption_300k.lance`：

```text
human.value = "<video>\nWrite an in-depth depiction of the video, covering all its aspects."
gpt.value   = "The video opens with a view of an indoor court ..."
```

dataset 输出：

`caption_retrieval` 模式：

```text
query.text   = "<|video_pad|>\nWrite an in-depth depiction of the video, covering all its aspects.\n"
query.images = 8
positive.text = "The video opens with a view of an indoor court ...\n"
```

`video_retrieval` 模式：

```text
query.text    = "Find a video that contains the following visual content: The video opens ...\n"
query.images  = 0
positive.text = "Understand the content of the provided video: <|video_pad|>\n"
positive.images = 8
```

结论：

- raw 指令仍是 conversation / `<video>` 形式
- runtime 负责把 conversation 解释成检索任务的 query/positive 布局

相关代码：
`src/vlm2emb/data/datasets/llavahound.py`（约第 105 行）

## MMEB eval 样本

### MSR-VTT（旧存储示例）

raw Lance：

```text
queries[0].text    = "Find a video that contains the following visual content: ..."
queries[0].images  = 0
candidates[0].text = "<|video_pad|>\nUnderstand the content of the provided video.\n"
candidates[0].images = 8
```

结论：

- 旧存储曾直接写入最终 query/candidate 文本
- 这解释了为何共享 benchmark runtime transform 会形成重复文本变换
- Phase 10 的目标不是继续固化这种做法，而是在保留旧 canonical 最终行为的前提下，把这类文本责任收回到 dataset-owned transform

### MSCOCO_i2t（旧存储示例）

raw Lance：

```text
queries[0].text   = "<|image_pad|>\nFind an image caption describing the given everyday image.\n"
queries[0].images = 1
candidates[0].text = "2 Zebras standing next to each other in plaines."
candidates[0].images = 0
```

结论：

- 旧存储直接携带 image token
- 这说明 image family 的 runtime text transform 需要按数据族显式声明，不能在 benchmark 层无差别补 token

### VisRAG_ChartQA（旧存储示例）

raw Lance：

```text
queries[0].text   = "Find a document image that matches the given query: ..."
queries[0].images = 0
candidates[0].text = "<|image_pad|>\nUnderstand the content of the provided document image.\n"
candidates[0].images = 1
```

结论：

- 旧存储已把 query prompt 和 candidate token/prompt 写好
- 未来 canonical 不应再把这类 prompt/token 固化为 conversion 默认职责
- runtime 是否注入这些文本，应由具体数据族和 profile 显式决定

### MomentSeeker

`queries.lance` 的模态分布：

| 类型 | 数量 |
|------|------|
| `video` query | 400 |
| `image` query | 400 |
| `text` query | 802 |
| `mixed token` query | 0 |

raw Lance 示例：

```text
video query:
  "<|video_pad|>\nFind the clip that corresponds to the given sentence and video segment: ..."
  n_images = 8

image query:
  "<|image_pad|>\nSelect the video clip that aligns with the given text and image: ..."
  n_images = 1

text query:
  "Find the clip that corresponds to the given text: ..."
  n_images = 0
```

经过 `build_mmeb_eval_dataset(..., transform_kwargs={"runtime_mode": ...})` 后：

```text
image query:
  "<|video_pad|>\n<|image_pad|>\nSelect the video clip that aligns with the given text and image: ..."
```

结论：

- raw Lance 完全正确，没有 mixed token
- mixed token 是 benchmark runtime transform 注入出来的
- 这是 `video tokens != video features` 的直接原因

## 当前 eval-time transform 清单

当前 eval 侧文本 runtime transform 使用 transform 命名 helper；旧 `render.py` 兼容重导出已删除：

1. `transform_eval_sample_with_media_token()`
   - 按 `images` 是否非空补媒体 token
   - 位置：`src/vlm2emb/data/datasets/transforms.py`
   - 旧 `render_eval_sample_with_media_token()` 兼容路径已删除

2. `transform_eval_sample_with_text_prefix()`
   - 补 query prefix
   - 位置：`src/vlm2emb/data/datasets/transforms.py`
   - 旧 `render_eval_sample_with_text_prefix()` 兼容路径已删除

3. `MMEBBenchmark` + `build_mmeb_eval_dataset()`
   - benchmark 只负责把 subset、artifact path 和 `transform_kwargs` 传给 MMEB-V2 runtime builder
   - parser 模块通过 `build_query_transform()` / `build_candidate_transform()` 拥有 subset-specific text transform
   - 位置：`src/vlm2emb/evaluation/benchmarks/mmeb.py` 与 `src/vlm2emb/data/datasets/mmeb_v2/loader.py`

## 建议

对 `MMEB benchmark`：

- 禁用 eval-time 文本 transform
- 直接使用 Lance 原文作为 query/candidate 文本
- 允许继续保留 benchmark materialization 所需的数据事实修复与 composite source 组装
- 保留非文本运行时逻辑：
  - 图片解码
  - 视频帧采样
  - `_index` 追踪

换句话说：

- 应禁用：`query_transform/candidate_transform` 这类改文本、补 token、补 prefix 的逻辑
- 应禁止把“统一补齐或移除文末换行”继续放在 eval conversion 的默认职责里
- 应保留：媒体解码与帧采样

训练侧 runtime transform 暂不建议一起删除，因为训练 raw 存储并未全部达到最终输入格式。
