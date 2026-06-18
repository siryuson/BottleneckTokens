# 数据集格式规范

本文档定义 BToks 框架中训练数据集和评测数据集的标准格式，并明确 source、conversion、runtime 三层边界，为新数据集接入和未来扩展提供统一准则。

## 1. 项目级数据准则

### 1.1 三层职责边界

所有数据集接入都应先区分三层职责：

1. **source 层**
- 原始下载来源、原始 split、原始文件布局
- 不承担模型语义，不注入 prompt / token

2. **conversion 层**
- 聚合原始源
- 修复数据事实问题：source freeze、路径归一、坏样本处理、qrels/label/page-index 修复、ID 稳定化
- 产出稳定 Lance 存储；基础训练转换默认保留 raw 字段，不生成 manifest、README、dataset_infos 或 split inventory

3. **runtime dataset 层**
- 从 Lance 读取样本
- 在不改变数据事实的前提下完成运行时变换
- 负责 prompt/token/newline/layout 等可调输入格式

原则：
- **数据事实在 conversion 层固定**
- **输入呈现在 runtime 层可调**
- **所有正式 Lance 转换脚本统一使用 `max_bytes_per_file=2 * 1024 * 1024 * 1024`**
- **禁止再使用 `max_rows_per_file` 作为 Lance 文件切分策略；`max_rows_per_group` 仅可作为内部 row group 调优参数**

### 1.2 MMEB-V2 eval 作为特例

MMEB-V2 不是普通“原样转存”数据集，而是：
- 来自多个原始来源的聚合结果
- 存在历史版本漂移、路径错配、坏样本和 qrels 问题
- strict / canonical 语义需要被明确冻结

因此 MMEB-V2 允许在 conversion 层做一次性数据修复，以保证存储产物稳定可靠；但 `strict` 与 `canonical` 都不应再把 prompt/token/newline/layout 这类 model-facing 呈现默认固化在 conversion 层。

补充说明：

- `MMEB-V2 eval` 的 benchmark materialization 允许组合多个 source；image 侧尤其需要同时承认 `MMEB-V2` 媒体源与 `MMEB-eval` 标注源。
- 这种 composite source 组装属于 benchmark/data-fact 边界，而不是 runtime text transform。
- `strict` 存储应保留原始内容与 runtime 复原所需 metadata；`canonical` 存储应保留数据事实 canonicalization 结果。二者最终的 prompt/token/newline/layout 都归 runtime 所有。

### 1.3 新数据集接入总原则

- 每个新数据集都应有**独立的数据集入口**，便于独立维护和特例处理。
- 但底层应优先复用**共享 schema、task archetype helper 与 thin dataset wrapper**，不要再把 family 基类层级继续做厚。
- retrieval 评测的标准存储格式固定为：`queries + candidates + qrels`。
- conversion 工具负责“存储正确”，dataset runtime 负责“输入可调”。
- 如果某项处理的主要作用是改变模型看到的文本格式，而不是修复数据事实，则不应默认放在 conversion。

### 1.4 family 与 archetype 的职责区别

当前项目中，`family` 与 `task archetype` 不是同一层概念：

- `family`：更偏向存储布局、上游来源、局部 helper 复用
- `task archetype`：更偏向 query / candidate / positive / negative / qrels 的最小语义结构

后续共享边界应优先围绕 **task archetype** 建立，而不是重新做一个更大的 family 级共享语义中心。

规则：

- archetype 负责最小共享 contract、runtime view 与 validation primitive
- family 只负责 schema/layout 级复用
- dataset-owned module 继续拥有 source-specific prompt、metadata 映射与历史兼容行为
- mixture 配比、过采样、双向任务展开不属于 archetype 或 family，后续应进入独立 mixture compiler
- ready dataset 的发现应优先通过显式 dataset registry / 配置入口；当前不保留通用 manifest catalog 兼容包

## 2. 训练数据集格式

### 2.0 基础训练 raw storage / runtime transform 边界

基础训练数据集的新主路径是 raw-preserving conversion + runtime transform：

- conversion 层尽量原样保留源表字段，只把大量小媒体文件聚合到 Lance side table。
- conversion 层不负责把不同数据集规整成统一训练字段；可以改名表格、重组 layout、物化派生媒体，但具体 query / positive / negative 语义由 runtime Dataset 解释。
- runtime Dataset 负责按数据集语义构造 `TrainSample`、规范化 visual token、解码媒体，并为每个 side 输出 `MultiModalInput`。
- 视频训练数据的原始视频聚合表统一命名为 `data/videos.lance`；预抽帧派生表统一命名为 `data/frames.lance`。默认训练 runtime 读取 `data/frames.lance`，缺失时应报错，除非配置显式要求 raw-video 兼容路径。
- 已迁移的四个基础训练家族（MMEB、ViDoRe、VisRAG、LLaVA-Hound）不再依赖 `materialize_training_view`；旧 `src/vlm2emb/data/contracts/` 已删除。
- 样本结构验证要么进入统一 validate 边界，要么由 standalone validate 覆盖，不在 runtime 内新增半套临时校验器。
- 仍未迁移的数据集可以临时保留旧 helper；新实现不得继续扩散 legacy `render` / materialized-view 命名。

### 2.1 统一输出格式：TrainSample

所有已迁移训练数据集最终输出 `src/vlm2emb/data/schema.py` 定义的 `TrainSample`：

```python
class MediaInput(TypedDict, total=False):
    kind: str
    content: Any
    metadata: dict[str, Any]

class MultiModalInput(TypedDict):
    text: str
    media: list[MediaInput]

class TrainSample(TypedDict, total=False):
    query: MultiModalInput
    positive: MultiModalInput
    negative: MultiModalInput  # text="" + media=[] 表示无负样本
    metadata: dict[str, Any]
```

`TrainSample` 是顶层训练样本关系；`MultiModalInput` 是每个 side 的统一 `text + media` 输入。`media` 不在 `TrainSample` 顶层，而在 `query`、`positive`、`negative` 各自的 `MultiModalInput` 中。

### 2.2 四种 Lance Schema

#### 1.2.1 mmeb_train（MMEB 图像训练）

raw sample 表 + image side table 结构：

- **sample 表** (`data/{subset}/{split}.lance`)：
  | 列名 | 类型 | 说明 |
  |------|------|------|
  | * | * | 原始 parquet 表字段原样保留，例如 `qry`、`qry_image_path`、`pos_text`、`pos_image_path`、`neg_text`、`neg_image_path` |

- **images 表** (`data/images/{subset}.lance`)：
  | 列名 | 类型 | 说明 |
  |------|------|------|
  | path | string | 原始 parquet 图片路径字符串，也是 runtime join key |
  | image | binary | 图像二进制数据 |

- **实现文件**: `src/vlm2emb/data/datasets/mmeb_train.py`
- **注册名**: `mmeb_train`
- **子集数**: 20 个（ImageNet_1K, N24News, ...）

#### 1.2.2 vidore_train（ViDoRe 文档检索训练）

单表结构：

- **数据表** (`data/train.lance`)：
  | 列名 | 类型 | 说明 |
  |------|------|------|
  | image | struct{bytes: binary, path: string} | 文档页面图像 |
  | query | string | 查询文本 |
  | answer | string | positive 侧答案文本 |

- **实现文件**: `src/vlm2emb/data/datasets/vidore.py`
- **注册名**: `vidore_train`
- **溯源**: 两层（原始 → Lance）

#### 1.2.3 visrag_train（VisRAG 文档检索训练）

单表结构，带 source-specific prompt 模板：

- **数据表** (`data/train.lance`)：
  | 列名 | 类型 | 说明 |
  |------|------|------|
  | query | string | 查询文本 |
  | image | struct{bytes: binary, path: string} | 文档页面图像 |
  | source | string | 数据来源（10 种：NeurIPS Papers, Textbooks, ICML Papers, Manuallib, ArxivQA, ChartQA, MP-DocVQA, InfoVQA, PlotQA, SlideVQA） |

- **Prompt 模板**: 根据 source 字段选择不同的 prompt 前缀
- **实现文件**: `src/vlm2emb/data/datasets/visrag.py`
- **注册名**: `visrag_train`
- **溯源**: 两层（原始 → Lance）

#### 1.2.4 llavahound_train（LLaVA-Hound 视频训练）

双表结构：

- **帧表** (`data/train_300k.lance`)：
  | 列名 | 类型 | 说明 |
  |------|------|------|
  | video_id | string | 视频 ID |
  | frame_idx | string | 帧顺序键 |
  | image | binary | 帧图像二进制数据 |

- **指令表** (`data/video_instruction/{subset}.lance`)：
  | 列名 | 类型 | 说明 |
  |------|------|------|
  | id | string | 样本 ID |
  | video | string | 关联帧表的 `video_id` |
  | conversations | list[struct] | 对话列表 |

- **三种模式**:
  - `caption_retrieval`: 视频→字幕检索（video_caption_300k）
  - `video_retrieval`: 字幕→视频检索（video_caption_300k-video）
  - QA 模式: 问答对（video_qa_240k）
- **实现文件**: `src/vlm2emb/data/datasets/llavahound.py`
- **注册名**: `llavahound_train`
- **溯源**: 两层（原始 → Lance）

### 2.3 多模态 Token 常量

定义于 `src/vlm2emb/data/datasets/const.py`：

| 常量 | 值 | 用途 |
|------|-----|------|
| `STANDARD_IMAGE_TOKEN` | `<\|image_pad\|>` | 标准格式（Qwen），新数据集 SHALL 使用此 token |
| `PHI3V_IMAGE_TOKEN` | `<\|image_1\|>` | MMEB 原始格式（Phi-3V），仅用于兼容 |
| `STANDARD_VIDEO_TOKEN` | `<\|video_pad\|>` | 视频帧 token |

> **规则**:
> - 项目级 runtime text formatting 优先使用 `STANDARD_IMAGE_TOKEN` / `STANDARD_VIDEO_TOKEN`
> - 历史 token（如 `PHI3V_IMAGE_TOKEN`）只作为兼容输入保留
> - `MMEB-V2 eval` 不应再把 legacy token 替换、token 布局整理、文末换行策略等规则默认固化在 conversion；这些模型可见输入形态统一按下文 runtime surface 规则管理
> - 非 MMEB-V2 的新数据集默认保持原始内容与 schema，不默认引入 strict/canonical token 语义

### 2.4 runtime surface 规则

本节合并原 `runtime-surface-rules.md`。它定义模型最终看到的 **text/media surface**，包括：

- visual token 是否出现、出现在哪里
- visual token 与正文之间使用空格还是换行
- instruction 和 body 的连接方式
- 文末是否保留换行
- 选项块、段落块和尾部空白如何保留

适用范围：

- 评测数据的 runtime text transform 与 canonical 收敛
- 训练数据的 runtime text transform 与 prompt 物化
- 数据转换与 runtime 的职责划分讨论
- 新数据集接入时对 text/media surface 的审查

不适用的理解：

- 不是所有训练数据都必须机械遵守同一 surface
- 不是所有 parser 都必须共享同一实现
- 不是所有 subset 都必须共享同一默认值

训练侧可以把本节作为默认参考基线，并在不破坏 contract 的前提下做受控扰动，例如随机切换部分 style 型开关、采样不同 instruction/body separator，或对可等价 surface 做轻量随机化以增强鲁棒性。但训练侧不能破坏 runtime invariant，尤其不能破坏 `visual_token_alignment`。

#### 2.4.1 origin 与 plan canonical

`origin` 的参考对象是 **原始 parser 行为**，不是 Lance 原始字段逐列透传。

也就是说：

- 如果原始 parser 会拼 prompt，就以拼出的 prompt 为 `origin`
- 如果原始 parser 会补一个换行，就以补过换行后的 surface 为 `origin`
- 如果原始 parser 用 dataset-local instruction，而不是 raw `qry_instruction`，就以 parser 结果为准

后续 dataset 级默认值只保留两套口径：

- `origin`
- `plan canonical`

不再长期维护“当前 canonical / 当前 non-origin”作为第三列。如果当前实现和 `plan canonical` 还不一致，应在说明中明确写出，但不再为此扩展长期表结构。

#### 2.4.2 核心规则词表

当前项目级规则词表固定为 10 条：

| 规则名 | 说明 |
| --- | --- |
| `trailing_newline` | 文本末尾是否保留换行，或保留到什么程度 |
| `whitespace_normalization` | 多余空格、脏 join、尾随空格等空白归一化 |
| `blankline_trim` | 多余空行裁剪 |
| `instruction_body_separator` | instruction 与 body 的连接方式 |
| `choice_prefix_cleanup` | `(A)`、`A.`、`1.` 这类选项前缀是否去除 |
| `terminal_format_artifact_cleanup` | 模板尾巴、尾随标点或尾随空格这类格式残留清理 |
| `visual_token_variant_normalization` | `<\|image_1\|>` / `<image>` / `<video>` 等 token 变体归一 |
| `visual_token_alignment` | token 与 media 的数量、类型、顺序一致性 |
| `visual_token_placement` | visual token 是前置、独占一行，还是嵌入正文 |
| `visual_token_separator` | visual token 与正文之间使用空格还是换行 |

`visual_token_alignment` 不是普通风格开关，而是 **runtime invariant**。它要求：

- token 数量与 media slot 数量对齐
- token 类型与 media 类型对齐
- token 顺序与 media 顺序对齐

因此：

- 有 media 但缺 token：必须补齐或 fail-fast
- 无 media 但残留 token：必须移除或 fail-fast
- token 重复：视为 alignment 破坏

训练与评测都不能关闭这条 invariant。

#### 2.4.3 plan canonical 的项目级结论

当前最重要的 `plan canonical` 结论有两条。

第一，visual token 如果不是嵌入正文语义的一部分，而只是独立模态占位符，应优先独占一行：

```text
<|image_pad|>
正文...
```

或：

```text
<|video_pad|>
正文...
```

而不是：

```text
<|video_pad|> 正文...
```

第二，只要正文已经形成完整 prompt 块，无论它是单句 instruction、多行选项块、对话块，还是紧凑型 `(A) yes; (B) no.` 结构，`plan canonical` 都应优先收敛到：

- `trailing_newline=ensure_single`

例外：

- 如果 `origin` 本身就存在混合尾部状态，`origin` 应保持 `preserve`
- 某些特殊 query（如 `VisDial.query`）更适合 `normalize_blank_tail`

#### 2.4.4 trailing_newline 四种状态

`trailing_newline` 当前统一分成四类：

| 状态 | 含义 |
| --- | --- |
| `preserve` | 保持原始 parser / 原始内容的尾部状态，不删也不增 |
| `strip` | 明确要求末尾没有换行 |
| `ensure_single` | 明确要求末尾恰好一个换行 |
| `normalize_blank_tail` | 允许清理多余尾部空白块，但保留“收尾感” |

当前唯一明确属于 `normalize_blank_tail` 的典型例子是 `VisDial.query`。

#### 2.4.5 当前大致默认判断

这一节是项目级默认判断，不代替 dataset 级细表。

**image classification / image QA**

- `query`
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- `candidate`
  - 通常不涉及 `visual_token_separator`
  - `trailing_newline=preserve`

**image retrieval / grounding**

- image-backed prompt/query 多数趋向：
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- template 型 image candidate 多数趋向：
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
  - 如果 candidate 正文是 parser 或配置提供的通用 instruction，应写成完整句子，例如 `Represent the given cropped image of the object.`，不要保留无句号的句片。
- 类别名、单词或短语型 candidate 多数趋向：
  - `trailing_newline=strip`
- 完整句子、多句 caption，或由 instruction 拼接出的 candidate 多数趋向：
  - `trailing_newline=ensure_single`
- 处于中间的 answer / caption 不按长度机械归类，应在对应数据集配置和样本 review 中确认。

关键例外：

- `VisDial.query`
  - `instruction_body_separator=newline`
  - `trailing_newline=normalize_blank_tail`
- `VisDial.candidate`
  - `origin` 保持单行 image token + instruction
  - `plan canonical` 收敛到 image token 独占一行 + 单个结尾换行
- `WebQA / EDIS.query`
  - 不再把“text-only query 去 visual token”视为 canonical 风格项
  - 应作为 `visual_token_alignment` invariant 修复，并记录 `surface_repairs` provenance
- `VizWiz.query`
  - `plan canonical` 额外开启窄范围 `whitespace_normalization`
  - 只清理尾随空格，不改变 parser-owned 单个结尾换行

MMEB-train 复用上述 surface 规则作为新 Dataset 输出标准。当前已确认的训练侧特例：

- `WebQA.query`
  - text-only query 中残留 legacy image token 属于 `visual_token_alignment` 修复
  - `trailing_newline=ensure_single`
- `ChartQA.positive` / `ChartQA.negative`
  - 对答案文本启用窄范围 `whitespace_normalization`
  - 只清理尾部水平空白，不改动换行或答案内容
- `MSCOCO.positive`
  - 属于 template 型 image positive
  - visual token 独占一行
  - `trailing_newline=ensure_single`
- `VisDial.query`
  - `instruction_body_separator=newline`
  - `trailing_newline=normalize_blank_tail`
- `VisDial.negative`
  - 如果 negative 侧带图片但原始文本为空，恢复为 image token 独占一行
  - `trailing_newline=ensure_single`

**video classification**

- `query`
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- `candidate`
  - 多为短 label
  - `trailing_newline=strip`

典型形态：

```text
<|video_pad|>
Recognize the category of the video content.
```

**video QA**

- `query`
  - `visual_token_separator=newline`
  - `trailing_newline=ensure_single`
- `candidate`
  - 多为答案文本
  - `trailing_newline=strip`

`Video-MME / MVBench / NExTQA` 的内部选项换行属于正文结构，必须保留。这里讨论的是文末换行，不是内部选项块换行。

**video retrieval / moment retrieval**

- `MSR-VTT / MSVD / DiDeMo / VATEX / YouCook2`
  - `query.trailing_newline=ensure_single`
  - `candidate.visual_token_separator=newline`
  - `candidate.trailing_newline=ensure_single`
- `QVHighlight / Charades-STA / MomentSeeker`
  - `query.visual_token_separator=newline`
  - `query.trailing_newline=ensure_single`
  - `candidate.visual_token_separator=newline`
  - `candidate.trailing_newline=ensure_single`

**visdoc**

- `query`
  - `origin` 多数更接近 `preserve`
  - `plan canonical` 收敛为 `trailing_newline=ensure_single`
- `candidate`
  - `origin` 多为 `visual_token_separator=space, trailing_newline=strip`
  - `plan canonical` 多为 `visual_token_separator=newline, trailing_newline=ensure_single`

关键例外：

- `MMLongBench-page`
- `MMLongBench-doc`

这两个 subset 的原始 query 全量存在混合尾部状态，因此当前只冻结：

- `origin=query preserve`
- `plan canonical=query ensure_single`

#### 2.4.6 与 origin 不同的处理必须显式记录

凡是与 `origin` 不同的 runtime surface 处理，都必须：

1. 显式命名
2. 显式注释
3. 显式写入文档
4. 精确到 dataset / subset

不能长期保留：

- 隐式 helper 顺手修一下
- 未注释的 parser 特判
- loader 层统一偷偷改写 surface

### 2.5 MMEB-V2 eval 的 strict / canonical 关系

- 本节仅适用于 `MMEB-V2 eval`。
- `strict` 是 MMEB-V2 转换与评测的参考口径，目标是通过 runtime 可审计地复现 the public reference parser contract 对应 parser 的输出语义与文本/图像内容。
- `canonical` 是 MMEB-V2 eval 的正式 reprocessed 分支；它不应被推广为普通训练数据或普通评测数据的默认存储语义。
- `strict / canonical` 可以继续承载 benchmark materialization 所需的数据事实修复；这些修复属于 conversion 层。
- prompt、token layout、newline、option layout、whitespace 这类模型可见输入形态，归属 runtime surface 层，遵循上文 `2.4 runtime surface 规则`。
- `strict` conversion 应保留原始内容、媒体、qrels、identity 与 runtime 复原所需 metadata；`strict` runtime 负责复原原 parser 可见文本。
- `canonical` 的 conversion / runtime 可以重新拆分职责，但两者联合后的最终输出必须遵守已记录的 canonical 数据事实修复与 runtime surface 规则。
- 训练侧与其它普通评测数据集不使用这一套 strict/canonical 存储语义；它们默认保留原始内容，并将输入格式变换放在 runtime 层调节。

允许的 MMEB-V2 canonical **数据事实修复**包括：

- source freeze 与 composite-source materialization
- 坏样本剔除、重复样本去重、稳定 ID 固定
- qrels / label / page-index / candidate identity 修复
- 路径别名解析：仅在指向同一资产时允许本地目录 alias

不允许的 MMEB-V2 canonical 修复包括：

- 未经文档冻结的临时 query / candidate wording 改写或模板化
- 未经审查的 prompt / token / newline / 空行临时改写
- 候选顺序重排
- qrels 语义重写

### 2.6 当前 MMEB-V2 数据事实修复覆盖与迁移说明

以下内容属于当前仍然有效、且应继续保留在 conversion 层的数据事实修复：

| 类别 | 数据集 / 范围 | canonical 行为 |
|------|---------------|----------------|
| 数据 | `ViDoSeek-page` | 使用 `siyrus/BToks-ViDoSeek-page-fixed`，应用 page-index 修复 |
| 数据 | `MMLongBench-page` | 使用 `siyrus/BToks-MMLongBench-page-fixed`，应用 page-index 修复 |
| 数据 | `RefCOCO-Matching` | 对重复 `candidate_id` 的 qrels 进行去重 |
| 数据 | `MomentSeeker` | 固定到去重后的 `siyrus/BToks-MomentSeeker` 1.6k 官方版本 |
| 数据 | `MMLongBench-doc` | 明确保持 graded qrels，不切换到 binary qrels |
| 数据 | `ViDoRe_esg_reports_human_labeled_v2` | 优先使用 `VLM2Vec` 镜像源 |
| 路径 | `video_retrieval` 家族 | 对解包后的 frame 路径做同语义归一，不回退到 raw video |

以下内容已经从“conversion 数据事实修复”中拆出，按 runtime surface 规则处理：

| 范围 | 当前归属 |
|------|----------|
| `WebQA`、`EDIS` text-only query 中残留 legacy image token | `visual_token_alignment` invariant 修复；不是 canonical-only 风格规则 |
| `VisDial` query 的 instruction/body 换行 | `instruction_body_separator` + `blankline_trim` / `normalize_blank_tail` |
| legacy token 的模型可见替换 | `visual_token_variant_normalization` |
| 独立 token 的文本布局整理 | `visual_token_separator` / `visual_token_placement` |
| 文末换行、额外空行、脏 join | `trailing_newline` / `blankline_trim` / `whitespace_normalization` |

### 2.7 MMEB-V2 上游问题覆盖状态

当前 `MMEB-V2 eval` 的 `strict / canonical` 已覆盖的公开上游问题如下：

| 来源 | 问题 | 当前处理 |
|------|------|----------|
| VLM2Vec `#167` | `ViDoSeek-page` / `MMLongBench-page` page index bug | `strict` 保留原始行为；`canonical` 切到 fixed source 并应用修复 |
| MMEB-V2 discussion `#5` | `MomentSeeker` 的 `query_images` / `video_frames` 路径敏感 | `strict` 与 `canonical` 都固定为 `frame-only`，并保持路径语义兼容 |
| MMEB-V2 discussion `#9` | `video_ret.tar.gz` 解包后带前缀目录 | `strict` 允许路径 alias；`canonical` 做同语义 frame path 归一 |
| `MomentSeeker` 官方双版本 | `1.8k` 与去重 `1.6k` source ambiguity | `strict` 冻结 source；`canonical` 固定到 `siyrus/BToks-MomentSeeker` |
| VLM2Vec `#188` | `esg_reports_human_labeled_v2` 上传图片数与 corpus 不一致 | `strict` 以 parser/source-of-truth 为准；`canonical` 优先使用 `VLM2Vec` 镜像源 |
| MMEB-V2 discussion `#8` | VisDoc 截断/坏图 | `strict` 不 silent 替换媒体；`canonical` 通过更稳定 source 规避已确认问题，但不擅自重写媒体内容 |

论文或榜单口径差异不属于当前转换层修复范围。

### 2.8 训练转换工具边界

训练数据转换工具当前保留在 `scripts/convert/train/`。

正式保留的 Python 转换器：

| 脚本 | 角色 |
|------|------|
| `convert_mmeb_train_to_lance.py` | MMEB-train 转 Lance |
| `convert_parquet_to_lance.py` | parquet 数据集转 Lance |
| `convert_jsonl_to_lance.py` | JSONL 数据集转 Lance |
| `convert_video_frames_to_lance.py` | 视频帧目录转 Lance |
| `convert_llavahound_train_to_lance.py` | LLaVA-Hound raw-preserving 双表转换 |

约束：
- 这些脚本负责**训练数据**的存储层转换，不负责 MMEB 评测 strict/canonical 逻辑。
- 历史 shell 包装器主要是机器路径包装器，已从正式工具层移除。
- 新训练转换逻辑优先进入 Python 工具层，不再继续扩散硬编码 shell 入口。
- 正式 `scripts/convert/**` Python 转换器写 Lance 时，统一显式传入 `max_bytes_per_file=2 * 1024 * 1024 * 1024`。
- 训练转换不得再使用 `max_rows_per_file` 控制物理文件切分；如需调优，仅允许保留 `max_rows_per_group`。

## 3. 评测数据集格式

### 3.1 MMEB-V2 统一 Lance 三文件布局

本节描述的是 `MMEB-V2 eval` 的固定存储布局，而不是项目中所有评测数据集的通用硬约束。

`MMEB-V2 eval` 的 78 个评测子集共享统一目录结构：

```
{dataset_name}/
├── queries.lance/        # 查询表
├── candidates.lance/     # 候选表
├── qrels.lance/          # 相关性判定表
└── metadata.json         # 元数据
```

补充约束：
- `queries.lance`、`candidates.lance`、`qrels.lance` 都属于 Lance dataset 目录；其底层物理数据文件统一限制为单文件不超过 `2GB`
- `MMEB-V2` 转换不得再依赖 `max_rows_per_file` 控制文件切分

运行时读取约束：
- `MMEB-V2 eval` 不再通过通用 schema 扫描猜列；当前主路径由 `src/vlm2emb/data/datasets/mmeb_v2/loader.py` 中的 `build_mmeb_eval_dataset(...)` 负责
- loader 会先按 subset 选择 parser 文件，再按 parser 声明精确读取 query/candidate 所需原始列
- benchmark 只透传 `runtime_mode`、`num_frames` 等运行时参数；prompt、newline、option layout、token separator 等 parser 语义下沉到各自 parser 文件

#### queries.lance

| 列名 | 类型 | 说明 |
|------|------|------|
| id | string | 查询唯一标识 |
| text | string | 查询文本（含指令） |
| images | list[binary] | 查询图像列表（可为空） |

#### candidates.lance

| 列名 | 类型 | 说明 |
|------|------|------|
| id | string | 候选唯一标识 |
| text | string | 候选文本 |
| images | list[binary] | 候选图像列表（可为空） |

#### qrels.lance

| 列名 | 类型 | 说明 |
|------|------|------|
| query_id | string | 关联的查询 ID |
| mode | string | 评测模式：`"sparse"` 或 `"exhaustive"` |
| candidate_ids | list[string] | 相关候选 ID 列表 |
| candidate_scores | list[float32] | 对应的相关性分数 |

#### metadata.json

```json
{
    "name": "ImageNet-1K",
    "task_type": "image_classification",
    "num_queries": 1000,
    "num_candidates": 1000
}
```

### 3.2 评测模式（qrels.mode）

评测模式是 **per-query 级别**的，同一数据集内不同 query 可能有不同 mode：

| mode | 含义 | 评测方式 |
|------|------|---------|
| `"sparse"` | 全局评测 | query 与**所有** candidates 排序 |
| `"exhaustive"` | 局部评测 | query 仅与其 `candidate_ids` 中的候选排序 |

### 3.3 task_type 枚举

| 模态 | task_type | 数据集数量 |
|------|-----------|-----------|
| image | `image_classification` | 10 |
| image | `image_question_answer` | 10 |
| image | `image_retrieval` | 12 |
| image | `image_visual_grounding` | 4 |
| video | `video_classification` | 5 |
| video | `video_question_answer` | 5 |
| video | `video_retrieval` | 5 |
| video | `video_moment_retrieval` | 3 |
| visdoc | `visdoc_vidore_v1` | 10 |
| visdoc | `visdoc_vidore_v2` | 4 |
| visdoc | `visdoc_visrag` | 6 |
| visdoc | `visdoc_ood` | 4 |

### 3.4 聚合指标

| 模态 | 聚合指标 |
|------|---------|
| image | `hit@1` |
| video | `hit@1` |
| visdoc | `ndcg@5` |

## 4. 命名约定

| 场景 | 格式 | 示例 |
|------|------|------|
| 训练子集（配置文件中） | 下划线 | `ImageNet_1K`, `VisualNews_t2i` |
| 评测子集（DATASET_REGISTRY 中） | 连字符 | `ImageNet-1K`, `VisualNews_t2i` |
| 文档文件名 | kebab-case | `imagenet-1k.md`, `visualnews-t2i.md` |
| 数据集注册名 | 任务语义化 snake_case | `mmeb_train`, `vidore_train` |
| 评测目录名 | 与 DATASET_REGISTRY 一致 | `ImageNet-1K/`, `MSR-VTT/` |

## 5. 新数据集接入检查清单

### 5.1 训练数据集

- [ ] 选择或创建 Lance schema（参考 2.2 节的 4 种现有 schema）
- [ ] 实现 transform，输出 `TrainSample` 格式
- [ ] 使用 `@AutoDataset.register()` 注册数据集类
- [ ] 在 `src/vlm2emb/data/datasets/__init__.py` 中添加导入
- [ ] 创建 YAML 配置文件（参考 `configs/datasets/mmeb_train.yaml`）
- [ ] 使用 `STANDARD_IMAGE_TOKEN` 而非 `PHI3V_IMAGE_TOKEN`
- [ ] 需要可调输入格式时，优先使用 runtime transform，而不是重新转换存储
- [ ] 默认 runtime transform 是否为 pickle-safe callable
- [ ] 创建溯源文档卡片（在 `docs/*/datasets/train/` 下）

### 5.2 评测数据集

- [ ] 先判断问题属于 source / conversion / runtime 哪一层
- [ ] retrieval 类型是否使用统一的 `queries + candidates + qrels`
- [ ] 数据事实问题（source/media/qrels/id）是否在 conversion 层固定
- [ ] 输入格式问题（prompt/token/newline）是否尽量保留在 runtime transform
- [ ] query/candidate transform hook 是否避免局部 `lambda` / 闭包，能穿过多进程 worker
- [ ] 必要时编写转换脚本（参考 `scripts/convert/convert_*.py`）
- [ ] 转换为统一 Lance 三文件布局（queries.lance / candidates.lance / qrels.lance）
- [ ] 写 Lance 时显式限制 `max_bytes_per_file=2 * 1024 * 1024 * 1024`
- [ ] 生成 metadata.json（包含 name, task_type, num_queries, num_candidates）
- [ ] 在 `DATASET_REGISTRY`（`src/vlm2emb/evaluation/benchmarks/mmeb.py`）中注册
- [ ] 创建溯源文档卡片（在 `docs/*/datasets/eval/` 下）
