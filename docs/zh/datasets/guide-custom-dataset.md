# 数据集接入指南

本文档说明如何在当前 BToks 中接入新的训练数据集或评测数据集。

> 相关文档：
> - [格式规范](./format-spec.md)
> - [数据处理流水线](./architecture-data-pipeline.md)

## 1. 先判断属于哪一层

新增数据集前，先判断改动属于哪一层：

1. **source 层**
- 原始下载地址
- 原始 split / 原始目录布局
- 不承担模型输入语义

2. **conversion 层**
- 聚合原始源
- 修复数据事实：source freeze、路径归一、坏样本处理、qrels/label/page-index 修复、ID 稳定化
- 产出 Lance 与 metadata

3. **runtime dataset 层**
- 从 Lance 读取样本
- 完成 prompt/token/newline/layout 等运行时可调格式变换
- 不改变已经冻结的数据事实

原则：
- **存储正确性在 conversion 层解决**
- **输入可调性在 runtime 层解决**

## 2. 默认接入策略

### 2.1 每个数据集有独立入口

每个新数据集都应有独立的数据集入口，便于：
- 独立维护
- 独立配置
- 独立记录特例

### 2.2 底层优先复用共享 schema 与 archetype helper

不建议为每个数据集复制整套实现。更推荐：

- 每个数据集有自己的注册名和入口类
- 底层复用按 task archetype 组织的 helper、公共 schema 和 thin dataset wrapper

例如：
- 训练 Lance 数据集优先直接使用 `SampleLanceDataset`；只有加入真实训练专属行为后才引入训练专属子类
- retrieval 评测复用 `RetrievalEvalDataset` 容器与 `EvalLanceDataset`
- `VideoEvalLanceDataset` 仅保留为兼容包装；像 `MMEB-V2 eval` 这类新主路径应优先使用 `EvalLanceDataset + 显式 transform`

新增 dataset 时优先走两步：

1. 先明确 source / conversion / runtime 三层边界
2. 再通过 runtime adapter 输出训练 `TrainSample` 或 retrieval `queries/candidates/qrels`

这样可以保证新增数据集只需要补 thin adapter 或 parser，而不需要改 trainer / evaluator 主干。

当前有效边界是：
- 共享层只保留 validate 与 common utilities
- 数据集特有的默认 runtime 行为由各自数据集文件承载
- 共享层可以减少重复代码，但不再承载 dataset 默认 text transform 策略

### 2.3 共享粒度优先使用 task archetype

后续新增数据集时，优先先判断它属于哪一种 **task archetype**，而不是先去找“是否能挂到某个 family”。

当前优先使用的 archetype 包括：

| archetype | 典型 task_type |
|-----------|----------------|
| `classification` | `image_classification`、`video_classification` |
| `question_answer` | `image_question_answer`、`video_question_answer`、`document_question_answer` |
| `retrieval` | `image_retrieval`、`video_retrieval` |
| `moment_retrieval` | `video_moment_retrieval` |
| `grounding` | `image_visual_grounding` |
| `document_retrieval` | `document_retrieval`、部分 `visdoc_ood` |
| `entity_retrieval` | `multimodal_retrieval`、`entity_retrieval` |

推荐的判断顺序：

1. 先确定 archetype
2. 再决定是否能复用某个 family 的 Lance schema / base class
3. 最后在数据集文件中保留 dataset-owned runtime 逻辑

规则：

- archetype 负责最小共享语义结构
- family 只负责 schema/layout 级复用
- dataset file 继续保留 prompt、metadata 映射、特例逻辑与历史兼容行为
- ready dataset 的发现应优先通过显式 dataset registry / 配置入口；当前不保留通用 manifest catalog 兼容包

## 3. 什么时候需要 conversion

默认情况下，优先把格式变换留在 runtime，避免未来调 prompt/token 时必须重跑数据转换。

但以下情况应放在 conversion 层：

- 需要从多个原始源聚合
- 需要修复媒体路径或坏样本
- 需要修复 qrels / label / page-index
- 需要把零散媒体合并进稳定 Lance 存储
- 对于 `MMEB-V2 eval` 这类特殊数据集，需要冻结 strict / canonical 数据事实

除 `MMEB-V2 eval` 外，其它评测数据集默认不引入 strict/canonical 双存储语义，应尽量保持原始内容与原 schema。

## 4. MMEB-V2 是特例

MMEB-V2 当前不是普通的“原样转存”数据集，而是：
- 多原始源聚合
- 存在历史版本漂移
- 存在路径、媒体、qrels、page-index 等问题

因此 MMEB-V2 允许在 conversion 层一次性修复这些**数据事实问题**，以保证存储稳定。

但即使对 MMEB-V2：
- prompt wording
- token 注入
- newline / whitespace
- 模型特化 text transform

也仍应尽量保留在 runtime 层可调。

## 5. 训练数据集接入

### 5.1 统一输出格式

已迁移训练 Dataset 输出 `src/vlm2emb/data/schema.py` 中的 `TrainSample`：

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
    negative: MultiModalInput
    metadata: dict[str, Any]
```

要求：
- 单个编码对象只保留 `text` 和 `media`
- `query / positive / negative` 是关系层
- `metadata` 保存数据集名、subset、source row 等透传元信息

### 5.2 优先复用现有 Lance schema

当前训练侧正式支持 4 类 Lance schema：

| 家族 | 注册名 | 特征 |
|------|--------|------|
| MMEB | `mmeb_train` | raw sample 表 + image side table |
| ViDoRe | `vidore_train` | 单表：文档图像内嵌 |
| VisRAG | `visrag_train` | 单表：文档图像 + source |
| LLaVA-Hound | `llavahound_train` | 双表：frames + video_instruction |

只有在现有 schema 明显不适用时，才考虑新增 schema。

但即使复用现有 schema，也不代表要复用同一套 family 默认语义。  
是否共用 schema 与是否共享 runtime 语义，应分开判断。

### 5.3 训练 loader 的最小步骤

1. 在 `src/vlm2emb/data/datasets/` 中新增 loader
2. 优先复用 `src/vlm2emb/data/schema.py` 中的 `TrainSample`，不要在 loader 内重新定义 sample schema
3. 用 `@AutoDataset.register(...)` 注册
4. `__getitem__()` 输出 `TrainSample`
5. 在 `src/vlm2emb/data/datasets/__init__.py` 中导入
6. 配置 YAML 指向对应 Lance 根目录

最小示例：

```python
from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import SampleLanceDataset


def transform_my_train_record(row):
    return {
        "query": {"text": row["query"], "media": []},
        "positive": {"text": row["positive"], "media": []},
        "negative": {"text": "", "media": []},
        "metadata": {"dataset_name": "my_train_dataset"},
    }


@AutoDataset.register("my_train_dataset")
class MyTrainDataset(SampleLanceDataset):
    def __init__(self, lance_path: str):
        super().__init__(
            lance_path,
            read_columns=["query", "positive"],
            transform=transform_my_train_record,
        )
```

说明：
- 新 dataset 优先声明 `read_columns`
- `required_columns` 仅为未迁移旧 family 的兼容回退，不再作为推荐写法

## 6. retrieval 评测数据集接入

### 6.1 标准存储格式

retrieval 评测数据集的标准格式固定为：

- `queries`
- `candidates`
- `qrels`

也就是 Lance 三文件布局：

```text
dataset_root/
├── queries.lance
├── candidates.lance
├── qrels.lance
└── metadata.json
```

### 6.2 统一运行时入口

当前 retrieval 评测的统一容器仍是：

- `src/vlm2emb/data/datasets/eval.py` 中的 `RetrievalEvalDataset`

但具体 builder 已按数据集职责拆开：

- 通用 retrieval Lance 读取由 `EvalLanceDataset` / `LanceDataset` 负责
- `MMEB-V2 eval` 通过 `src/vlm2emb/data/datasets/mmeb_v2/loader.py` 中的 `build_mmeb_eval_dataset(...)` 进入 parser-driven runtime
- benchmark 只负责选择 subset、定位 artifact 路径、透传 `runtime_mode`，不再承载 parser-specific 默认值

它支持：
- 图像/视频解码
- 视频帧采样
- query/candidate 级 runtime transform

因此：
- qrels/source/media/id 这类数据事实应在 conversion 层固定
- prompt/token/newline 这类输入格式应尽量在 runtime transform 中完成
- conversion 产物应带 deterministic dataset/sample/relation ID 与 machine-readable manifest
- token/media mismatch 必须直接 fail-fast，不能在 runtime 中静默兜底

### 6.3 运行时可调格式

当前 sample Lance 基类已经支持运行时 transform：

- `src/vlm2emb/data/datasets/base.py` 中的 `SampleLanceDataset`
- `src/vlm2emb/data/datasets/base.py` 中的 `EvalLanceDataset`
- `src/vlm2emb/data/datasets/mmeb_v2/loader.py` 中的 `build_mmeb_eval_dataset(...)`

这允许在不重转 Lance 的前提下，按 query/candidate 分别做：
- token 注入
- whitespace/newline 归一
- 模型特化 text transform

对 `MMEB-V2 eval` 而言，runtime transform 还承担：
- 按 subset 显式声明 query/candidate 所需原始列
- 通过静态 dispatch 挂接 parser 文件
- 在 `origin` / `canonical` mode 间切换 parser 可见文本，而不是回退到共享 heuristic formatter

说明：
- 无媒体 query 的 legacy token cleanup 仅用于历史兼容清理，不是 runtime 降级策略
- 如果样本声明了 visual token 却没有 media，或存在 media 却没有对应 token，仍然按 fail-fast 处理

工程约束：
- 任何会挂到 dataset、benchmark、evaluator，并可能进入 `DataLoader(num_workers>0)` 或多进程训练/评测 worker 的 transform，都必须是 **pickle-safe callable**
- 禁止使用局部 `lambda`、局部闭包作为默认 runtime transform
- 允许的形式应限定为：
  - 顶层函数
  - 顶层 `dataclass` callable
  - `functools.partial(顶层函数, ...)`
- review 这类改动时，除了检查语义是否正确，还必须检查 transform 是否可被 worker 进程序列化

## 7. 训练转换工具层

训练数据转换工具当前保留在 `scripts/convert/train/`。

正式保留的 Python 工具：

| 脚本 | 用途 |
|------|------|
| `convert_mmeb_train_to_lance.py` | MMEB-train -> Lance |
| `convert_parquet_to_lance.py` | parquet -> Lance |
| `convert_jsonl_to_lance.py` | JSONL -> Lance |
| `convert_video_frames_to_lance.py` | 帧目录 -> Lance |
| `convert_llavahound_train_to_lance.py` | raw-preserving LLaVA-Hound -> Lance |

这些脚本只负责**训练数据存储转换**，不负责 MMEB eval strict/canonical 语义。

## 8. 接入检查清单

### 8.1 通用

- [ ] 是否先判断了 source / conversion / runtime 三层职责
- [ ] 是否给新数据集提供了独立入口
- [ ] 是否优先复用了现有共享 schema / archetype helper，而不是继续堆新的 family hierarchy

### 8.2 训练

- [ ] 能否复用现有 4 类 Lance schema
- [ ] Dataset 输出是否为 `TrainSample`
- [ ] 图像是否在 Dataset 中解码，而不是放到 Collator
- [ ] 默认 `sample_transform` 是否为 pickle-safe callable，能穿过多进程 DataLoader

### 8.3 retrieval eval

- [ ] 是否使用 `queries + candidates + qrels` 标准布局
- [ ] 数据事实问题是否在 conversion 层固定
- [ ] prompt/token/newline 是否尽量放在 runtime transform 中处理
- [ ] query/candidate transform 是否避免使用局部 `lambda` / 闭包

### 8.4 工具层

- [ ] 是否已有 Python 转换器可复用
- [ ] 是否避免新增硬编码 shell 包装器
