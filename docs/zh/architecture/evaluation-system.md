# 评估系统设计

> **版本**: 0.3
> **更新日期**: 2026-03-26
> **架构设计版本**: 2.1
> **参考**: sentence-transformers 评估框架

---

## 1. 设计理念

### 1.1 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     AutoBenchmark / AutoEvaluator               │
│  - 从配置/注册表创建实例                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  MMEBBenchmark   │  │  MTEBBenchmark   │  │ (其他 Benchmark) │
│  (传入 data_path)│  │  (传入 data_path)│  │                  │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │ 创建                 │                     │
         ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RetrievalEvaluator                           │
│  - 接收 RetrievalEvalDataset                                    │
│  - 负责编码 + 指标计算                                           │
│  - 支持单卡/多卡分布式                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RetrievalEvalDataset                           │
│  - queries: EvalLanceDataset (+ parser transform)              │
│  - candidates: EvalLanceDataset (+ parser transform)           │
│  - qrels: LanceDataset                                          │
│  - metadata: dict                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LanceDataset                               │
│  - 位置: vlm2emb.data.datasets.base.LanceDataset                │
│  - 继承 torch.utils.data.Dataset                                │
│  - 使用 lance.dataset(path) 初始化                               │
│  - 提供精确 raw API 与惰性访问                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件职责

| 组件 | 职责 | 不负责 |
|------|------|--------|
| **Benchmark** | 加载数据集、创建 Evaluator、聚合结果 | 编码、指标计算 |
| **Evaluator** | 分布式编码、相似度计算、指标计算 | 数据加载 |
| **RetrievalEvalDataset** | 组合 query/candidate eval dataset 与 qrels raw dataset | 任何计算逻辑 |
| **LanceDataset** | 通用 Lance raw 包装，惰性加载 | decode、runtime transform、业务逻辑 |

### 1.3 设计原则

- **单一职责**: Evaluator 只做评测，不做数据加载
- **运行时注入**: model、processor_wrapper、accelerator 通过 `__call__` 传入，构造函数仅接收配置
- **透明分布式**: 用户代码不感知分布式细节
- **内存管理**: Benchmark 逐个加载数据集，评测后释放
- **惰性加载**: LanceDataset 在第一次访问时再打开 Lance 句柄
- **raw/sample 分离**: LanceDataset 只负责精确 raw 访问；sample 语义由更高层 dataset 处理
- **MMEB parser 下沉**: `MMEBBenchmark` 只负责 subset 选择、路径校验与聚合；subset-specific text transform 由 `datasets/mmeb_v2/` parser 决定

说明：
- `VideoEvalLanceDataset` 仍保留为框架兼容包装，但 `MMEB-V2 eval` 主路径已经改为 `EvalLanceDataset + 显式 parser transform`

---

## 2. 数据格式

### 2.1 Lance 存储格式

```
dataset_path/
├── queries.lance       # 查询数据
├── candidates.lance    # 候选数据
├── qrels.lance         # 相关性判断
└── metadata.json       # 元信息
```

### 2.2 Schema 定义

**queries.lance / candidates.lance**:
```python
schema = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("images", pa.list_(pa.binary())),
])
```

**qrels.lance** (每行一个 query):
```python
schema = pa.schema([
    pa.field("query_id", pa.string(), nullable=False),
    pa.field("mode", pa.string(), nullable=False),              # "sparse" | "exhaustive"
    pa.field("candidate_ids", pa.list_(pa.string())),           # [candidate_id, ...]
    pa.field("candidate_scores", pa.list_(pa.float32())),       # [score, ...]
])
```

### 2.3 qrels 模式 (per-sample)

`mode` 字段在每个 qrel 样本中，而非全局属性：

| mode | 含义 | 场景 |
|------|------|------|
| `sparse` | 只存 score > 0 的候选 | Global eval，候选集是全集 |
| `exhaustive` | 存该 query 的所有候选 | Local eval，每 query 独立候选集 |


### 2.4 MMEB Lance 转换维护决策（2026-02-06）

以下结论来自对原始 MMEB `test-*.parquet` 数据的核查和本地写入验证，作为后续维护约束：

1. `qrels.lance` 的相关性数据使用两个平行列存储：`candidate_ids: list<string>` 和 `candidate_scores: list<float32>`。
2. `--image-root` 支持 `~`，转换脚本应统一做 `expanduser()`，避免默认路径导致图像全部读取失败。
4. Query 侧 token/图像一致性检查（36 个数据集，共 36,000 条）显示：
   - `text 有 image token 但无图像`：仅出现在 `WebQA`/`EDIS`（共 2,000 条）
   - `有图像但 text 无 image token`：0 条
5. 对于**无图像输入**的 query（如 T2I query），应移除 legacy image token（`<|image_1|>`/`<|image_pad|>`/`<image>`），保持语义一致性。
6. I2I/VG 候选文本在“无 caption（image-only）”时，不追加结尾换行；有 caption 时保留结尾换行（与 v1 parser 一致）。


### 2.5 Eval 转换与原始 Parser 不一致记录（2026-02-07）

说明：本节只记录“原始 parser 与转换脚本不一致的现象及当前理由（含必要推断）”；逐项决策过程在对话中进行，不在文档维护待办状态。

1. **尾换行策略（已按 parser 对齐）**
   - 当前状态：已按 the public MMEB parser implementation 下各 parser 的显式换行逻辑对齐（如 `image_*` parser 显式追加 `\n`，`vidore/visrag` parser 不追加结尾换行）。
   - 当前理由：仅在 parser 明确追加换行时保持换行；未发现 parser 存在把 `\n` 作为字面文本导致换行失效的情况。
2. **路径约定差异（大小写/层级）**
   - 不一致：video/visdoc 脚本路径已改为“显式路径/别名表”解析，不再做自动变体推断；若本地目录未在别名表声明，将直接报错。
   - 当前理由：该策略提升可控性与可追踪性，避免隐式猜测路径导致的静默偏差。
3. **视频帧采样策略差异（批准例外）**
   - 不一致：原始 video parser 多处使用 `process_video_frames(..., num_frames=...)` 做帧采样；当前转换脚本不执行 `num_frames` 采样，直接保留目录下全部帧图像。
   - 当前理由：这是已确认的业务例外，用于保留评测输入的完整帧信息。
4. **无媒体 query 的 legacy token 清理差异（批准例外）**
   - 不一致：原始 `image_t2i_eval.py` 会将 `<|image_1|>` 替换为模型 image token；当前转换脚本在确认 query 无图像输入时移除 legacy token（`<|image_1|>`/`<|image_pad|>`/`<image>`）。
   - 当前理由：该差异仅作用于“无媒体输入”场景，用于消除 token 与实际输入不一致问题；其余场景保持 parser 原逻辑。


---

## 3. 核心接口

### 3.1 LanceDataset

```python
from vlm2emb.data.datasets.base import LanceDataset

# 从 .lance 目录加载
ds = LanceDataset("/path/to/queries.lance")

# 惰性访问
print(len(ds))      # 5000
row = ds[0]         # 单行访问
rows = ds.__getitems__([0, 1, 2])  # 批量访问
for row in ds:      # 迭代访问 (使用 scanner)
    ...
```

**实现原理**:
- 继承 `torch.utils.data.Dataset`
- 使用 `lance.dataset(path)` 初始化
- `__getitem__`: 通过 `take([idx])` 单行访问
- `__getitems__`: 通过 `take(indices)` 批量访问
- `__iter__`: 通过 `scanner` 高效迭代

### 3.2 RetrievalEvalDataset

```python
from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset

# 通过 MMEB-V2 parser runtime 加载
dataset = build_mmeb_eval_dataset(
    dataset_name="ImageNet-1K",
    artifact_path="/path/to/MMEB-V2/image-tasks/ImageNet-1K",
    transform_kwargs={"runtime_mode": "canonical"},
)

# 访问数据
print(dataset.name)           # "ImageNet-1K"
print(len(dataset))           # 查询数量
print(dataset.num_candidates) # 候选数量

# Query / candidate 走 parser transform 后的 eval item，qrels 走 raw LanceDataset
dataset.queries[0]     # {id, query={text, media}, metadata?}
dataset.candidates[0]  # {id, candidate={text, media}, metadata?}
dataset.qrels[0]       # {relation_id, query_id, mode, candidate_ids, candidate_scores}
```

### 3.3 RetrievalEvaluator

```python
from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.evaluation import RetrievalEvaluator

# 创建评测器（构造函数仅接收配置）
dataset = build_mmeb_eval_dataset(
    dataset_name="MVBench",
    artifact_path="/path/to/MMEB-V2/video-tasks/MVBench",
    transform_kwargs={"runtime_mode": "canonical", "num_frames": 8},
)
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=["hit@1", "hit@5", "ndcg@10", "mrr@10", "map@10"],
    batch_size=32,
)

# 运行评测（运行时资源通过 __call__ 传入）
results = evaluator(model, processor_wrapper, accelerator)
# {"hit@1": 0.85, "hit@5": 0.92, "ndcg@10": 0.72, ...}
```

检索评测的 `metrics` 使用完整指标名，格式为 `<metric>@<k>`。旧写法
`metrics=["hit", "ndcg"]` + `k_values=(1, 5)` 需要展开为
`metrics=["hit@1", "hit@5", "ndcg@1", "ndcg@5"]`。

### 3.4 Benchmark

```python
from vlm2emb.evaluation import MMEBBenchmark

# 创建 Benchmark（构造函数仅接收配置）
benchmark = MMEBBenchmark(
    data_path="/path/to/MMEB-V2",
    datasets=["ImageNet-1K", "MVBench"],  # 可选，不指定则全部
    batch_size=32,
    runtime_mode="canonical",
)

# 运行评测（运行时资源通过 __call__ 传入）
results = benchmark(model, processor_wrapper, accelerator)
# {
#     "ImageNet-1K": {"hit@1": 0.85, ...},
#     "MVBench": {"hit@1": 0.72, ...},
#     "_summary": {"hit@1": 0.78, ...},
# }
```

### 3.5 AutoBenchmark

```python
from vlm2emb.auto import AutoBenchmark

# 从配置创建
benchmark = AutoBenchmark.from_config({
    "type": "mmeb",
    "data_path": "/path/to/MMEB-V2",
    "datasets": ["ImageNet-1K"],
})

# 列出可用 Benchmark
print(AutoBenchmark.list_modules())  # ["mmeb", ...]
```

### 3.6 evaluate() 入口函数

```python
from vlm2emb import evaluate

# 从配置运行评估（统一入口）
results = evaluate(
    config,
    checkpoint="/path/to/checkpoint",
    output_dir="/path/to/output",
    config_path="configs/presets/xxx.yaml",
)
```

`evaluate(...)` 的当前职责是：

- 从 runtime `checkpoint` 或 `config["model"]` 解析模型来源
- 加载 model / processor
- 运行 benchmark
- 在主进程写入 `results.json` 与解析后的 `config.yaml`
- 执行收尾同步，并结束评测 runtime

模型来源支持两种模式：

| 配置方式 | 说明 |
|----------|------|
| `checkpoint` 运行时参数 | 优先使用 checkpoint；PEFT 场景下 processor 优先从 `base_model` 初始化 |
| `config["model"]` | 无 checkpoint 时，从 YAML 配置创建模型 |

---

## 4. 分布式支持

### 4.1 单卡/多卡兼容

```python
# 单卡运行
python scripts/eval.py configs/presets/xxx.yaml --checkpoint /path/to/checkpoint

# 多卡运行
accelerate launch scripts/eval.py configs/presets/xxx.yaml --checkpoint /path/to/checkpoint
```

用户代码无需修改，评测入口内部通过 `Accelerator` 管理运行时：

| 运行方式 | accelerator | 行为 |
|----------|-------------|------|
| `python scripts/eval.py` | 单进程 `Accelerator` | 单卡，直接编码所有数据 |
| `accelerate launch ...` | 多进程 `Accelerator` | 多卡，数据分片 + gather |

### 4.2 分布式编码流程

```
1. 数据分片: 以 effective eval `batch_size` 作为 block size，把全局样本切成 block-cyclic blocks 并循环分配给各 rank
2. 各卡仅编码自身 `shard_blocks` 覆盖的离散 block
3. gather 同时收集 embeddings 与对应的全局 sample indices
4. 主进程按 sample indices 回排，恢复 query / candidate embeddings 的原始全局顺序
5. 主进程基于与原始 qrels / ids 严格对齐的张量计算指标
6. broadcast 结果到所有进程
```

在 block-cyclic 模式下，各 rank 拥有离散的 `shard_blocks`，不再等价于单个连续 `shard=[a,b)` 区间。详细的 rank、shard、gather、trim、reorder 状态只保留在 DEBUG 级别的 `retrieval_protocol` 日志中；默认 INFO 日志不再输出逐 rank 编码窗口。

---

## 5. 使用场景

### 5.1 场景 1: 单数据集评测

```python
from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.evaluation import RetrievalEvaluator

dataset = build_mmeb_eval_dataset(
    dataset_name="ImageNet-1K",
    artifact_path="/path/to/MMEB-V2/image-tasks/ImageNet-1K",
)
evaluator = RetrievalEvaluator(dataset=dataset, batch_size=32)
results = evaluator(model, processor_wrapper, accelerator)
```

### 5.2 场景 2: Benchmark 评测

```python
from vlm2emb.evaluation import MMEBBenchmark

benchmark = MMEBBenchmark(data_path="/path/to/MMEB-V2")
results = benchmark(model, processor_wrapper, accelerator)
```

### 5.3 场景 3: Trainer 集成

```python
from vlm2emb.training import VLM2VecTrainer
from vlm2emb.evaluation import MMEBBenchmark

trainer = VLM2VecTrainer(
    model=model,
    args=args,
    eval_benchmark=MMEBBenchmark(data_path="..."),
)
trainer.train()
```

### 5.4 场景 4: CLI 运行

```bash
# 单卡
python scripts/eval.py configs/presets/vlm2vec_qwen2vl_2b.yaml \
    --checkpoint /path/to/checkpoint \
    eval.data_path=/path/to/MMEB-V2

# 多卡
accelerate launch scripts/eval.py configs/presets/vlm2vec_qwen2vl_2b.yaml \
    --checkpoint /path/to/checkpoint \
    eval.data_path=/path/to/MMEB-V2
```

#### MomentSeeker 单子集 baseline

当需要为 `MomentSeeker` 建立干净的 baseline 日志时，推荐直接使用 `scripts/eval.py` 的真实 CLI 入口，并同时打开 `--log-file`、`eval.datasets=[MomentSeeker]`、`eval.show_progress=false` 与 `--verbose`：

```bash
python scripts/eval.py configs/presets/vlm2vec_qwen2vl_2b.yaml \
    --checkpoint /path/to/checkpoint \
    --log-file output/momentseeker.log \
    --verbose \
    eval.data_path=/path/to/MMEB-V2 \
    eval.datasets=[MomentSeeker] \
    eval.show_progress=false \
    eval.num_workers=0
```

说明：
- `--log-file` 会把 baseline 日志落到文件，便于后续查看结果与 DEBUG 级别协议日志，不受终端滚动影响。
- `eval.datasets=[MomentSeeker]` 只运行单子集，避免其他 MMEB 子集干扰慢窗口分析。
- `eval.show_progress=false` 会关闭 progress bar，日志更容易阅读。
- `--verbose` 会额外保留 `retrieval_protocol` 的阶段日志，便于结合 `rank`、`gather_counts`、`trimmed_shape` 与 `shard_blocks` 做排障。
- `MomentSeeker` 的 query 侧混合了 text / image / video 条件输入，分析耗时时需要把 query 编码和 candidate 编码分开看，否则容易把 query 模态波动误判成候选编码退化。
- gather 后的 query / candidate embeddings 必须按 sample indices 回到原始全局顺序；任何 qrels / ids 对齐偏移都应视为 correctness bug，而不是可接受的日志噪声。

---

## 6. 评估指标

### 6.1 支持的指标

| 指标 | 说明 |
|------|------|
| Hit@K | 命中率，top-K 中是否有相关文档 |
| MRR | 平均倒数排名 |
| MAP | 平均精确率 |
| NDCG@K | 归一化折损累计增益 |

### 6.2 NDCG 两种形式

```python
# Linear form (默认)
ndcg_at_k(scores, relevance, k, form="linear")

# Exponential form
ndcg_at_k(scores, relevance, k, form="exponential")
```

---

## 7. 文件结构

```
src/vlm2emb/
├── data/
│   └── utils/
│       └── base.py              # LanceDataset / EvalLanceDataset 等基础组件
└── evaluation/
    ├── __init__.py              # 公开 API
    ├── evaluate.py              # evaluate(config, checkpoint=...) 入口函数
    ├── benchmarks/
    │   ├── __init__.py
    │   ├── base.py              # BaseBenchmark
    │   └── mmeb.py              # MMEBBenchmark
    ├── evaluators/
    │   ├── __init__.py
    │   ├── base.py              # BaseEvaluator
    │   └── retrieval.py         # RetrievalEvaluator
    ├── datasets/
    │   ├── __init__.py
    │   └── retrieval.py         # RetrievalEvalDataset
    └── metrics.py               # 指标计算
```

---

## 8. 相关文档

- [数据集 Layout 设计](../datasets/design-eval-lance-layout.md) - Lance 存储格式
- [注册表系统](./registry-system.md) - AutoBenchmark 使用
- [训练系统](./training-system.md) - Trainer 集成
