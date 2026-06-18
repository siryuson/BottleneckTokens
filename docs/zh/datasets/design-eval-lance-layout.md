# 评测数据集 Lance Layout 设计

## 1. 目录结构

```
MMEB-V2/
├── image-tasks/                          # 图像任务 (36)
│   ├── ImageNet-1K/
│   │   ├── queries.lance
│   │   ├── qrels.lance
│   │   ├── candidates.lance
│   │   └── metadata.json
│   ├── ...
│   └── Visual7W-Pointing/
├── video-tasks/                          # 视频任务 (18)
│   ├── SmthSmthV2/
│   │   ├── queries.lance
│   │   ├── qrels.lance
│   │   ├── candidates.lance
│   │   └── metadata.json
│   ├── ...
│   └── ActivityNetQA/
└── visdoc-tasks/                         # 文档检索任务 (24)
    ├── ViDoRe_arxivqa/
    │   ├── queries.lance
    │   ├── qrels.lance
    │   ├── candidates.lance
    │   └── metadata.json
    ├── ...
    └── MMLongBench-page/
```

实际子集名称与数量以 `src/vlm2emb/data/conversion/mmeb_v2/datasets.py` 为准；本节只描述当前正式目录层级。

---

## 2. 单个数据集结构

每个数据集目录包含 4 个文件：

```
MMEB-V2/image-tasks/A-OKVQA/
├── queries.lance           # 查询数据
├── candidates.lance        # 候选数据
├── qrels.lance             # 相关性判断
└── metadata.json           # 元信息
```

### 2.1 queries.lance Schema

```python
QUERY_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("images", pa.list_(pa.binary())),       # 原始图像二进制
])
```

**示例数据**（仅示意 schema，实际 `text` 是否已渲染取决于具体评测数据集；`MMEB-V2 canonical` 会固化稳定文本，其它评测数据集可在 runtime 再渲染）：
```python
{
    "id": "q_0",
    "text": "<|image_pad|> Answer the question based on the image.\nQ: What is the dog doing?\n",
    "images": [b"...original image bytes..."],
}
```

### 2.2 candidates.lance Schema

```python
CANDIDATE_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("images", pa.list_(pa.binary())),       # 可选，文本候选为空
])
```

**示例数据**（文本候选）：
```python
{
    "id": "playing_fetch",
    "text": "playing fetch",
    "images": [],
}
```

**示例数据**（图像候选，如 t2i 任务）：
```python
{
    "id": "img_12345",
    "text": "<|image_pad|>",
    "images": [b"...original image bytes..."],
}
```

### 2.3 qrels.lance Schema

```python
QRELS_SCHEMA = pa.schema([
    pa.field("query_id", pa.string(), nullable=False),
    pa.field("mode", pa.string(), nullable=False),              # "sparse" | "exhaustive"
    pa.field("candidate_ids", pa.list_(pa.string())),           # [candidate_id, ...]
    pa.field("candidate_scores", pa.list_(pa.float32())),       # [score, ...]
])
```

**示例数据**：
```python
# Global eval (sparse) - 只存 score > 0
{"query_id": "q_0", "mode": "sparse", "candidate_ids": ["dog"], "candidate_scores": [1.0]}
{"query_id": "q_1", "mode": "sparse", "candidate_ids": ["cat", "tiger"], "candidate_scores": [1.0, 0.5]}

# Local eval (exhaustive) - 存储该 query 的所有候选
{"query_id": "q_0", "mode": "exhaustive", "candidate_ids": ["playing_fetch", "sleeping", "eating"], "candidate_scores": [1.0, 0.0, 0.0]}
{"query_id": "q_1", "mode": "exhaustive", "candidate_ids": ["red", "blue", "green"], "candidate_scores": [1.0, 0.0, 0.0]}
```

**说明**：
| mode | 含义 | 使用场景 |
|------|------|---------|
| `sparse` | 只包含 score > 0 的候选 | Global eval，候选集是全集 |
| `exhaustive` | 包含该 query 的所有候选 | Local eval，每 query 独立候选集 |

### 2.4 metadata.json

```json
{
  "name": "A-OKVQA",
  "modality": "image",
  "task_type": "image_question_answer",
  "num_queries": 5000,
  "num_candidates": 20000
}
```

**字段说明**：
| 字段 | 说明 |
|-----|------|
| `name` | 数据集名称 |
| `modality` | `image` / `video` / `visdoc` |
| `task_type` | 细粒度任务类型，例如 `image_question_answer`、`video_retrieval` |
| `num_queries` | 查询数量 |
| `num_candidates` | 候选数量 |

### 2.5 MMEB-V2 strict / canonical 转换约定

- 本节只适用于 `MMEB-V2 eval`。
- `strict` 以 the public reference parser contract 的 parser 输出为 gold，要求文本、图像/帧内容、顺序和 qrels 语义一致。
- `canonical` 只允许显式登记的修复，不改变任务语义。
- 允许的 canonical 修复包括：
  - legacy image / video token 分别归一到 `STANDARD_IMAGE_TOKEN` / `STANDARD_VIDEO_TOKEN`
  - 对独立占位式 token 统一为“token 首行 + 正文下一行”的布局；若 token 已明确嵌入句内语义，则保留原位置
  - whitespace 归一化（统一换行、去除多余外侧空白，并为非空文本保留单个结尾换行）
  - 仅在语义不变时使用本地路径 alias
- 不允许的 canonical 修复包括：
  - family prompt 重写
  - 候选顺序重排
  - qrels 关系改写

> 训练侧与其它普通评测数据集不使用这一套 strict/canonical 存储语义。

当前已经显式批准并进入 canonical 生成代码的 MMEB 数据级修复如下：
- `WebQA`、`EDIS`：移除 text-only query 中与真实输入不一致的 legacy image token
- `ViDoSeek-page`：切换到 `siyrus/BToks-ViDoSeek-page-fixed`
- `MMLongBench-page`：切换到 `siyrus/BToks-MMLongBench-page-fixed`
- `MomentSeeker`：固定到去重后的 `siyrus/BToks-MomentSeeker`
- `RefCOCO-Matching`：去重重复 qrels
- `MMLongBench-doc`：保持 graded qrels
- `ViDoRe_esg_reports_human_labeled_v2`：优先使用 `VLM2Vec` 镜像源
- `video_retrieval`：只做同语义 frame path 归一，不回退到 raw video

---

## 3. 数据访问模式

### 3.1 加载数据集

```python
import lance
import json
from pathlib import Path

def load_eval_dataset(dataset_path: str) -> dict:
    """加载评测数据集"""
    path = Path(dataset_path)

    # 加载元数据
    with open(path / "metadata.json") as f:
        metadata = json.load(f)

    # 惰性打开 Lance 数据集
    queries_ds = lance.dataset(str(path / "queries.lance"))
    candidates_ds = lance.dataset(str(path / "candidates.lance"))
    qrels_ds = lance.dataset(str(path / "qrels.lance"))

    return {
        "queries": queries_ds,
        "candidates": candidates_ds,
        "qrels": qrels_ds,
        "metadata": metadata,
    }
```

### 3.2 惰性迭代 queries

```python
def iter_queries(queries_ds: lance.LanceDataset, batch_size: int = 32):
    """惰性迭代查询"""
    for batch in queries_ds.to_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield {
                "id": row["id"],
                "text": row["text"],
                "images": {
                    "paths": [],
                    "bytes": row.get("images", []),
                },
            }
```

### 3.3 随机访问 candidates

```python
def get_candidates_by_ids(
    candidates_ds: lance.LanceDataset,
    ids: list[str]
) -> list[dict]:
    """按 ID 获取候选"""
    filter_expr = f"id IN ({','.join(repr(i) for i in ids)})"
    table = candidates_ds.to_table(filter=filter_expr)
    return table.to_pylist()
```

### 3.4 构建 qrels 字典

```python
def build_qrels_dict(qrels_ds: lance.LanceDataset) -> dict[str, dict[str, float]]:
    """从 qrels.lance 构建 qrels 字典"""
    qrels = {}
    for batch in qrels_ds.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            query_id = row["query_id"]
            qrels[query_id] = dict(zip(row["candidate_ids"], row["candidate_scores"]))
    return qrels
```

### 3.5 获取单个 query 的 qrels

```python
def get_qrels_for_query(
    qrels_ds: lance.LanceDataset,
    query_id: str
) -> tuple[str, dict[str, float]]:
    """获取单个 query 的相关性判断，返回 (mode, candidates)"""
    table = qrels_ds.to_table(filter=f"query_id = '{query_id}'")
    row = table.to_pylist()[0]
    return row["mode"], dict(zip(row["candidate_ids"], row["candidate_scores"]))
```

---

## 4. 配置文件

### 4.1 eval_lance.yaml

```yaml
# configs/datasets/eval_lance.yaml

_base_path: &base_path /data/MMEB-V2

_defaults: &defaults
  type: lance_eval
  lazy: true

# ============================================================================
# Image Tasks (36)
# ============================================================================

ImageNet-1K:
  <<: *defaults
  path: !join [*base_path, /image-tasks/ImageNet-1K]

A-OKVQA:
  <<: *defaults
  path: !join [*base_path, /image-tasks/A-OKVQA]

# ... 其他 image-tasks

# ============================================================================
# Video Tasks (18)
# ============================================================================

SmthSmthV2:
  <<: *defaults
  path: !join [*base_path, /video-tasks/SmthSmthV2]
  num_frames: 8

MSR-VTT:
  <<: *defaults
  path: !join [*base_path, /video-tasks/MSR-VTT]
  num_frames: 8

# ... 其他 video-tasks

# ============================================================================
# VisDoc Tasks (24)
# ============================================================================

VisRAG_ArxivQA:
  <<: *defaults
  path: !join [*base_path, /visdoc-tasks/VisRAG_ArxivQA]

ViDoSeek-page:
  <<: *defaults
  path: !join [*base_path, /visdoc-tasks/ViDoSeek-page]

# ... 其他 visdoc-tasks
```

---

## 5. 存储估算

当前正式产物会把 query / candidate 需要的图像与视频帧内嵌到 Lance 中，因此总空间会接近原始媒体体量。精确大小取决于：

- 每个视频任务保留的帧数
- VisDoc 页图体量
- Lance 分组与压缩效果
- 是否同时保留 strict 与 canonical 两套产物

因此不在本文档中维护固定容量数字，避免随着数据重生成和压缩参数变化而过时。

---

## 6. 验证清单

- [ ] 每个数据集目录包含 queries.lance, candidates.lance, qrels.lance, metadata.json
- [ ] queries/candidates/qrels 数量与 metadata.json 一致
- [ ] qrels 覆盖所有 query（global 或 local）
- [ ] 图像可正常解码
- [ ] strict 产物与 the public reference parser contract parser gold 对齐
- [ ] canonical 仅包含已登记修复项，且不引入未批准语义改动
