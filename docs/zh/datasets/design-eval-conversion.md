# 评测数据集 Lance 转换计划

> 目标：减少文件数量，将评测数据集统一转换为 Lance 格式，支持惰性加载

## 1. 现状分析

### 1.1 数据集类型总览（81个数据集）

| 类别 | 数量 | 当前格式 | 图像存储方式 |
|-----|------|---------|------------|
| MMEB 图像任务 | 36 | HuggingFace Dataset | 散乱图像文件 |
| 视频任务 | 18 | JSONL + 帧目录 | 每视频一个目录（8-64帧） |
| ViDoRe 文档检索 | 17 | BEIR 格式（3子集） | 散乱 PNG 文件 |
| VisRAG 文档检索 | 6 | Parquet + 图像目录 | 散乱图像文件 |
| 其他文档检索 | 4 | BEIR 格式 | 散乱图像文件 |

**总大小**：~300GB

### 1.2 当前问题

1. **文件数量过多**：单个数据集可能有数万个独立图像文件
2. **无法惰性加载**：必须预先读取所有图像
3. **内存压力大**：300GB 无法一次性加载
4. **格式不统一**：JSONL、BEIR、HuggingFace Dataset 混用

---

## 2. 为什么选择 Lance

### 2.1 Lance vs Parquet vs HuggingFace Dataset

| 特性 | Lance | Parquet | HF Dataset |
|-----|-------|---------|------------|
| 惰性加载 | ✅ 原生支持 | ⚠️ 需要 pyarrow | ⚠️ 有限支持 |
| 内存映射 | ✅ 零拷贝 | ⚠️ 部分支持 | ❌ |
| 向量搜索 | ✅ 原生 ANN | ❌ | ❌ |
| 随机访问 | ✅ O(1) | ⚠️ 需要索引 | ✅ |
| 增量更新 | ✅ | ❌ 需重写 | ⚠️ |
| 图像存储 | ✅ 二进制列 | ✅ | ✅ |
| 流式写入 | ✅ | ⚠️ | ⚠️ |

### 2.2 Lance 核心优势

```python
import lance

# 惰性加载 - 不读取数据到内存
ds = lance.dataset("/path/to/dataset.lance")

# 按需读取单行
row = ds.take([0])  # 只读取第一行

# 谓词下推 - 只读取需要的列
df = ds.to_table(columns=["id", "text"]).to_pandas()

# 流式迭代 - 内存友好
for batch in ds.to_batches(batch_size=32):
    process(batch)

# 向量搜索（可选，用于检索加速）
ds.create_index("embedding", index_type="IVF_PQ")
results = ds.search(query_vector).limit(10).to_pandas()
```

---

## 3. 目标格式设计

### 3.1 统一 Lance Schema

```python
# eval_dataset.lance - 单文件包含所有数据
schema = pa.schema([
    # 公共字段
    ("id", pa.string()),
    ("role", pa.string()),              # "query" | "candidate"
    ("text", pa.string()),
    ("image_bytes", pa.list_(pa.binary())),  # 图像二进制列表
    ("image_format", pa.string()),      # "webp" | "png" | "jpg"

    # Qrels 相关（仅 query 有值）
    ("relevant_ids", pa.list_(pa.string())),   # 相关候选ID列表
    ("relevant_scores", pa.list_(pa.float32())), # 相关性分数列表
])

# metadata 存储在 Lance dataset 元信息中
metadata = {
    "name": "ImageNet-1K",
    "type": "classification",
    "eval_type": "global",  # "global" | "local"
    "num_queries": 1000,
    "num_candidates": 1000,
}
```

### 3.2 替代方案：分表存储

如果单表过大，可以分为三个 Lance 文件：

```
datasets/eval/ImageNet-1K/
├── queries.lance        # 查询数据
├── candidates.lance     # 候选数据
└── metadata.json        # 元信息
```

**Qrels 处理**：嵌入 queries.lance 的 `relevant_ids` 和 `relevant_scores` 列。

### 3.3 目录结构

```
datasets/eval/
├── mmeb/
│   ├── ImageNet-1K.lance
│   ├── N24News.lance
│   └── ...
├── video/
│   ├── SmthSmthV2.lance
│   └── ...
├── vidore/
│   └── ...
└── visrag/
    └── ...
```

---

## 4. 流式转换策略

### 4.1 核心原则：流式处理，避免全量加载

```python
import lance
import pyarrow as pa
from PIL import Image
import io

class StreamingConverter:
    """流式转换器 - 逐条处理，避免内存溢出"""

    def __init__(self, output_path: str, schema: pa.Schema):
        self.output_path = output_path
        self.schema = schema
        self.writer = None
        self.batch_size = 100  # 每批写入数量
        self.buffer = []

    def add_record(self, record: dict):
        """添加单条记录到缓冲区"""
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        """将缓冲区写入 Lance"""
        if not self.buffer:
            return

        table = pa.Table.from_pylist(self.buffer, schema=self.schema)

        if self.writer is None:
            # 首次写入，创建数据集
            lance.write_dataset(table, self.output_path, mode="create")
        else:
            # 追加写入
            lance.write_dataset(table, self.output_path, mode="append")

        self.buffer = []

    def finalize(self, metadata: dict):
        """完成写入，添加元信息"""
        self._flush()
        # 更新数据集元信息
        ds = lance.dataset(self.output_path)
        # Lance 支持在 dataset 上存储自定义元数据
```

### 4.2 图像流式读取

```python
def read_image_bytes(image_path: str) -> bytes:
    """读取原始图像字节"""
    with open(image_path, "rb") as f:
        return f.read()

def process_images_streaming(image_paths: list[str]) -> list[bytes]:
    """流式处理图像列表（保留原始格式）"""
    result = []
    for path in image_paths:
        if path and os.path.exists(path):
            result.append(read_image_bytes(path))
        else:
            result.append(b"")  # 空图像占位
    return result
```

### 4.3 大数据集分片处理

对于特别大的数据集（如视频），使用分片策略：

```python
def convert_large_dataset(
    source_loader,
    output_path: str,
    shard_size: int = 10000,
):
    """分片处理大数据集"""
    shard_idx = 0
    converter = StreamingConverter(f"{output_path}_shard_{shard_idx}.lance", schema)
    count = 0

    for item in source_loader:  # 迭代器，不预加载
        record = transform_item(item)
        converter.add_record(record)
        count += 1

        if count >= shard_size:
            converter.finalize({})
            shard_idx += 1
            converter = StreamingConverter(f"{output_path}_shard_{shard_idx}.lance", schema)
            count = 0

    converter.finalize({})

    # 可选：合并分片
    merge_shards(output_path, shard_idx + 1)
```

---

## 5. 实现计划

### 5.1 文件结构

```
scripts/
├── converters/
│   ├── __init__.py
│   ├── base.py              # StreamingConverter 基类
│   ├── mmeb_converter.py    # MMEB 数据集转换
│   ├── video_converter.py   # 视频数据集转换
│   ├── vidore_converter.py  # ViDoRe 转换
│   └── visrag_converter.py  # VisRAG 转换
└── convert_eval_to_lance.py # 命令行入口
```

### 5.2 基类设计

```python
# scripts/converters/base.py

from abc import ABC, abstractmethod
from pathlib import Path
import lance
import pyarrow as pa

# 统一 Schema
QUERY_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("images", pa.list_(pa.binary())),
])

CANDIDATE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("images", pa.list_(pa.binary())),
])

QRELS_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("mode", pa.string()),                                    # "sparse" | "exhaustive"
    ("candidate_ids", pa.list_(pa.string())),                 # [candidate_id, ...]
    ("candidate_scores", pa.list_(pa.float32())),             # [score, ...]
])

class BaseConverter(ABC):
    """评测数据集转换基类"""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.batch_size = 100

    @abstractmethod
    def iter_queries(self) -> Iterator[dict]:
        """迭代器：逐条返回 query 数据"""
        pass

    @abstractmethod
    def iter_candidates(self) -> Iterator[dict]:
        """迭代器：逐条返回 candidate 数据"""
        pass

    @abstractmethod
    def iter_qrels(self) -> Iterator[dict]:
        """迭代器：逐条返回 qrels 数据"""
        pass

    def convert(self) -> Path:
        """执行转换"""
        output_path = self.output_dir / self.config['name']
        output_path.mkdir(parents=True, exist_ok=True)

        # 写入 queries.lance
        self._write_lance(
            (self._make_query_record(q) for q in self.iter_queries()),
            output_path / "queries.lance",
            QUERY_SCHEMA
        )

        # 写入 candidates.lance
        self._write_lance(
            (self._make_candidate_record(c) for c in self.iter_candidates()),
            output_path / "candidates.lance",
            CANDIDATE_SCHEMA
        )

        # 写入 qrels.lance
        self._write_lance(
            self.iter_qrels(),
            output_path / "qrels.lance",
            QRELS_SCHEMA
        )

        # 写入 metadata.json
        self._write_metadata(output_path / "metadata.json")

        return output_path
```

### 5.3 命令行工具

```python
# scripts/convert_eval_to_lance.py

"""
评测数据集 Lance 转换工具

Usage:
    # 转换单个数据集
    python scripts/convert_eval_to_lance.py \
        configs/datasets/eval.yaml \
        --dataset ImageNet-1K \
        --output_dir /data/eval/lance/

    # 批量转换某类数据集
    python scripts/convert_eval_to_lance.py \
        configs/datasets/eval.yaml \
        --category mmeb \
        --output_dir /data/eval/lance/

    # 全量转换（使用多进程）
    python scripts/convert_eval_to_lance.py \
        configs/datasets/eval.yaml \
        --output_dir /data/eval/lance/ \
        --num_workers 4
"""

import argparse
from pathlib import Path
import yaml
from converters import get_converter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Dataset configuration file")
    parser.add_argument("--dataset", help="Single dataset name")
    parser.add_argument("--category", help="Dataset category: mmeb|video|vidore|visrag")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--image_format", default="webp")
    parser.add_argument("--image_quality", type=int, default=85)
    args = parser.parse_args()

    # Load config and convert
    ...

if __name__ == "__main__":
    main()
```

---

## 6. 加载器适配

### 6.1 新增 Lance 加载器

```python
# src/vlm2emb/data/datasets/lance_eval.py

import lance
import json
from pathlib import Path
from vlm2emb.auto import AutoDataset

@AutoDataset.register("lance_eval")
def load_lance_eval(
    path: str,
    dataset_name: str = "",
    lazy: bool = True,
    **kwargs,
) -> dict:
    """
    Load evaluation dataset from Lance format.

    Args:
        path: Path to dataset directory (contains queries.lance, candidates.lance, qrels.lance)
        dataset_name: Dataset name for metadata
        lazy: If True, return Lance datasets directly for lazy iteration
    """
    dataset_path = Path(path)

    # Load metadata
    with open(dataset_path / "metadata.json") as f:
        metadata = json.load(f)

    # Open Lance datasets (lazy - no data loaded yet)
    queries_ds = lance.dataset(str(dataset_path / "queries.lance"))
    candidates_ds = lance.dataset(str(dataset_path / "candidates.lance"))
    qrels_ds = lance.dataset(str(dataset_path / "qrels.lance"))

    if lazy:
        return {
            "queries": LanceIterator(queries_ds, transform_query),
            "candidates": LanceIterator(candidates_ds, transform_candidate),
            "qrels": qrels_ds,
            "metadata": metadata,
        }
    else:
        # Full load for small datasets
        return {
            "queries": [transform_query(r) for r in queries_ds.to_table().to_pylist()],
            "candidates": [transform_candidate(r) for r in candidates_ds.to_table().to_pylist()],
            "qrels": build_qrels_dict(qrels_ds),
            "metadata": metadata,
        }


class LanceIterator:
    """惰性迭代器包装"""

    def __init__(self, ds: lance.LanceDataset, transform_fn, batch_size: int = 32):
        self.ds = ds
        self.transform_fn = transform_fn
        self.batch_size = batch_size

    def __iter__(self):
        for batch in self.ds.to_batches(batch_size=self.batch_size):
            for row in batch.to_pylist():
                yield self.transform_fn(row)

    def __len__(self):
        return self.ds.count_rows()


def transform_query(row: dict) -> dict:
    """转换 query 行为标准格式"""
    return {
        "id": row["id"],
        "text": row["text"],
        "images": {"paths": [], "bytes": row.get("images", [])},
    }


def transform_candidate(row: dict) -> dict:
    """转换 candidate 行为标准格式"""
    return {
        "id": row["id"],
        "text": row["text"],
        "images": {"paths": [], "bytes": row.get("images", [])},
    }


def build_qrels_dict(qrels_ds: lance.LanceDataset) -> dict[str, dict[str, float]]:
    """构建 qrels 字典"""
    qrels = {}
    for batch in qrels_ds.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            qid = row["query_id"]
            qrels[qid] = dict(row["candidates"])
    return qrels
```

### 6.2 更新配置文件

```yaml
# configs/datasets/eval_lance.yaml

# Lance 格式数据集配置
_lance_defaults: &lance_defaults
  type: lance_eval
  lazy: true

ImageNet-1K:
  <<: *lance_defaults
  path: /data/eval/lance/mmeb/ImageNet-1K.lance

N24News:
  <<: *lance_defaults
  path: /data/eval/lance/mmeb/N24News.lance

# ... 其他数据集
```

---

## 7. 预估收益

### 7.1 存储空间

由于保留原始图像（无压缩），存储空间与原始数据相近：

| 数据集类别 | 当前大小 | 转换后大小（估算） |
|----------|---------|-----------------|
| MMEB 图像 | ~30GB | ~33GB |
| 视频帧 | ~200GB | ~200GB |
| ViDoRe | ~50GB | ~47GB |
| VisRAG | ~20GB | ~19GB |
| **总计** | **~300GB** | **~326GB** |

**说明**：转换后略有增加是因为 Lance 格式的元数据开销，但换来惰性加载能力。

### 7.2 加载性能

| 指标 | 当前 | Lance 格式 |
|-----|-----|-----------|
| 文件数量 | 数十万 | ~324（81 × 4） |
| 首次加载 | 分钟级 | 毫秒级（仅读元数据） |
| 随机访问 | 需要读取整个目录 | O(1) |
| 内存占用 | 全量加载 | 按需加载 |

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|-----|---------|
| Lance 版本不兼容 | 数据可能无法读取 | 记录使用的 lance 版本，保留原始数据 |
| 转换中断 | 数据不完整 | 实现检查点，支持断点续传 |
| 单文件过大 | 打开缓慢 | Lance 自动分片，按需合并 |

---

## 9. 实施步骤

### Phase 1: 基础设施（1-2天）
- [ ] 实现 `scripts/converters/base.py` 流式转换基类
- [ ] 实现 `scripts/convert_eval_to_lance.py` 命令行工具
- [ ] 添加 lance 依赖到 pyproject.toml

### Phase 2: 转换器实现（2-3天）
- [ ] `mmeb_converter.py` - MMEB 36 个数据集
- [ ] `video_converter.py` - 视频 18 个数据集
- [ ] `vidore_converter.py` - ViDoRe 17 个数据集
- [ ] `visrag_converter.py` - VisRAG 6 个数据集

### Phase 3: 加载器适配（1天）
- [ ] 实现 `src/vlm2emb/data/datasets/lance_eval.py`
- [ ] 创建 `configs/datasets/eval_lance.yaml`

### Phase 4: 测试与验证（1-2天）
- [ ] 小数据集转换测试（N24News）
- [ ] 大数据集转换测试（视频）
- [ ] 评测结果对比验证

### Phase 5: 全量转换（按需）
- [ ] 批量转换所有数据集
- [ ] 更新文档

---

## 10. 下一步

1. [ ] 确认计划
2. [ ] 先实现一个最小可用版本（N24News 转换 + 加载）
3. [ ] 验证评测结果一致性
4. [ ] 推广到其他数据集
