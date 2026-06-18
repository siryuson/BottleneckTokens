# Evaluation Dataset Lance Conversion Plan

> Goal: Reduce file count by converting evaluation datasets to Lance format with lazy loading support

## 1. Current Status Analysis

### 1.1 Dataset Type Overview (81 datasets)

| Category | Count | Current Format | Image Storage |
|----------|--------|----------------|---------------|
| MMEB Image Tasks | 36 | HuggingFace Dataset | Scattered image files |
| Video Tasks | 18 | JSONL + Frame Directories | 8-64 frames per video |
| ViDoRe Document Retrieval | 17 | BEIR Format (3 subsets) | Scattered PNG files |
| VisRAG Document Retrieval | 6 | Parquet + Image Directories | Scattered image files |
| Other Document Retrieval | 4 | BEIR Format | Scattered image files |

**Total Size**: ~300GB

### 1.2 Current Issues

1. **Excessive file count**: Single datasets may have tens of thousands of independent image files
2. **No lazy loading support**: Must pre-read all images
3. **High memory pressure**: 300GB cannot be loaded at once
4. **Inconsistent formats**: Mix of JSONL, BEIR, HuggingFace Dataset

---

## 2. Why Lance

### 2.1 Lance vs Parquet vs HuggingFace Dataset

| Feature | Lance | Parquet | HF Dataset |
|---------|-------|---------|------------|
| Lazy Loading | Native | Requires pyarrow | Limited |
| Memory Mapping | Zero-copy | Partial | None |
| Vector Search | Native ANN | None | None |
| Random Access | O(1) | Requires index | Yes |
| Incremental Updates | Yes | Requires rewrite | Limited |
| Image Storage | Binary columns | Yes | Yes |
| Streaming Write | Yes | Limited | Limited |

### 2.2 Lance Core Advantages

```python
import lance

# Lazy loading - don't read data into memory
ds = lance.dataset("/path/to/dataset.lance")

# Read single row on demand
row = ds.take([0])  # Only read first row

# Predicate pushdown - only read needed columns
df = ds.to_table(columns=["id", "text"]).to_pandas()

# Streaming iteration - memory friendly
for batch in ds.to_batches(batch_size=32):
    process(batch)

# Vector search (optional, for retrieval acceleration)
ds.create_index("embedding", index_type="IVF_PQ")
results = ds.search(query_vector).limit(10).to_pandas()
```

---

## 3. Target Format Design

### 3.1 Unified Lance Schema

```python
# eval_dataset.lance - Single file containing all data
schema = pa.schema([
    # Common fields
    ("id", pa.string()),
    ("role", pa.string()),              # "query" | "candidate"
    ("text", pa.string()),
    ("image_bytes", pa.list_(pa.binary())),  # Image binary list
    ("image_format", pa.string()),      # "webp" | "png" | "jpg"

    # Qrels related (only query has values)
    ("relevant_ids", pa.list_(pa.string())),   # Relevant candidate ID list
    ("relevant_scores", pa.list_(pa.float32())), # Relevance score list
])

# Metadata stored in Lance dataset metadata
metadata = {
    "name": "ImageNet-1K",
    "type": "classification",
    "eval_type": "global",  # "global" | "local"
    "num_queries": 1000,
    "num_candidates": 1000,
}
```

### 3.2 Alternative: Separate Tables

If single table is too large, split into three Lance files:

```
datasets/eval/ImageNet-1K/
├── queries.lance        # Query data
├── candidates.lance     # Candidate data
└── metadata.json        # Metadata

# Qrels handling: embedded in queries.lance as relevant_ids and relevant_scores columns
```

### 3.3 Directory Structure

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

## 4. Streaming Conversion Strategy

### 4.1 Core Principle: Streaming Processing, Avoid Full Loading

```python
import lance
import pyarrow as pa
from PIL import Image
import io

class StreamingConverter:
    """Streaming converter - process item by item to avoid memory overflow"""

    def __init__(self, output_path: str, schema: pa.Schema):
        self.output_path = output_path
        self.schema = schema
        self.writer = None
        self.batch_size = 100  # Records per batch
        self.buffer = []

    def add_record(self, record: dict):
        """Add single record to buffer"""
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        """Write buffer to Lance"""
        if not self.buffer:
            return

        table = pa.Table.from_pylist(self.buffer, schema=self.schema)

        if self.writer is None:
            # First write, create dataset
            lance.write_dataset(table, self.output_path, mode="create")
        else:
            # Append write
            lance.write_dataset(table, self.output_path, mode="append")

        self.buffer = []

    def finalize(self, metadata: dict):
        """Complete write, add metadata"""
        self._flush()
        # Update dataset metadata
        ds = lance.dataset(self.output_path)
        # Lance supports storing custom metadata on dataset
```

### 4.2 Image Streaming Read

```python
def read_image_bytes(image_path: str) -> bytes:
    """Read raw image bytes"""
    with open(image_path, "rb") as f:
        return f.read()

def process_images_streaming(image_paths: list[str]) -> list[bytes]:
    """Process image list streaming (preserve original format)"""
    result = []
    for path in image_paths:
        if path and os.path.exists(path):
            result.append(read_image_bytes(path))
        else:
            result.append(b"")  # Empty image placeholder
    return result
```

### 4.3 Large Dataset Sharding

For especially large datasets (e.g., video), use sharding strategy:

```python
def convert_large_dataset(
    source_loader,
    output_path: str,
    shard_size: int = 10000,
):
    """Process large dataset with sharding"""
    shard_idx = 0
    converter = StreamingConverter(f"{output_path}_shard_{shard_idx}.lance", schema)
    count = 0

    for item in source_loader:  # Iterator, no preload
        record = transform_item(item)
        converter.add_record(record)
        count += 1

        if count >= shard_size:
            converter.finalize({})
            shard_idx += 1
            converter = StreamingConverter(f"{output_path}_shard_{shard_idx}.lance", schema)
            count = 0

    converter.finalize({})

    # Optional: merge shards
    merge_shards(output_path, shard_idx + 1)
```

---

## 5. Implementation Plan

### 5.1 File Structure

```
scripts/
├── converters/
│   ├── __init__.py
│   ├── base.py              # StreamingConverter base class
│   ├── mmeb_converter.py    # MMEB dataset conversion
│   ├── video_converter.py   # Video dataset conversion
│   ├── vidore_converter.py  # ViDoRe conversion
│   └── visrag_converter.py  # VisRAG conversion
└── convert_eval_to_lance.py  # CLI entry point
```

### 5.2 Base Class Design

```python
# scripts/converters/base.py

from abc import ABC, abstractmethod
from pathlib import Path
import lance
import pyarrow as pa

# Unified Schema
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
    """Evaluation dataset conversion base class"""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.batch_size = 100

    @abstractmethod
    def iter_queries(self) -> Iterator[dict]:
        """Iterator: return query data item by item"""
        pass

    @abstractmethod
    def iter_candidates(self) -> Iterator[dict]:
        """Iterator: return candidate data item by item"""
        pass

    @abstractmethod
    def iter_qrels(self) -> Iterator[dict]:
        """Iterator: return qrels data item by item"""
        pass

    def convert(self) -> Path:
        """Execute conversion"""
        output_path = self.output_dir / self.config['name']
        output_path.mkdir(parents=True, exist_ok=True)

        # Write queries.lance
        self._write_lance(
            (self._make_query_record(q) for q in self.iter_queries()),
            output_path / "queries.lance",
            QUERY_SCHEMA
        )

        # Write candidates.lance
        self._write_lance(
            (self._make_candidate_record(c) for c in self.iter_candidates()),
            output_path / "candidates.lance",
            CANDIDATE_SCHEMA
        )

        # Write qrels.lance
        self._write_lance(
            self.iter_qrels(),
            output_path / "qrels.lance",
            QRELS_SCHEMA
        )

        # Write metadata.json
        self._write_metadata(output_path / "metadata.json")

        return output_path
```

### 5.3 CLI Tool

```python
# scripts/convert_eval_to_lance.py

"""
Evaluation Dataset Lance Conversion Tool

Usage:
    # Convert single dataset
    python scripts/convert_eval_to_lance.py \
        configs/datasets/eval.yaml \
        --dataset ImageNet-1K \
        --output_dir /data/eval/lance/

    # Batch convert by category
    python scripts/convert_eval_to_lance.py \
        configs/datasets/eval.yaml \
        --category mmeb \
        --output_dir /data/eval/lance/

    # Full conversion (with multiprocessing)
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

## 6. Loader Adaptation

### 6.1 New Lance Loader

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
    """Lazy iterator wrapper"""

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
    """Transform query row to standard format"""
    return {
        "id": row["id"],
        "text": row["text"],
        "images": {"paths": [], "bytes": row.get("images", [])},
    }


def transform_candidate(row: dict) -> dict:
    """Transform candidate row to standard format"""
    return {
        "id": row["id"],
        "text": row["text"],
        "images": {"paths": [], "bytes": row.get("images", [])},
    }


def build_qrels_dict(qrels_ds: lance.LanceDataset) -> dict[str, dict[str, float]]:
    """Build qrels dictionary"""
    qrels = {}
    for batch in qrels_ds.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            qid = row["query_id"]
            qrels[qid] = dict(row["candidates"])
    return qrels
```

### 6.2 Update Config File

```yaml
# configs/datasets/eval_lance.yaml

# Lance format dataset configuration
_lance_defaults: &lance_defaults
  type: lance_eval
  lazy: true

ImageNet-1K:
  <<: *lance_defaults
  path: /data/eval/lance/mmeb/ImageNet-1K.lance

N24News:
  <<: *lance_defaults
  path: /data/eval/lance/mmeb/N24News.lance

# ... Other datasets
```

---

## 7. Estimated Benefits

### 7.1 Storage Space

Since original images are preserved (no compression), storage space is similar to original data:

| Dataset Category | Current Size | After Conversion (Estimated) |
|-----------------|--------------|------------------------------|
| MMEB Images | ~30GB | ~33GB |
| Video Frames | ~200GB | ~200GB |
| ViDoRe | ~50GB | ~47GB |
| VisRAG | ~20GB | ~19GB |
| **Total** | **~300GB** | **~326GB** |

**Note**: Slight increase after conversion is due to Lance format metadata overhead, but换来 (换来 = in exchange for) lazy loading capability.

### 7.2 Loading Performance

| Metric | Current | Lance Format |
|--------|---------|--------------|
| File Count | Hundreds of thousands | ~324 (81 × 4) |
| First Load | Minutes | Milliseconds (metadata only) |
| Random Access | Requires reading entire directory | O(1) |
| Memory Usage | Full load | On-demand |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Lance version incompatibility | Data may not be readable | Record lance version used, keep original data |
| Conversion interruption | Incomplete data | Implement checkpoint, support resume |
| Single file too large | Slow to open | Lance auto-sharding, merge on demand |

---

## 9. Implementation Steps

### Phase 1: Infrastructure (1-2 days)
- [ ] Implement `scripts/converters/base.py` streaming conversion base class
- [ ] Implement `scripts/convert_eval_to_lance.py` CLI tool
- [ ] Add lance dependency to pyproject.toml

### Phase 2: Converter Implementation (2-3 days)
- [ ] `mmeb_converter.py` - 36 MMEB datasets
- [ ] `video_converter.py` - 18 video datasets
- [ ] `vidore_converter.py` - 17 ViDoRe datasets
- [ ] `visrag_converter.py` - 6 VisRAG datasets

### Phase 3: Loader Adaptation (1 day)
- [ ] Implement `src/vlm2emb/data/datasets/lance_eval.py`
- [ ] Create `configs/datasets/eval_lance.yaml`

### Phase 4: Testing and Validation (1-2 days)
- [ ] Small dataset conversion test (N24News)
- [ ] Large dataset conversion test (video)
- [ ] Evaluation results consistency verification

### Phase 5: Full Conversion (on demand)
- [ ] Batch convert all datasets
- [ ] Update documentation

---

## 10. Next Steps

1. [ ] Confirm plan
2. [ ] Implement minimum viable version first (N24News conversion + loading)
3. [ ] Verify evaluation results consistency
4. [ ] Extend to other datasets
