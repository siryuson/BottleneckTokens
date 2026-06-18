# Evaluation Dataset Lance Layout Design

## 1. Directory Structure

```
MMEB-V2/
├── image-tasks/                          # Image tasks (36)
│   ├── ImageNet-1K/
│   │   ├── queries.lance
│   │   ├── qrels.lance
│   │   ├── candidates.lance
│   │   └── metadata.json
│   ├── ...
│   └── Visual7W-Pointing/
├── video-tasks/                          # Video Tasks (18)
│   ├── SmthSmthV2/
│   │   ├── queries.lance
│   │   ├── qrels.lance
│   │   ├── candidates.lance
│   │   └── metadata.json
│   ├── ...
│   └── ActivityNetQA/
└── visdoc-tasks/                         # Document retrieval tasks (24)
    ├── ViDoRe_arxivqa/
    │   ├── queries.lance
    │   ├── qrels.lance
    │   ├── candidates.lance
    │   └── metadata.json
    ├── ...
    └── MMLongBench-page/
```

The exact dataset enumeration is maintained in `src/vlm2emb/data/conversion/mmeb_v2/datasets.py`; this section only documents the current directory and file layout.

---

## 2. Single Dataset Structure

Each dataset directory contains 4 files:

```
MMEB-V2/image-tasks/A-OKVQA/
├── queries.lance           # Query data
├── candidates.lance        # Candidate data
├── qrels.lance            # Relevance judgments
└── metadata.json           # Metadata
```

### 2.1 queries.lance Schema

```python
QUERY_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("images", pa.list_(pa.binary())),       # Original image binary
])
```

**Example Data** (schema illustration only; whether `text` is already formatted depends on the specific evaluation dataset. `MMEB-V2 canonical` freezes stable text, while other evaluation datasets may still format text at runtime):
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
    pa.field("images", pa.list_(pa.binary())),       # Optional, empty for text candidates
])
```

**Example Data** (text candidates):
```python
{
    "id": "playing_fetch",
    "text": "playing fetch",
    "images": [],
}
```

**Example Data** (image candidates, e.g., t2i task):
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

**Example Data**:
```python
# Global eval (sparse) - only store score > 0
{"query_id": "q_0", "mode": "sparse", "candidate_ids": ["dog"], "candidate_scores": [1.0]}
{"query_id": "q_1", "mode": "sparse", "candidate_ids": ["cat", "tiger"], "candidate_scores": [1.0, 0.5]}

# Local eval (exhaustive) - store all candidates for this query
{"query_id": "q_0", "mode": "exhaustive", "candidate_ids": ["playing_fetch", "sleeping", "eating"], "candidate_scores": [1.0, 0.0, 0.0]}
{"query_id": "q_1", "mode": "exhaustive", "candidate_ids": ["red", "blue", "green"], "candidate_scores": [1.0, 0.0, 0.0]}
```

**Explanation**:
| mode | meaning | use case |
|------|---------|----------|
| `sparse` | Only candidates with score > 0 | Global eval, candidate set is the full set |
| `exhaustive` | All candidates for this query | Local eval, per-query independent candidate set |

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

**Field Description**:
| field | description |
|-------|-------------|
| `name` | Dataset name |
| `modality` | `image` / `video` / `visdoc` |
| `task_type` | Fine-grained task type such as `image_question_answer` or `video_retrieval` |
| `num_queries` | Number of queries |
| `num_candidates` | Number of candidates |

### 2.5 MMEB-V2 strict / canonical Conversion Contract

- This section only applies to `MMEB-V2 eval`.
- `strict` uses the parser output from the public reference parser contract as gold. Text, media/frame content, ordering, and qrels semantics must match.
- `canonical` only allows explicitly registered repairs and must not change task semantics.
- Allowed canonical repairs:
  - Normalize legacy image / video tokens to `STANDARD_IMAGE_TOKEN` / `STANDARD_VIDEO_TOKEN`
  - Normalize standalone placeholder tokens into a "token first line + body on next line" layout; preserve the original position if the token is explicitly embedded in sentence semantics
  - Normalize whitespace (unify line endings, trim accidental outer whitespace, and keep a single trailing newline for non-empty text)
  - Use a local path alias only when it resolves to the same underlying asset
- Disallowed canonical repairs:
  - Family prompt rewrites
  - Candidate reordering
  - Qrels rewrites

> Training data and other ordinary evaluation datasets do not use this strict/canonical storage contract.

The following MMEB dataset-level repairs are explicitly approved and already implemented in canonical generation:
- `WebQA`, `EDIS`: remove legacy image tokens from text-only queries when they do not match actual inputs
- `ViDoSeek-page`: switch to `siyrus/BToks-ViDoSeek-page-fixed`
- `MMLongBench-page`: switch to `siyrus/BToks-MMLongBench-page-fixed`
- `MomentSeeker`: freeze to the deduplicated `siyrus/BToks-MomentSeeker`
- `RefCOCO-Matching`: deduplicate repeated qrels entries
- `MMLongBench-doc`: keep graded qrels
- `ViDoRe_esg_reports_human_labeled_v2`: prefer the `VLM2Vec` mirror source
- `video_retrieval`: only normalize frame paths with the same semantics and never fall back to raw video

---

## 3. Data Access Patterns

### 3.1 Load Evaluation Dataset

```python
import lance
import json
from pathlib import Path

def load_eval_dataset(dataset_path: str) -> dict:
    """Load evaluation dataset"""
    path = Path(dataset_path)

    # Load metadata
    with open(path / "metadata.json") as f:
        metadata = json.load(f)

    # Lazily open Lance datasets
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

### 3.2 Lazy Query Iteration

```python
def iter_queries(queries_ds: lance.LanceDataset, batch_size: int = 32):
    """Iterate queries lazily"""
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

### 3.3 Random Candidate Access

```python
def get_candidates_by_ids(
    candidates_ds: lance.LanceDataset,
    ids: list[str]
) -> list[dict]:
    """Get candidates by IDs"""
    filter_expr = f"id IN ({','.join(repr(i) for i in ids)})"
    table = candidates_ds.to_table(filter=filter_expr)
    return table.to_pylist()
```

### 3.4 Build Qrels Dictionary

```python
def build_qrels_dict(qrels_ds: lance.LanceDataset) -> dict[str, dict[str, float]]:
    """Build qrels dictionary from qrels.lance"""
    qrels = {}
    for batch in qrels_ds.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            query_id = row["query_id"]
            qrels[query_id] = dict(zip(row["candidate_ids"], row["candidate_scores"]))
    return qrels
```

### 3.5 Get Qrels for Single Query

```python
def get_qrels_for_query(
    qrels_ds: lance.LanceDataset,
    query_id: str
) -> tuple[str, dict[str, float]]:
    """Get relevance judgments for single query, return (mode, candidates)"""
    table = qrels_ds.to_table(filter=f"query_id = '{query_id}'")
    row = table.to_pylist()[0]
    return row["mode"], dict(zip(row["candidate_ids"], row["candidate_scores"]))
```

---

## 4. Configuration File

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
# ... Other image-tasks

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
# ... Other video-tasks

# ============================================================================
# VisDoc Tasks (24)
# ============================================================================

VisRAG_ArxivQA:
  <<: *defaults
  path: !join [*base_path, /visdoc-tasks/VisRAG_ArxivQA]

ViDoSeek-page:
  <<: *defaults
  path: !join [*base_path, /visdoc-tasks/ViDoSeek-page]
# ... Other visdoc-tasks
```

---

## 5. Storage Estimates

The official outputs embed query/candidate images and video frames directly into Lance, so storage remains close to the original media volume. Exact size depends on:

- how many frames each video task keeps
- the size of VisDoc page images
- Lance grouping and compression settings
- whether both `strict` and `canonical` outputs are retained

Because these numbers change with regeneration parameters, this document intentionally avoids maintaining fixed capacity totals.

---

## 6. Validation Checklist

- [ ] Each dataset directory contains queries.lance, candidates.lance, qrels.lance, metadata.json
- [ ] queries/candidates/qrels count matches metadata.json
- [ ] qrels cover all queries (global or local)
- [ ] Images decode correctly
- [ ] strict output matches the public reference parser contract parser gold
- [ ] canonical output only contains approved repairs and introduces no unapproved semantic changes
