# QVHighlights Training Dataset

> Query-driven video moment retrieval dataset used to train “given a full video and a language description, retrieve the matching clip”.

| Property | Value |
|------|-----|
| Modality | video |
| Task Type | `moment_retrieval` |
| Usage | Training / held-out validation |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | QVHighlights |
| Paper | QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries |
| License | See the original QVHighlights terms |

### Scale

- `official_train`: `7,218`
- `official_val`: `1,550`
- `official_test`: `1,542` (no positive windows; written as a held-out audit split, not used as a default training positive view)

### Local Source State

- annotation root: `./data/converted`
- current video root: `./data/converted`
- The current video source is the expanded directory from the community HF split-video source `yeliudev/VideoMind-Dataset`; `videos/` contains `12,562` mp4 files and covers all `10,148` unique annotated `vid` values, with `0` missing videos.
- The local official-named tar `./data/converted` is only about `219MB`; `gzip -t` and `tar -tzf` fail with EOF and only a small number of videos can be listed, so it is not used as a conversion source.
- A three-part `videos.tar.gz.*` archive from `yeliudev/VideoMind-Dataset` is also present locally. If archive-direct reading is added later, those parts should be integrity-checked and documented for provenance before considering the broken official-named tar.

### Schema

| Field | Type | Description |
|------|------|------|
| `qid` | string / int | Query ID |
| `query` | string | Language description |
| `duration` | float | Video duration-related metadata |
| `vid` | string | Video ID |
| `relevant_windows` | list[list[float]] | Positive temporal windows |

### Sample

**query.text**: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: some military patriots takes us through their safety procedures and measures.\n`
**query.images**: sampled full-video frames
**positive.text**: `<|video_pad|>\nUnderstand the content of the provided video.\n`
**positive.images**: sampled positive clip frames chosen from `relevant_windows`
**negative.text**: empty

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `qvhighlight_train` |
| Implementation | `src/vlm2emb/data/datasets/qvhighlight_train.py` |
| Registry Name | `qvhighlight_train` |
| Conversion Script | `scripts/convert/train/convert_qvhighlight_to_lance.py` |
| Root Directory | `./data/converted` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/{split}.lance.qid` | string | Query ID |
| `data/{split}.lance.vid` | string | Video ID |
| `data/{split}.lance.query_frame_unit_key` | string | Query-side full-video frame-unit key |
| `data/{split}.lance.positive_frame_unit_keys` | list[string] | Positive candidate segment frame-unit keys, aligned with `relevant_windows` |
| `data/{split}.lance.query` | string | Language description |
| `data/{split}.lance.relevant_windows` | list | All positive windows |
| `data/frames.lance.frame_unit_key` | string | Frame-unit join key |
| `data/frames.lance.frame_unit_type` | string | `full_video` or `segment` |
| `data/frames.lance.frames` | list[binary] | Pre-sampled RGB JPEG frames |
| `data/frames.lance.sampled_frame_count` | int | Number of frames kept during conversion |
| `data/frames.lance.source_total_num_frames` | int | Source-video frame count |
| `data/frames.lance.clip_total_num_frames` | int | Frame count covered by this frame unit |
| `data/frames.lance.clip_start` / `clip_end` | float | Segment boundaries in seconds; `0` for full videos |
| `data/frames.lance.sampled_indices` | list[int] | Sampled frame indices relative to the source video or segment |

### Training-side Mapping

- query: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: {query_text}\n`
- positive: `<|video_pad|>\nUnderstand the content of the provided video.\n`
- negative: empty text (placeholder)
- `num_frames` controls the second-stage sampling from pre-extracted frame units; the default runtime path does not raw-decode source videos.
- The query side uses a `full_video` frame unit capped at 64 frames during conversion; the positive side selects one `segment` frame unit from `positive_frame_unit_keys`, with each segment capped at 8 frames.
- For positive media, `clip_total_num_frames` and `sampled_indices` refer to the segment-local frame count and segment-local indices; source-video context is kept in `source_total_num_frames`, `clip_start`, and `clip_end`.

### Split Strategy

- `official_train_without_mmeb_v2_eval`: default training view
- `official_train`: original training view kept for audit
- `official_val`: held-out validation view
- `official_test`: not kept as a trainable split when the raw annotations do not contain positive windows; audit rows are retained in exclusion / audit tables when needed

### Local Audit Result

- `MMEB-V2 / QVHighlight` has `1081` unique queries that align almost entirely with raw `official_val`
- It has only minimal query overlap with raw `official_train`
- The current implementation therefore preserves `official_train / official_val` and keeps `official_val` out of training by default
- The current conversion writes `data/frames.lance`: one full-video frame unit per video plus one segment frame unit per positive window. Trainable splits keep only rows with positive windows. Raw `official_test` rows without positive windows are not retained as trainable rows; they are kept only in audit / exclusion records when needed.
