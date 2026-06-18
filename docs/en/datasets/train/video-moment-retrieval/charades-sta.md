# Charades-STA Training Dataset

> Video moment retrieval dataset used to train “given a full video and a language description, retrieve the matching clip”.

| Property | Value |
|------|-----|
| Modality | video |
| Task Type | `moment_retrieval` |
| Usage | Training / held-out testing |
| Provenance Layers | 2 layers |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | Charades-STA |
| Paper | Temporal Activity Localization via Language Query |
| License | See the original Charades / Charades-STA terms |

### Scale

- `official_train`: `12,408`
- `official_test`: `3,720`
- `official_train_without_mmeb_v2_eval`: `12,408`

### Schema

| Field | Type | Description |
|------|------|------|
| `id` | string | Charades video ID |
| `start_time` | float | Positive clip start time in seconds |
| `end_time` | float | Positive clip end time in seconds |
| `duration` | float | Positive clip duration in seconds |
| `description` | string | Language description |

### Sample

**query.text**: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: a person is putting a book on a shelf.\n`
**query.images**: sampled full-video frames
**positive.text**: `<|video_pad|>\nUnderstand the content of the provided video.\n`
**positive.images**: sampled positive clip frames
**negative.text**: empty

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `charades_sta_train` |
| Implementation | `src/vlm2emb/data/datasets/charades_sta_train.py` |
| Registry Name | `charades_sta_train` |
| Conversion Script | `scripts/convert/train/convert_charades_sta_to_lance.py` |
| Root Directory | `./data/converted` |

### Schema

| Column | Type | Description |
|------|------|------|
| `data/{split}.lance.id` | string | Charades video ID, also the video-table join key |
| `data/{split}.lance.query_frame_unit_key` | string | Query-side full-video frame-unit key |
| `data/{split}.lance.positive_frame_unit_key` | string | Positive-side target-segment frame-unit key |
| `data/{split}.lance.start_time` | float | Positive clip start time in seconds |
| `data/{split}.lance.end_time` | float | Positive clip end time in seconds |
| `data/{split}.lance.duration` | float | Positive clip duration in seconds |
| `data/{split}.lance.description` | string | Language description |
| `data/frames.lance.frame_unit_key` | string | Frame-unit join key |
| `data/frames.lance.frame_unit_type` | string | `full_video` or `segment` |
| `data/frames.lance.frames` | list[binary] | Pre-sampled RGB JPEG frames |
| `data/frames.lance.sampled_frame_count` | int | Number of frames kept during conversion |
| `data/frames.lance.source_total_num_frames` | int | Source-video frame count |
| `data/frames.lance.clip_total_num_frames` | int | Frame count covered by this frame unit |
| `data/frames.lance.clip_start` / `clip_end` | float | Segment boundaries in seconds; `0` for full videos |
| `data/frames.lance.sampled_indices` | list[int] | Sampled frame indices relative to the source video or segment |

### Training-side Mapping

- query: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: {description}\n`
- positive: `<|video_pad|>\nUnderstand the content of the provided video.\n`
- negative: empty text (placeholder)
- `num_frames` controls the second-stage sampling from pre-extracted frame units; the default runtime path does not raw-decode source videos.
- The query side uses a `full_video` frame unit capped at 64 frames during conversion; the positive side uses a `segment` frame unit capped at 8 frames.
- For positive media, `clip_total_num_frames` and `sampled_indices` refer to the segment-local frame count and segment-local indices; source-video context is kept in `source_total_num_frames`, `clip_start`, and `clip_end`.

### Split Strategy

- `official_train`: original training view
- `official_test`: held-out test view, not mixed into training by default
- `official_train_without_mmeb_v2_eval`: default training view; it currently has the same row count as `official_train`, and the name keeps the eval-exclusion boundary explicit

### Sample

The current runtime default shape is:

```json
{
  "query": {
    "text": "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: a person is putting a book on a shelf.\n",
    "images": ["<PIL.Image>", "<PIL.Image>", "..."],
    "media_metadata": {
      "fps": 24.0,
      "total_num_frames": 100,
      "sampled_indices": [0, 14, 28, 42, 56, 70, 84, 99]
    }
  },
  "positive": {
    "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
    "images": ["<PIL.Image>", "<PIL.Image>", "..."],
    "media_metadata": {
      "fps": 24.0,
      "total_num_frames": 29,
      "sampled_indices": [0, 4, 8, 12, 16, 20, 24, 28],
      "source_total_num_frames": 100,
      "source_sampled_indices": [10, 14, 18, 22, 26, 30, 34, 38],
      "segment_start_frame": 10,
      "segment_end_frame_exclusive": 39,
      "clip_start": 0.4,
      "clip_end": 1.6
    }
  },
  "negative": {
    "text": "",
    "images": []
  }
}
```
