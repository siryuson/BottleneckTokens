# Charades-STA 训练数据集

> 视频时刻检索数据集，用于训练“给定整段视频和文本描述，检索对应片段”的能力。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `moment_retrieval` |
| 用途 | 训练 / held-out 测试 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Charades-STA |
| 论文 | Temporal Activity Localization via Language Query |
| 许可证 | 参见原始 Charades / Charades-STA 说明 |

### 样本规模

- `official_train`: `12,408`
- `official_test`: `3,720`
- `official_train_without_mmeb_v2_eval`: `12,408`

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | Charades 视频 ID |
| `start_time` | float | 目标片段起始时间（秒） |
| `end_time` | float | 目标片段结束时间（秒） |
| `duration` | float | 目标片段时长（秒） |
| `description` | string | 语言描述 |

### 样本

**query.text**: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: a person is putting a book on a shelf.\n`
**query.images**: 采样后的整段视频帧列表
**positive.text**: `<|video_pad|>\nUnderstand the content of the provided video.\n`
**positive.images**: 采样后的目标片段帧列表
**negative.text**: 空

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `charades_sta_train` |
| 实现文件 | `src/vlm2emb/data/datasets/charades_sta_train.py` |
| 注册名 | `charades_sta_train` |
| 转换脚本 | `scripts/convert/train/convert_charades_sta_to_lance.py` |
| 根目录 | `./data/converted` |

### Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `data/{split}.lance.id` | string | Charades 视频 ID，也是视频表 join key |
| `data/{split}.lance.query_frame_unit_key` | string | query 侧完整视频 frame unit key |
| `data/{split}.lance.positive_frame_unit_key` | string | positive 侧目标片段 frame unit key |
| `data/{split}.lance.start_time` | float | 正样本片段起始时间（秒） |
| `data/{split}.lance.end_time` | float | 正样本片段结束时间（秒） |
| `data/{split}.lance.duration` | float | 正样本片段时长（秒） |
| `data/{split}.lance.description` | string | 语言描述 |
| `data/frames.lance.frame_unit_key` | string | frame unit join key |
| `data/frames.lance.frame_unit_type` | string | `full_video` 或 `segment` |
| `data/frames.lance.frames` | list[binary] | 预抽样 RGB JPEG 帧 |
| `data/frames.lance.sampled_frame_count` | int | 转换阶段保留的帧数 |
| `data/frames.lance.source_total_num_frames` | int | 原始视频总帧数 |
| `data/frames.lance.clip_total_num_frames` | int | 当前 frame unit 覆盖的帧数 |
| `data/frames.lance.clip_start` / `clip_end` | float | 片段秒级边界；完整视频为 `0` |
| `data/frames.lance.sampled_indices` | list[int] | 相对源视频或片段的采样帧索引 |

### 训练侧映射

- query: `<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: {description}\n`
- positive: `<|video_pad|>\nUnderstand the content of the provided video.\n`
- negative: 空文本（占位）
- `num_frames` 控制每个 video media slot 从预抽帧 frame unit 中二次采样返回的帧数；默认路径不在 runtime raw decode 原始视频。
- query 使用 `full_video` frame unit，转换阶段最多保留 64 帧；positive 使用 `segment` frame unit，转换阶段最多保留 8 帧。
- 对 positive media，`clip_total_num_frames` 与 `sampled_indices` 指片段内部帧数和片段内索引；源视频上下文保存在 `source_total_num_frames`、`clip_start` 与 `clip_end`。

### Split 策略

- `official_train`: 原始训练视图
- `official_test`: held-out 测试视图，不默认混入训练
- `official_train_without_mmeb_v2_eval`: 默认训练视图；当前行数与 `official_train` 相同，保留该命名用于明确 eval 排除边界

### 样本

以下为当前 runtime 默认形态：

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
