# Video QA 240K

> ShareGPTVideo 问答训练子集（配置键 `video_qa_240k`），在 BToks 中复用 LLaVA-Hound 双表结构。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_question_answer_train` |
| 用途 | 训练 |
| 溯源层数 | 2 层 |
| 训练权重 | 5 |
| 样本规模 | 240,000 |
| 采样帧数 | 8 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | ShareGPTVideo（LLaVA-Hound 训练来源） |
| 论文 | LLaVA-Hound: Visual Instruction Tuning for Video Understanding ([arXiv:2404.01258](https://arxiv.org/abs/2404.01258)) |
| 主要作者 | Yuanhan Zhang, Kaiyang Zhou, Bo Li, Wayne Xin Zhao, Ji-Rong Wen, Weipeng Chen, Yanyong Zhang, Weizhi Ma |
| HuggingFace | [ShareGPTVideo](https://huggingface.co/datasets/ShareGPTVideo) |
| 许可证 | 参见原始数据集 |

### Schema
原始数据在 BToks 中按双表组织（帧表 + 指令表），其中该子集由 `subset=train/sft/video_240k_caption_15k` 指定。

### 样本

**查询**: `<|video_pad|>...`
**目标**: `The setting of the video appears to be a newsroom with desks and computers in the background, sugges`

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `llavahound_train` |
| 实现文件 | `src/vlm2emb/data/datasets/llavahound.py` |
| 子集 | `train/sft/video_240k_caption_15k` |
| data_mode | `caption_retrieval` |
| 配置键 | `video_qa_240k` |
| 帧表路径 | `./data/converted` |
| 指令表路径 | `./data/converted` |

### Lance 布局

**帧表 `data/train_300k.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| video_id | string | 视频 ID |
| frame_idx | string | 帧文件名 / 顺序键 |
| image | binary | 帧图像二进制 |

**指令表 `data/video_instruction/train/sft/video_240k_caption_15k.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| id | string | 样本 ID |
| video | string | 关联 `video_id` |
| conversations | list[struct{from, value}] | 对话轮次 |

### 训练侧映射
- `caption_retrieval`: query 运行时为 `"<|video_pad|>\n{instruction}\n"` + 采样后的视频帧，positive 为 `"{caption}\n"`
- `video_retrieval`: query 运行时为 `"{transform.query.instruction} {caption}\n"`，positive 为 `"{transform.positive.instruction} <|video_pad|>\n"` + 采样后的视频帧
- `video_qa_240k` 当前配置运行时为 `"<|video_pad|>\nAnswer a question based on the content of a video.\n{question}\n"` + 采样后的视频帧，positive 为 `"{answer}\n"`

### 样本

以下示例直接采样自真实训练 Lance 表。

- 指令行保留原始 `conversations`。
- 帧表部分展示与当前 `num_frames=8` 采样策略对应的 8 行真实 Lance 记录。
- `image` 按 `bytes / sha1 / format / size` 展示摘要。

#### 示例 1

```json
{
  "data/video_instruction/train/sft/video_240k_caption_15k.lance": {
    "video": "v_XNTy5ZTMqVU-Scene-011",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is the setting of the video?"
      },
      {
        "from": "gpt",
        "value": "The setting of the video appears to be a newsroom with desks and computers in the background, suggesting an office environment with other personnel working."
      }
    ],
    "id": "v_XNTy5ZTMqVU-Scene-011_1"
  },
  "data/train_300k.lance (8 sampled frame rows for video_id=v_XNTy5ZTMqVU-Scene-011, matching current num_frames=8)": [
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0001.jpeg",
      "image": {
        "bytes": 9300,
        "sha1": "e5ff5b900128",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0004.jpeg",
      "image": {
        "bytes": 17579,
        "sha1": "568f7cd2c226",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0007.jpeg",
      "image": {
        "bytes": 17682,
        "sha1": "3c145f1c5079",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0011.jpeg",
      "image": {
        "bytes": 17542,
        "sha1": "fe6095859e17",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0014.jpeg",
      "image": {
        "bytes": 17904,
        "sha1": "c72f147c66cf",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0017.jpeg",
      "image": {
        "bytes": 17220,
        "sha1": "ec756977cb31",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0021.jpeg",
      "image": {
        "bytes": 17379,
        "sha1": "926c2808afb4",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    },
    {
      "video_id": "v_XNTy5ZTMqVU-Scene-011",
      "frame_idx": "c01_0024.jpeg",
      "image": {
        "bytes": 17507,
        "sha1": "12c308f5658d",
        "format": "JPEG",
        "size": [
          336,
          189
        ]
      }
    }
  ]
}
```

#### 示例 2

```json
{
  "data/video_instruction/train/sft/video_240k_caption_15k.lance": {
    "video": "SRhIRNhtdxQ",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nAre there any additional elements or informative text visible within the video frames?"
      },
      {
        "from": "gpt",
        "value": "No, there are no additional elements or informative text visible within the frames of the video."
      }
    ],
    "id": "SRhIRNhtdxQ_2"
  },
  "data/train_300k.lance (8 sampled frame rows for video_id=SRhIRNhtdxQ, matching current num_frames=8)": [
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0001.jpeg",
      "image": {
        "bytes": 10280,
        "sha1": "b76028cf3058",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0004.jpeg",
      "image": {
        "bytes": 13705,
        "sha1": "5fcb30c5d754",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0008.jpeg",
      "image": {
        "bytes": 20530,
        "sha1": "ed4a2fa4890d",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0012.jpeg",
      "image": {
        "bytes": 19298,
        "sha1": "bd5390cc2168",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0016.jpeg",
      "image": {
        "bytes": 16375,
        "sha1": "496fb7c62286",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0019.jpeg",
      "image": {
        "bytes": 18119,
        "sha1": "d9fe976f8187",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0023.jpeg",
      "image": {
        "bytes": 13948,
        "sha1": "d9b961508ea3",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    },
    {
      "video_id": "SRhIRNhtdxQ",
      "frame_idx": "c01_0027.jpeg",
      "image": {
        "bytes": 13230,
        "sha1": "5db101428ed8",
        "format": "JPEG",
        "size": [
          336,
          597
        ]
      }
    }
  ]
}
```
