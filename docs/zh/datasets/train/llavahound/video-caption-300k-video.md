# Video Caption 300K (Video Retrieval)

> ShareGPTVideo 视频字幕检索训练子集（video_retrieval 模式），查询为文本描述，正样本为视频帧序列。
> 视频帧与 caption 属于存储事实；video retrieval prompt 与 `<|video_pad|>` 注入属于 runtime transform policy。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_retrieval_train` |
| 用途 | 训练 |
| 溯源层数 | 2 层 |
| 训练权重 | 5 |
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
与 `video-caption-300k` 使用同一原始双表，仅训练映射模式不同。

### 样本

**查询**: `Find a video that contains the following visual content: The video opens with a view of an indoor court marked with various lines...`
**目标**: `Understand the content of the provided video: <|video_pad|>`

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `llavahound_train` |
| 实现文件 | `src/vlm2emb/data/datasets/llavahound.py` |
| 子集 | `train/sft/video_caption_300k` |
| data_mode | `video_retrieval` |
| 配置键 | `video_caption_300k-video` |
| 帧表路径 | `./data/converted` |
| 指令表路径 | `./data/converted` |

### Lance 布局

**帧表 `data/train_300k.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| video_id | string | 视频 ID |
| frame_idx | string | 帧文件名 / 顺序键 |
| image | binary | 帧图像二进制 |

**指令表 `data/video_instruction/train/sft/video_caption_300k.lance`**:

| 列名 | 类型 | 说明 |
|------|------|------|
| id | string | 样本 ID |
| video | string | 关联 `video_id` |
| conversations | list[struct{from, value}] | 对话轮次 |

### 训练侧映射
- `caption_retrieval`: query 运行时为 `"<|video_pad|>\n{instruction}\n"` + 采样后的视频帧，positive 为 `"{caption}\n"`
- `video_retrieval`: query 运行时为 `"{transform.query.instruction} {caption}\n"`，positive 为 `"{transform.positive.instruction} <|video_pad|>\n"` + 采样后的视频帧
- `video_qa_240k` 当前配置仍走 `caption_retrieval` 模式，只是 human / gpt 对话是 QA 形式

### 样本

以下示例直接采样自真实训练 Lance 表。

- 指令行保留原始 `conversations`。
- 帧表部分展示与当前 `num_frames=8` 采样策略对应的 8 行真实 Lance 记录。
- `image` 按 `bytes / sha1 / format / size` 展示摘要。

#### 示例 1

```json
{
  "data/video_instruction/train/sft/video_caption_300k.lance": {
    "id": "v_--0edUL8zmA-Scene-001",
    "video": "v_--0edUL8zmA-Scene-001",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWrite an in-depth depiction of the video, covering all its aspects."
      },
      {
        "from": "gpt",
        "value": "The video opens with a view of an indoor court marked with various lines, presumably for different sports. There are four red balls aligned on the center line. In subsequent frames, people enter the frame, and a dodgeball game commences. The players are seen throwing balls at one another, maneuvering around the court to avoid being hit, and attempting to catch the balls. The participants vary in their attire, indicating a casual and informal game setting. The color contrasts of the red balls and the colored lines on the floor are visually prominent throughout the video. As the game progresses, players are occasionally hit by a ball and step out of the main play area. There appears to be good-natured interaction between the players, supportive of a friendly game rather than a competitive match."
      }
    ]
  },
  "data/train_300k.lance (8 sampled frame rows for video_id=v_--0edUL8zmA-Scene-001, matching current num_frames=8)": [
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0001.jpeg",
      "image": {
        "bytes": 9564,
        "sha1": "d56d10eb136e",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0022.jpeg",
      "image": {
        "bytes": 13399,
        "sha1": "f3cc53653870",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0043.jpeg",
      "image": {
        "bytes": 13446,
        "sha1": "c0f9c3cc76fc",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0064.jpeg",
      "image": {
        "bytes": 13711,
        "sha1": "c8bac63a61c8",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0085.jpeg",
      "image": {
        "bytes": 14352,
        "sha1": "f394fc447781",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0106.jpeg",
      "image": {
        "bytes": 13451,
        "sha1": "454bc59ede76",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0127.jpeg",
      "image": {
        "bytes": 13081,
        "sha1": "c9178392c0c3",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    },
    {
      "video_id": "v_--0edUL8zmA-Scene-001",
      "frame_idx": "c01_0148.jpeg",
      "image": {
        "bytes": 14267,
        "sha1": "0b039f3bbb9d",
        "format": "JPEG",
        "size": [
          336,
          252
        ]
      }
    }
  ]
}
```

#### 示例 2

```json
{
  "data/video_instruction/train/sft/video_caption_300k.lance": {
    "id": "OO7MpplyI2s",
    "video": "OO7MpplyI2s",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nElaborate on the visual and narrative elements of the video in detail."
      },
      {
        "from": "gpt",
        "value": "In the first few frames, a fluffy, light-brown cat with dark-tipped ears is shown tilting its head playfully while looking into the camera, suggestive of a quizzical or confused expression often found humorous by viewers. Subsequent frames depict another cat with a white and fluffy face and chest, bearing a small pink tag. A later scene transitions to a beige dog lying on a small bed, draped in a plaid blanket with a stuffed blue toy between its paws, its eyes peering out endearingly. The next sequence introduces a second dog, multicolored with black, tan, and white patches. The dog approaches the first dog, turning a red squeaky toy in its mouth which draws the attention of the first dog, leading to a playful interaction. Further along, a black cat wearing a green collar is shown rotating its head while the camera angle shifts, creating a humorous effect as the cat's gaze follows the moving lens."
      }
    ]
  },
  "data/train_300k.lance (8 sampled frame rows for video_id=OO7MpplyI2s, matching current num_frames=8)": [
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0001.jpeg",
      "image": {
        "bytes": 6972,
        "sha1": "4ed1ea943e58",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0004.jpeg",
      "image": {
        "bytes": 10757,
        "sha1": "53b430768769",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0007.jpeg",
      "image": {
        "bytes": 11026,
        "sha1": "348af55f7d41",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0010.jpeg",
      "image": {
        "bytes": 11665,
        "sha1": "fd8c0690726a",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0014.jpeg",
      "image": {
        "bytes": 12106,
        "sha1": "096147fc7c46",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0017.jpeg",
      "image": {
        "bytes": 12265,
        "sha1": "0ad83bccb188",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0020.jpeg",
      "image": {
        "bytes": 11422,
        "sha1": "e349654a74a3",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    },
    {
      "video_id": "OO7MpplyI2s",
      "frame_idx": "c01_0023.jpeg",
      "image": {
        "bytes": 10027,
        "sha1": "5267f434a721",
        "format": "JPEG",
        "size": [
          336,
          336
        ]
      }
    }
  ]
}
```
