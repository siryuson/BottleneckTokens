# Video Caption 300K

> ShareGPTVideo video caption retrieval training subset (caption_retrieval mode), organized with dual-table Lance structure for video frames and instructions.
> Video frames and `conversations` are stored facts; replacing `<video>` with `<|video_pad|>` is runtime transform policy.

| Property | Value |
|------|-----|
| Modality | video |
| Task Type | `video_retrieval_train` |
| Usage | Training |
| Provenance Layers | 2 layers |
| Training Weight | 5 |
| Sample Size | 300,000 |
| Sampled Frames | 8 |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | ShareGPTVideo (LLaVA-Hound training source) |
| Paper | LLaVA-Hound: Visual Instruction Tuning for Video Understanding ([arXiv:2404.01258](https://arxiv.org/abs/2404.01258)) |
| Primary Authors | Yuanhan Zhang, Kaiyang Zhou, Bo Li, Wayne Xin Zhao, Ji-Rong Wen, Weipeng Chen, Yanyong Zhang, Weizhi Ma |
| HuggingFace | [ShareGPTVideo](https://huggingface.co/datasets/ShareGPTVideo) |
| License | See original dataset |

### Schema
The original data is organized in dual-table format in BToks:

| Table | Key Fields | Description |
|----|----------|------|
| Frame Table | video_id, frame_idx, image | Stores video frames |
| Instruction Table | id, video, conversations | Stores video conversation instructions |

### Sample
> Full examples are provided in the BToks Lance section below.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `llavahound_train` |
| Implementation | `src/vlm2emb/data/datasets/llavahound.py` |
| Subset | `train/sft/video_caption_300k` |
| data_mode | `caption_retrieval` |
| Config Key | `video_caption_300k` |
| Frame Table Path | `./data/converted` |
| Instruction Table Path | `./data/converted` |

### Lance Layout

**Frame Table `data/train_300k.lance`**:

| Column | Type | Description |
|------|------|------|
| video_id | string | Video ID |
| frame_idx | string | Frame filename / ordering key |
| image | binary | Frame image bytes |

**Instruction Table `data/video_instruction/train/sft/video_caption_300k.lance`**:

| Column | Type | Description |
|------|------|------|
| id | string | Sample ID |
| video | string | Associated `video_id` |
| conversations | list[struct{from, value}] | Conversation turns |

### Training-side Mapping
- `caption_retrieval`: runtime query is `"<|video_pad|>\n{instruction}\n"` plus sampled video frames, and positive is `"{caption}\n"`
- `video_retrieval`: runtime query is `"{transform.query.instruction} {caption}\n"`, and positive is `"{transform.positive.instruction} <|video_pad|>\n"` plus sampled video frames
- `video_qa_240k` still uses `caption_retrieval` in the current config, but the human/gpt turns are QA-style

### Sample

The examples below are sampled directly from the real training Lance tables.

- Instruction rows keep the original `conversations`.
- The frame-table section shows the 8 real Lance rows selected by the current `num_frames=8` sampling policy.
- `image` is summarized as `bytes / sha1 / format / size`.

#### Example 1

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

#### Example 2

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
