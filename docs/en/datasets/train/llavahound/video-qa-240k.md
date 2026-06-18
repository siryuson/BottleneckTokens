# Video QA 240K

> ShareGPTVideo question-answer training subset (config key `video_qa_240k`), reuses LLaVA-Hound dual-table structure in BToks.

| Property | Value |
|------|-----|
| Modality | video |
| Task Type | `video_question_answer_train` |
| Usage | Training |
| Provenance Layers | 2 layers |
| Training Weight | 5 |
| Sample Size | 240,000 |
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
The original data is organized in dual-table format in BToks (frame table + instruction table), with this subset specified by `subset=train/sft/video_240k_caption_15k`.

### Sample
> Full examples are provided in the BToks Lance section below.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `llavahound_train` |
| Implementation | `src/vlm2emb/data/datasets/llavahound.py` |
| Subset | `train/sft/video_240k_caption_15k` |
| data_mode | `caption_retrieval` |
| Config Key | `video_qa_240k` |
| Frame Table Path | `./data/converted` |
| Instruction Table Path | `./data/converted` |

### Lance Layout

**Frame Table `data/train_300k.lance`**:

| Column | Type | Description |
|------|------|------|
| video_id | string | Video ID |
| frame_idx | string | Frame filename / ordering key |
| image | binary | Frame image bytes |

**Instruction Table `data/video_instruction/train/sft/video_240k_caption_15k.lance`**:

| Column | Type | Description |
|------|------|------|
| id | string | Sample ID |
| video | string | Associated `video_id` |
| conversations | list[struct{from, value}] | Conversation turns |

### Training-side Mapping
- `caption_retrieval`: runtime query is `"<|video_pad|>\n{instruction}\n"` plus sampled video frames, and positive is `"{caption}\n"`
- `video_retrieval`: runtime query is `"{transform.query.instruction} {caption}\n"`, and positive is `"{transform.positive.instruction} <|video_pad|>\n"` plus sampled video frames
- `video_qa_240k`: in the current config, runtime query is `"<|video_pad|>\nAnswer a question based on the content of a video.\n{question}\n"` plus sampled video frames, and positive is `"{answer}\n"`

### Sample

The examples below are sampled directly from the real training Lance tables.

- Instruction rows keep the original `conversations`.
- The frame-table section shows the 8 real Lance rows selected by the current `num_frames=8` sampling policy.
- `image` is summarized as `bytes / sha1 / format / size`.

#### Example 1

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

#### Example 2

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
