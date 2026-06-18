# MomentSeeker

> 面向长视频时刻检索的任务导向基准，支持文本/图像/视频条件查询。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_moment_retrieval` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | MomentSeeker |
| 论文 | MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval（Yuan et al., 2025）([OpenReview](https://openreview.net/forum?id=gKkA9Oc13m)) |
| HuggingFace | [siyrus/BToks-MomentSeeker](https://huggingface.co/datasets/siyrus/BToks-MomentSeeker) |
| 许可证 | 参见原始数据集 |

### Schema
- `query`: 文本查询
- `input_frames`: 查询条件（可为视频片段、图像或空）
- `positive_frames`: 正样本片段信息
- `negative_frames`: 负样本片段信息

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-MomentSeeker](https://huggingface.co/datasets/siyrus/BToks-MomentSeeker) |
| 分割 | `test` |

### Schema
- `query`: 文本查询
- `input_frames`: 查询条件路径
- `positive_frames` / `negative_frames`: 候选片段集合

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/MomentSeeker/` |
| task_type | `video_moment_retrieval` |
| 评测模式 | local |

### Lance 布局
- `queries.lance`: 查询（文本/图像/视频条件混合，`id`, `text`, `images`）
- `candidates.lance`: 候选视频片段（`id`, `text`, `images`）
- `qrels.lance`: 穷举相关性标注（`mode=exhaustive`）

### 运行基线与运行时说明

- `queries.lance` 是混合 query 模态：同一个子集中同时存在纯文本 query、图像条件 query 与视频条件 query。
- `candidates.lance` 则始终表示候选视频片段，因此 query 侧和 candidate 侧的编码成本不应混读。
- 建议用 `scripts/eval.py` 运行单子集 baseline，并显式传入 `--log-file output/momentseeker.log eval.datasets=[MomentSeeker] eval.show_progress=false --verbose`。
- 在 block-cyclic 协议下，默认 INFO 日志不再输出逐 rank 编码窗口；需要细查时，通过 DEBUG 级别的 `retrieval_protocol` 日志查看 shard、gather、trim、reorder 状态。
- gather 后的 query / candidate embeddings 必须按 sample indices 回排回原始全局顺序，确保 qrels / ids 严格对齐；若排序错位，MomentSeeker 指标即失真，应直接视为协议错误。

### 样本

以下示例基于当前 runtime canonical 输出生成，数据来源为 `./data/converted`.

- `text` 展示当前 runtime canonical 的模型可见文本；parser 可能在 Lance 原始字段基础上重组 visual token 布局、选项块和尾部换行。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_0`，`negative_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nFind the clip that corresponds to the given sentence and video segment: What happened to this window afterward?\n",
    "images": {
      "count": 8,
      "items": [
        {
          "index": 0,
          "bytes": 11240,
          "sha1": "36688d6f4493",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 1,
          "bytes": 16099,
          "sha1": "fce8e7c795b2",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 2,
          "bytes": 17521,
          "sha1": "19b21ab5d7cc",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 3,
          "bytes": 20722,
          "sha1": "94e1c0b72986",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 4,
          "bytes": 24712,
          "sha1": "279774cd01a4",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 5,
          "bytes": 18308,
          "sha1": "17458a2e2ef3",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 6,
          "bytes": 13775,
          "sha1": "159155d0d7e2",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        },
        {
          "index": 7,
          "bytes": 11614,
          "sha1": "b3843aadcd19",
          "format": "JPEG",
          "size": [
            720,
            480
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 17764,
            "sha1": "da2743bc4e5e",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 23837,
            "sha1": "5d49baa1a50c",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 30725,
            "sha1": "0592e00cf29d",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 23733,
            "sha1": "8279cd30ecff",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 15708,
            "sha1": "cd98c0700b7f",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 16465,
            "sha1": "530e3921c9c8",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 29410,
            "sha1": "68834ffebd5b",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 25601,
            "sha1": "f0e016457a82",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_1",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 29798,
            "sha1": "31562ea14ce5",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 1,
            "bytes": 36062,
            "sha1": "d76cf2d98b3c",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 2,
            "bytes": 48848,
            "sha1": "9ef5de55919f",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 3,
            "bytes": 29880,
            "sha1": "d5a3f30a5b64",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 4,
            "bytes": 24289,
            "sha1": "0bb05bbb832a",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 5,
            "bytes": 17050,
            "sha1": "0626c5e1c8bb",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 6,
            "bytes": 15834,
            "sha1": "f153eabe4fcd",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          },
          {
            "index": 7,
            "bytes": 9626,
            "sha1": "0df4be726cb2",
            "format": "JPEG",
            "size": [
              720,
              480
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_0"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```

#### 示例 2
`query_id=q_533`，`positive_candidate_id=c_5425`，`negative_candidate_id=c_5426`

```json
{
  "queries.lance": {
    "id": "q_533",
    "text": "Find the clip that corresponds to the given text: When the man with the UNIQLO logo and the black woman in pink clothes took a selfie together, who held the phone?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_5425",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 63593,
            "sha1": "ba128193494c",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 77828,
            "sha1": "17be7b5918f4",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 92079,
            "sha1": "3bccb39af4e0",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 92242,
            "sha1": "a5800676f6d2",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 69607,
            "sha1": "3d846db5bddc",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 53883,
            "sha1": "94938ec722e1",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 47652,
            "sha1": "2f6df7993808",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 42285,
            "sha1": "8d76dcef28db",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_5426",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 72281,
            "sha1": "3e56d305a755",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 89074,
            "sha1": "91499c60429f",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 110251,
            "sha1": "d2992e6646ca",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 75369,
            "sha1": "e2fa02804def",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 65617,
            "sha1": "598f86adcfb0",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 47915,
            "sha1": "182b08068a5a",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 41091,
            "sha1": "f277bb994593",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 36082,
            "sha1": "b71faa8c572f",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_533",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_5425"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```

#### 示例 3
`query_id=q_1200`，`positive_candidate_id=c_12139`，`negative_candidate_id=c_12140`

```json
{
  "queries.lance": {
    "id": "q_1200",
    "text": "<|image_pad|>\nSelect the video clip that aligns with the given text and image: What am I holding in my left hand in this picture?\n",
    "images": {
      "count": 1,
      "items": [
        {
          "index": 0,
          "bytes": 1252417,
          "sha1": "dc942030b194",
          "format": "JPEG",
          "size": [
            2560,
            1920
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_12139",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 316835,
            "sha1": "d5dcaf8eeafc",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 1,
            "bytes": 431783,
            "sha1": "f1e7a81be47f",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 2,
            "bytes": 238515,
            "sha1": "92a44368d5cf",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 3,
            "bytes": 285772,
            "sha1": "79190d4a3d9b",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 4,
            "bytes": 261027,
            "sha1": "4e713dbf5ff2",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 5,
            "bytes": 217957,
            "sha1": "6da790068293",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 6,
            "bytes": 190792,
            "sha1": "5f0fac34d2b2",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 7,
            "bytes": 196959,
            "sha1": "96751a51e23b",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          }
        ]
      },
      "score": 1.0
    },
    "negative": {
      "id": "c_12140",
      "text": "<|video_pad|>\nUnderstand the content of the provided video clip.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 294988,
            "sha1": "97810d4a1a0c",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 1,
            "bytes": 316136,
            "sha1": "07ff9202966b",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 2,
            "bytes": 388686,
            "sha1": "f80584505c17",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 3,
            "bytes": 345965,
            "sha1": "6304b353e04d",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 4,
            "bytes": 240051,
            "sha1": "aae29544d2d9",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 5,
            "bytes": 244562,
            "sha1": "5217139493f8",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 6,
            "bytes": 255804,
            "sha1": "ad712a083ea0",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          },
          {
            "index": 7,
            "bytes": 224098,
            "sha1": "f7251eaad8a7",
            "format": "JPEG",
            "size": [
              2560,
              1920
            ]
          }
        ]
      },
      "score": 0.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1200",
    "mode": "exhaustive",
    "positive_candidate_ids": [
      "c_12139"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 10,
    "negative_candidate_count": 9
  }
}
```
