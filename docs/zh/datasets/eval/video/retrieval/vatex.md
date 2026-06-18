# VATEX

> 高质量多语言视频描述与检索数据集，支持跨语言视频语义匹配。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_retrieval` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | VATEX |
| 论文 | VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research（Wang et al., 2019）([arXiv](https://arxiv.org/abs/1904.03493)) |
| HuggingFace | [siyrus/BToks-VATEX](https://huggingface.co/datasets/siyrus/BToks-VATEX) |
| 许可证 | 参见原始数据集 |

### Schema
- `videoID`: 视频标识
- `enCap`: 英文描述列表

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-VATEX](https://huggingface.co/datasets/siyrus/BToks-VATEX) |
| 分割 | `test` |

### Schema
- `videoID`: 视频 ID
- `enCap`: 文本查询候选描述

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/VATEX/` |
| task_type | `video_retrieval` |
| 评测模式 | global |

### Lance 布局
- `queries.lance`: 文本查询（`id`, `text`, `images=[]`）
- `candidates.lance`: 视频帧候选（`id`, `text`, `images`）
- `qrels.lance`: 稀疏相关性标注（单正样本为主）

### 样本

以下示例基于当前 runtime canonical 输出生成，数据来源为 `./data/converted`.

- `text` 展示当前 runtime canonical 的模型可见文本；parser 可能在 Lance 原始字段基础上重组 visual token 布局、选项块和尾部换行。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=c_0`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "Select a video that fits the description provided: A young man is showing the polish, water. old soft cloth and brush needed to polish shoes with.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_0",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 1618,
            "sha1": "ac7db4321cb4",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 1,
            "bytes": 8870,
            "sha1": "18e8b5a62201",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 2,
            "bytes": 8627,
            "sha1": "cd3f1e2d26e9",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 3,
            "bytes": 9246,
            "sha1": "774a103a7e7f",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 4,
            "bytes": 11619,
            "sha1": "1d3cc02f8f45",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 5,
            "bytes": 12574,
            "sha1": "96c512cc1e1a",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 6,
            "bytes": 15885,
            "sha1": "481329fdfbf2",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          },
          {
            "index": 7,
            "bytes": 13125,
            "sha1": "406ec3adc166",
            "format": "JPEG",
            "size": [
              640,
              360
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_0"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```

#### 示例 2
`query_id=q_1`，`positive_candidate_id=c_1`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "Select a video that fits the description provided: A woman explains how to sew a beanie as she sews a beanie in her hands.\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "c_1",
      "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
      "images": {
        "count": 8,
        "items": [
          {
            "index": 0,
            "bytes": 5622,
            "sha1": "3e26306d44b5",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 1,
            "bytes": 17016,
            "sha1": "918a9186de22",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 2,
            "bytes": 17997,
            "sha1": "385cd73ad356",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 3,
            "bytes": 16883,
            "sha1": "1e4419fc592e",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 4,
            "bytes": 18213,
            "sha1": "31deb5f5a3e6",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 5,
            "bytes": 21265,
            "sha1": "f95c38cfa190",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 6,
            "bytes": 22182,
            "sha1": "f48613d48642",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          },
          {
            "index": 7,
            "bytes": 34561,
            "sha1": "e157e9ce164d",
            "format": "JPEG",
            "size": [
              1280,
              720
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "c_1"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
