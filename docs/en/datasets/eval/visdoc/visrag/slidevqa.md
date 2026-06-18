# VisRAG SlideVQA

> 演示文稿场景文档检索子集，关注图文混排页的视觉检索。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `visdoc_visrag` |
| 用途 | 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [openbmb/VisRAG-Ret-Test-SlideVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-SlideVQA) |
| 分割 | `test` |
| 论文 | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（Yu et al., 2024）([arXiv](https://arxiv.org/abs/2410.10594)) |

### Schema
- `queries`: 幻灯片问答查询
- `corpus`: 幻灯片页面图像候选
- `qrels`: 查询与页面相关性分数

### 样本
> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/visdoc-tasks/VisRAG_SlideVQA/` |
| task_type | `visdoc_visrag` |
| 聚合指标 | `ndcg@5` |

### Lance 布局
- `queries.lance`
- `candidates.lance`
- `qrels.lance`

### 样本

The examples below are sampled directly from `./data/converted`.

- The `text` field keeps the original Lance value.
- The `images` field is binary in Lance, so the docs summarize it with `count / format / size / bytes / sha1` instead of raw bytes.

#### Example 1
`query_id=2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-7-1024.jpgtcy62012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-13-1024.jpgquery_number_1`, `positive_candidate_id=2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-7-1024.jpg`

```json
{
  "queries.lance": {
    "id": "2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-7-1024.jpgtcy62012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-13-1024.jpgquery_number_1",
    "text": "Find a document image that matches the given query: In which year did the company that divested Eismann in 2004 achieve higher Organic Growth, 2003 or 2004?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-7-1024.jpg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 72106,
            "sha1": "9b10a1f60a0d",
            "format": "JPEG",
            "size": [
              1024,
              576
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-7-1024.jpgtcy62012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-13-1024.jpgquery_number_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-7-1024.jpg",
      "2012-02-20fy11roadshow-120221022442-phpapp02_95__feb-20-2012-nestl-2011-fullyear-roadshow-presentation-13-1024.jpg"
    ],
    "positive_candidate_scores": [
      1.0,
      1.0
    ],
    "total_candidate_ids": 2,
    "negative_candidate_count": 0
  }
}
```

#### Example 2
`query_id=barcampbournemouthcanvaskeynote-100410160445-phpapp02_95__html5-canvas-the-future-of-graphics-on-the-web-2-1024.jpgquery_number_1`, `positive_candidate_id=barcampbournemouthcanvaskeynote-100410160445-phpapp02_95__html5-canvas-the-future-of-graphics-on-the-web-2-1024.jpg`

```json
{
  "queries.lance": {
    "id": "barcampbournemouthcanvaskeynote-100410160445-phpapp02_95__html5-canvas-the-future-of-graphics-on-the-web-2-1024.jpgquery_number_1",
    "text": "Find a document image that matches the given query: What is Robin Hawkes' university affiliation?\n",
    "images": {
      "count": 0,
      "items": []
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "barcampbournemouthcanvaskeynote-100410160445-phpapp02_95__html5-canvas-the-future-of-graphics-on-the-web-2-1024.jpg",
      "text": "<|image_pad|>\nUnderstand the content of the provided document image.\n",
      "images": {
        "count": 1,
        "items": [
          {
            "index": 0,
            "bytes": 84974,
            "sha1": "b3f816fe78c4",
            "format": "JPEG",
            "size": [
              1024,
              768
            ]
          }
        ]
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "barcampbournemouthcanvaskeynote-100410160445-phpapp02_95__html5-canvas-the-future-of-graphics-on-the-web-2-1024.jpgquery_number_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "barcampbournemouthcanvaskeynote-100410160445-phpapp02_95__html5-canvas-the-future-of-graphics-on-the-web-2-1024.jpg"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
