# Breakfast

> 长时程早餐制作活动识别数据集，强调复杂动作序列建模。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_classification` |
| 用途 | 评测 |
| 溯源层数 | 3 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Breakfast Actions |
| 论文 | The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities（Kuehne et al., 2014）([CVPR](https://openaccess.thecvf.com/content_cvpr_2014/html/Kuehne_The_Language_of_2014_CVPR_paper.html)) |
| HuggingFace | [siyrus/BToks-Breakfast](https://huggingface.co/datasets/siyrus/BToks-Breakfast) |
| 许可证 | 参见原始数据集 |

### Schema
- `video_id`: 视频唯一标识
- `pos_text`: 正样本动作标签文本
- `neg_text`（可选）: 干扰标签列表

### 样本

> 样本截图待补充。

---

## VLM2Vec HF 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| HuggingFace | [siyrus/BToks-Breakfast](https://huggingface.co/datasets/siyrus/BToks-Breakfast) |
| 分割 | `test` |

### Schema
- `video_id`: 视频 ID
- `pos_text`: 正确类别标签
- `neg_text`（可选）: 候选负标签

### 样本

> 样本截图待补充。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 目录 | `MMEB-V2/video-tasks/Breakfast/` |
| task_type | `video_classification` |
| 评测模式 | global |

### Lance 布局
- `queries.lance`: 视频帧查询（`id`, `text`, `images`）
- `candidates.lance`: 全局类别候选（`id`, `text`, `images=[]`）
- `qrels.lance`: 稀疏相关性标注（`query_id`, `candidate_ids`, `candidate_scores`）

### 样本

以下示例基于当前 runtime canonical 输出生成，数据来源为 `./data/converted`.

- `text` 展示当前 runtime canonical 的模型可见文本；parser 可能在 Lance 原始字段基础上重组 visual token 布局、选项块和尾部换行。
- `images` 为二进制列，文档中按 `count / format / size / bytes / sha1` 展示摘要，不直接展开原始 bytes。

#### 示例 1
`query_id=q_0`，`positive_candidate_id=milk`

```json
{
  "queries.lance": {
    "id": "q_0",
    "text": "<|video_pad|>\nRecognize the breakfast type that the person is cooking in the video.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 27723,
          "sha1": "71e6804263c5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 30346,
          "sha1": "e00deb9d6ada",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 30577,
          "sha1": "d3d123f5e954",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 30740,
          "sha1": "b9546f251292",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 31100,
          "sha1": "4f67536fd556",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 31171,
          "sha1": "612f0a056bca",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 30390,
          "sha1": "7fb6d8ffc1f1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 29328,
          "sha1": "52ea9484aac8",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 29943,
          "sha1": "8ac620812ca1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 30302,
          "sha1": "5c87d4c0e21c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 29320,
          "sha1": "9a2bde6977bf",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 29558,
          "sha1": "9a464e55f987",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 27933,
          "sha1": "a088a9d021e5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 27781,
          "sha1": "faa7277a2e0c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 28709,
          "sha1": "4327283e9a8c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 26697,
          "sha1": "bb716dd7a6ea",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 26466,
          "sha1": "3cd62ad4acc9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 26965,
          "sha1": "7810c78f6522",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 26557,
          "sha1": "afc32e17d0be",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 28867,
          "sha1": "aa2d4ef83d4d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 27246,
          "sha1": "e88779941d1a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 27764,
          "sha1": "a6debc1cb6f4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 26714,
          "sha1": "9e8b46b5cfa9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 28130,
          "sha1": "03ae0f38c16a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 28989,
          "sha1": "b0e5bb39512e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 29301,
          "sha1": "51c5f540cd49",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 28818,
          "sha1": "1d067a52ddab",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 27084,
          "sha1": "1613cf6debcd",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 27286,
          "sha1": "142260a62e5a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 25522,
          "sha1": "0c46ea433cea",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 25793,
          "sha1": "215dddac9705",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 26832,
          "sha1": "9da70a13b621",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 27206,
          "sha1": "36087572d8b9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 27378,
          "sha1": "a8171115a283",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 27597,
          "sha1": "830c54308ff6",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 35,
          "bytes": 27338,
          "sha1": "840c3b490758",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 36,
          "bytes": 27155,
          "sha1": "68b3e762dcd3",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 37,
          "bytes": 26986,
          "sha1": "1b93cbea4f23",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 38,
          "bytes": 27058,
          "sha1": "616a4f77f37d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 39,
          "bytes": 27066,
          "sha1": "4d2185dc4607",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 40,
          "bytes": 27189,
          "sha1": "ed2f085d4007",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 41,
          "bytes": 27032,
          "sha1": "cbb38a3cb7c8",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 42,
          "bytes": 27939,
          "sha1": "b5de2d72d87e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 43,
          "bytes": 25200,
          "sha1": "a23d977017c8",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 44,
          "bytes": 26585,
          "sha1": "c89745af4c7e",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 45,
          "bytes": 27186,
          "sha1": "8ca920334c6a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 46,
          "bytes": 27314,
          "sha1": "f0f349c2f8a7",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 47,
          "bytes": 27020,
          "sha1": "bc2a5625f0a9",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 48,
          "bytes": 27900,
          "sha1": "6f61a1ae3912",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 49,
          "bytes": 27946,
          "sha1": "716f46f19ce3",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 50,
          "bytes": 27989,
          "sha1": "923a6e97ca8f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 51,
          "bytes": 27366,
          "sha1": "75f88e19c1d4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 52,
          "bytes": 28790,
          "sha1": "7d8883c006a1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 53,
          "bytes": 28817,
          "sha1": "f98cbd070168",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 54,
          "bytes": 29554,
          "sha1": "98bbb001149f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 55,
          "bytes": 27824,
          "sha1": "56e5ca8434d0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 56,
          "bytes": 27562,
          "sha1": "ccfd4f3efae4",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 57,
          "bytes": 27827,
          "sha1": "2a2c5179d859",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 58,
          "bytes": 26115,
          "sha1": "72a955e367e1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 59,
          "bytes": 26632,
          "sha1": "52d0290b7bc1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 60,
          "bytes": 26897,
          "sha1": "69a714023dbd",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 61,
          "bytes": 26950,
          "sha1": "6906b538789f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 62,
          "bytes": 28768,
          "sha1": "69d799d3c6fc",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 63,
          "bytes": 30257,
          "sha1": "d217b2d265be",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "milk",
      "text": "milk",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_0",
    "mode": "sparse",
    "positive_candidate_ids": [
      "milk"
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
`query_id=q_1`，`positive_candidate_id=pancake`

```json
{
  "queries.lance": {
    "id": "q_1",
    "text": "<|video_pad|>\nRecognize the breakfast type that the person is cooking in the video.\n",
    "images": {
      "count": 64,
      "items": [
        {
          "index": 0,
          "bytes": 19719,
          "sha1": "adddae2e4329",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 1,
          "bytes": 22572,
          "sha1": "9bdd081cd010",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 2,
          "bytes": 23313,
          "sha1": "c178c226711b",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 3,
          "bytes": 23683,
          "sha1": "6c5bd8b2ab2f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 4,
          "bytes": 21868,
          "sha1": "cda4ad65aa09",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 5,
          "bytes": 23263,
          "sha1": "87b06e2c0865",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 6,
          "bytes": 23669,
          "sha1": "a7367e545918",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 7,
          "bytes": 23797,
          "sha1": "02bef56f4c55",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 8,
          "bytes": 22006,
          "sha1": "b5fbbeba9d40",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 9,
          "bytes": 22973,
          "sha1": "2165b48fc883",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 10,
          "bytes": 23221,
          "sha1": "5d9446ad7b13",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 11,
          "bytes": 23797,
          "sha1": "2a672462c361",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 12,
          "bytes": 21977,
          "sha1": "839670a06dbc",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 13,
          "bytes": 22304,
          "sha1": "32c62b624484",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 14,
          "bytes": 22971,
          "sha1": "5df2d504492d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 15,
          "bytes": 23092,
          "sha1": "3967f3b6aaf1",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 16,
          "bytes": 21728,
          "sha1": "34ef22426d05",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 17,
          "bytes": 22419,
          "sha1": "a30457deeabf",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 18,
          "bytes": 22692,
          "sha1": "6df5ee8d9cec",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 19,
          "bytes": 22998,
          "sha1": "d30b9164a820",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 20,
          "bytes": 21959,
          "sha1": "eb23d016f3fc",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 21,
          "bytes": 22897,
          "sha1": "2c7ce08128ba",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 22,
          "bytes": 23852,
          "sha1": "1da47acc895a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 23,
          "bytes": 24200,
          "sha1": "a1acbddf660f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 24,
          "bytes": 22752,
          "sha1": "f006f56dbcee",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 25,
          "bytes": 23573,
          "sha1": "e465e3a3b88c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 26,
          "bytes": 23098,
          "sha1": "494a6ab3ef99",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 27,
          "bytes": 23750,
          "sha1": "497aa84e1bc0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 28,
          "bytes": 22182,
          "sha1": "83cf1701a052",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 29,
          "bytes": 22207,
          "sha1": "2695b2f9ad8a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 30,
          "bytes": 22845,
          "sha1": "6c2c68ab198a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 31,
          "bytes": 24187,
          "sha1": "885dadf80478",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 32,
          "bytes": 22364,
          "sha1": "14fd93ecf33f",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 33,
          "bytes": 23503,
          "sha1": "9a1915e12a69",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 34,
          "bytes": 23726,
          "sha1": "1fd0e47abf58",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 35,
          "bytes": 24290,
          "sha1": "f453f0723366",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 36,
          "bytes": 22978,
          "sha1": "4359ab304ec5",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 37,
          "bytes": 24392,
          "sha1": "6caa4e3f0de2",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 38,
          "bytes": 22261,
          "sha1": "0315c66e5c16",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 39,
          "bytes": 23622,
          "sha1": "8459f67ec10d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 40,
          "bytes": 23429,
          "sha1": "31bb16b6c1cc",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 41,
          "bytes": 22835,
          "sha1": "c91307bfdc67",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 42,
          "bytes": 23577,
          "sha1": "0e80a76ddfe6",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 43,
          "bytes": 24567,
          "sha1": "d4fb91a84461",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 44,
          "bytes": 23311,
          "sha1": "cdb1f87ce027",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 45,
          "bytes": 23591,
          "sha1": "797f38d34fb6",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 46,
          "bytes": 24004,
          "sha1": "d4f49e6771fd",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 47,
          "bytes": 24272,
          "sha1": "d6a6528c8716",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 48,
          "bytes": 23873,
          "sha1": "4ee159d8d33d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 49,
          "bytes": 24158,
          "sha1": "a89dd4e7d2f6",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 50,
          "bytes": 23033,
          "sha1": "66dc83d2acb0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 51,
          "bytes": 23421,
          "sha1": "6ffd916f803a",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 52,
          "bytes": 23985,
          "sha1": "2466e1be515d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 53,
          "bytes": 24475,
          "sha1": "f89eb3dc044c",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 54,
          "bytes": 24477,
          "sha1": "ae5e76152d8d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 55,
          "bytes": 24491,
          "sha1": "913ba059b117",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 56,
          "bytes": 23035,
          "sha1": "b92369205d1d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 57,
          "bytes": 24317,
          "sha1": "d9b67c636907",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 58,
          "bytes": 24629,
          "sha1": "0480d88780ce",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 59,
          "bytes": 24883,
          "sha1": "7a464ff85c5d",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 60,
          "bytes": 23740,
          "sha1": "8632328b6258",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 61,
          "bytes": 23988,
          "sha1": "24825efa93b7",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 62,
          "bytes": 24334,
          "sha1": "2f27e42529c0",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        },
        {
          "index": 63,
          "bytes": 23784,
          "sha1": "4d7c5a149c10",
          "format": "JPEG",
          "size": [
            320,
            240
          ]
        }
      ]
    }
  },
  "candidates.lance": {
    "positive": {
      "id": "pancake",
      "text": "pancake",
      "images": {
        "count": 0,
        "items": []
      },
      "score": 1.0
    }
  },
  "qrels.lance": {
    "query_id": "q_1",
    "mode": "sparse",
    "positive_candidate_ids": [
      "pancake"
    ],
    "positive_candidate_scores": [
      1.0
    ],
    "total_candidate_ids": 1,
    "negative_candidate_count": 0
  }
}
```
