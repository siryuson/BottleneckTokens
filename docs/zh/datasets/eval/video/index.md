# Video 评测数据集

18 个评测子集，4 种任务类型。指标: hit@1

---

## 样本口径

各子集页面的 `BToks Lance 格式` JSON 示例展示当前 runtime canonical 输出中的模型可见文本，而不是简单复印 raw Lance `text` 列。MMEB-V2 video eval 的 parser 会在读取 Lance 原始字段后重组 visual token 布局、选项块和尾部换行；文档中的 `images` 二进制列只展示摘要。

`MVBench` 页面单独记录 raw storage 与 runtime parser 的拆分，不使用这些 JSON 样本段落。

---

## 子集列表

### 分类 (5) — task_type: `video_classification`

| 子集 |
|------|
| [K700](./classification/k700.md) |
| [UCF101](./classification/ucf101.md) |
| [HMDB51](./classification/hmdb51.md) |
| [SmthSmthV2](./classification/smthsmthv2.md) |
| [Breakfast](./classification/breakfast.md) |

### 视频问答 (5) — task_type: `video_question_answer`

| 子集 |
|------|
| [Video-MME](./qa/video-mme.md) |
| [MVBench](./qa/mvbench.md) |
| [NExTQA](./qa/nextqa.md) |
| [EgoSchema](./qa/egoschema.md) |
| [ActivityNetQA](./qa/activitynetqa.md) |

### 视频检索 (5) — task_type: `video_retrieval`

| 子集 |
|------|
| [MSR-VTT](./retrieval/msr-vtt.md) |
| [MSVD](./retrieval/msvd.md) |
| [DiDeMo](./retrieval/didemo.md) |
| [VATEX](./retrieval/vatex.md) |
| [YouCook2](./retrieval/youcook2.md) |

### 时序检索 (3) — task_type: `video_moment_retrieval`

| 子集 |
|------|
| [QVHighlight](./moment-retrieval/qvhighlight.md) |
| [Charades-STA](./moment-retrieval/charades-sta.md) |
| [MomentSeeker](./moment-retrieval/momentseeker.md) |
