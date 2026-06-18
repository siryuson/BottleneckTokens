# MVBench

> 聚焦视频多任务时序理解的多选问答评测子集。

| 属性 | 值 |
|------|-----|
| 模态 | video |
| 任务类型 | `video_question_answer` |
| 用途 | 评测 |
| benchmark 主指标 | `hit@1` |

## 数据来源

| 属性 | 值 |
|------|-----|
| 原始名称 | MVBench |
| HuggingFace | [siyrus/BToks-MVBench](https://huggingface.co/datasets/siyrus/BToks-MVBench) |
| 帧目录 | `MMEB-V2/video-tasks/frames/video_qa/MVBench/` |
| 原始 split | `train` |

补充说明：

- conversion 会合并 MVBench 的 20 个原始 subset。
- 每条 query 会保留 `_source_file`，用于从 `frames/video_qa/MVBench/<subset>/<video>` 定位帧目录。

## Conversion 存储格式

当前 `MMEB-V2/video-tasks/MVBench/` 的 Lance 产物强调“保留原始字段 + 固定评测事实”，而不是直接存模型最终可见文本。

### `queries.lance`

核心字段：

- `id`
- `question`
- `candidates`
- `answer`
- `_source_file`
- `video`

说明：

- `video` 存完整帧列表，运行时再按 `num_frames` 做确定性均匀采样。
- `id` 当前使用整数 ID；runtime 只会做 `str(id)`，不会再补 `q_` 前缀。

### `candidates.lance`

核心字段：

- `id`
- `text`

说明：

- candidate 文本在 conversion 时就写成带 `(A)/(B)/(C)...` 前缀的多选项文本。
- `id` 同样使用整数 ID，不会再补 `c_` 前缀。

### `qrels.lance`

核心字段：

- `query_id`
- `mode`
- `candidate_ids`
- `candidate_scores`

说明：

- `MVBench` 当前使用 `mode=sparse`。
- 每条 query 只记录正确答案对应的正例 `candidate_id`，不把整组选项写成 `0/1` 穷举分数。
- 文档中如果展示负例 candidate，只能作为人工审查对照项，不能当成 qrels 原始字段。

## Runtime 默认呈现

`MMEBBenchmark` 当前通过 `src/vlm2emb/data/datasets/mmeb_v2/loader.py` 中的 `build_mmeb_eval_dataset(...)` 进入 `MVBench` parser。

默认行为：

- `runtime_mode="canonical"`
- `num_frames=8`

runtime 负责的事情：

1. 读取 `question + candidates + video`，而不是直接读取一个已经渲染好的 `queries.lance.text`
2. 用 parser 把问题和选项重组为多行 `Options:` 模板
3. 在 query 侧补 `<|video_pad|>`
4. 对完整帧列表做确定性均匀采样
5. candidate 侧直接沿用 conversion 已写好的 `(A)/(B)/(C)...` 前缀文本

`origin` / `canonical` 的边界：

- `origin`：保留 archive parser 的原始换行与标点表面
- `canonical`：在 `origin` 基础上做项目批准的 whitespace 归一

## 当前维护约束

1. `MVBench` 页面必须区分“conversion 存储事实”和“runtime 模型可见文本”，不能混写。
2. 如果展示样本，需要明确它来自：
   - raw Lance row；或
   - parser transform 输出的 runtime item。
3. 当前 12-03 的本地 smoke / 回归根目录是 `./data/converted`.

## 相关实现

- `src/vlm2emb/data/conversion/mmeb_v2/video.py`
- `src/vlm2emb/data/datasets/mmeb_v2/parsers/mvbench.py`
- `src/vlm2emb/data/datasets/mmeb_v2/loader.py`
