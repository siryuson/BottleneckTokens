# Video Evaluation Datasets

18 evaluation subsets across 4 task types. Metric: hit@1

---

## Sample Contract

The `BToks Lance Format` JSON examples in each subset page show the current runtime-canonical model-visible text, not a literal copy of the raw Lance `text` column. The MMEB-V2 video eval parser reconstructs visual-token layout, option blocks, and trailing newlines after reading the raw Lance fields; binary `images` columns are summarized in the docs.

The `MVBench` page documents the raw-storage and runtime-parser split separately, so it does not use these JSON sample sections.

---

## Subsets

### Classification (5) — task_type: `video_classification`

| Subset |
|--------|
| [K700](./classification/k700.md) |
| [UCF101](./classification/ucf101.md) |
| [HMDB51](./classification/hmdb51.md) |
| [SmthSmthV2](./classification/smthsmthv2.md) |
| [Breakfast](./classification/breakfast.md) |

### Video QA (5) — task_type: `video_question_answer`

| Subset |
|--------|
| [Video-MME](./qa/video-mme.md) |
| [MVBench](./qa/mvbench.md) |
| [NExTQA](./qa/nextqa.md) |
| [EgoSchema](./qa/egoschema.md) |
| [ActivityNetQA](./qa/activitynetqa.md) |

### Video Retrieval (5) — task_type: `video_retrieval`

| Subset |
|--------|
| [MSR-VTT](./retrieval/msr-vtt.md) |
| [MSVD](./retrieval/msvd.md) |
| [DiDeMo](./retrieval/didemo.md) |
| [VATEX](./retrieval/vatex.md) |
| [YouCook2](./retrieval/youcook2.md) |

### Moment Retrieval (3) — task_type: `video_moment_retrieval`

| Subset |
|--------|
| [QVHighlight](./moment-retrieval/qvhighlight.md) |
| [Charades-STA](./moment-retrieval/charades-sta.md) |
| [MomentSeeker](./moment-retrieval/momentseeker.md) |
