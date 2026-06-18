# MVBench

> A multi-choice video QA benchmark focused on temporal understanding.

| Property | Value |
|------|-----|
| Modality | video |
| Task type | `video_question_answer` |
| Usage | evaluation |
| Benchmark primary metric | `hit@1` |

## Data Sources

| Property | Value |
|------|-----|
| Original name | MVBench |
| HuggingFace | [siyrus/BToks-MVBench](https://huggingface.co/datasets/siyrus/BToks-MVBench) |
| Frame root | `MMEB-V2/video-tasks/frames/video_qa/MVBench/` |
| Original split | `train` |

Additional notes:

- conversion merges the 20 original MVBench subsets
- each query keeps `_source_file` so frames can be resolved from `frames/video_qa/MVBench/<subset>/<video>`

## Conversion Storage Format

The current `MMEB-V2/video-tasks/MVBench/` artifact preserves raw fields and frozen evaluation facts. It does not store the final model-visible prompt surface directly.

### `queries.lance`

Primary fields:

- `id`
- `question`
- `candidates`
- `answer`
- `_source_file`
- `video`

Notes:

- `video` stores the full frame list; runtime performs deterministic uniform sampling with `num_frames`
- `id` is currently an integer id; runtime only converts it with `str(id)` and does not add a `q_` prefix

### `candidates.lance`

Primary fields:

- `id`
- `text`

Notes:

- candidate text is written during conversion with the `(A)/(B)/(C)...` multiple-choice prefix already attached
- `id` is also an integer id; runtime does not add a `c_` prefix

### `qrels.lance`

Primary fields:

- `query_id`
- `mode`
- `candidate_ids`
- `candidate_scores`

Notes:

- MVBench currently uses `mode=sparse`
- each query stores only the positive `candidate_id` for the correct answer, instead of writing the full option pool as exhaustive `0/1` scores
- if this page shows a negative candidate example, treat it as a manual audit contrast only, not as a raw qrels field

## Runtime Default Transform

`MMEBBenchmark` now enters MVBench through `build_mmeb_eval_dataset(...)` in `src/vlm2emb/data/datasets/mmeb_v2/loader.py`.

Default runtime knobs:

- `runtime_mode="canonical"`
- `num_frames=8`

Runtime is responsible for:

1. reading `question + candidates + video` instead of a preformatted `queries.lance.text`
2. rebuilding the question prompt into the multi-line `Options:` template
3. prepending `<|video_pad|>` on the query side
4. applying deterministic uniform frame sampling
5. reusing the conversion-written `(A)/(B)/(C)...` candidate text directly

`origin` vs `canonical`:

- `origin`: preserves the archive parser's original line-wrap and punctuation surface
- `canonical`: keeps the same parser-owned structure but applies the approved whitespace normalization

## Current Maintenance Constraints

1. This page must keep conversion storage facts separate from runtime model-visible text.
2. Any sample shown here must be labeled explicitly as either:
   - a raw Lance row; or
   - a runtime item emitted by the parser transform.
3. The current local smoke / regression root for 12-03 is `./data/converted`.

## Related Implementation

- `src/vlm2emb/data/conversion/mmeb_v2/video.py`
- `src/vlm2emb/data/datasets/mmeb_v2/parsers/mvbench.py`
- `src/vlm2emb/data/datasets/mmeb_v2/loader.py`
