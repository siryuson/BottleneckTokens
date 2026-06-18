# MMLongBench-Doc Training Dataset

> Status: implemented. The current implementation provides raw-preserving Lance conversion, a rendered PDF page table, and the `mmlongbench_doc_train` runtime.

MMLongBench-Doc is a long-context document understanding dataset. The archive training parser only used rows whose `evidence_pages` value contains exactly one page index, then rendered that PDF page as the positive document image.

| Property | Value |
|------|-----|
| Modality | visdoc |
| Task Type | `document_image_retrieval` |
| Usage | Training / OOD evaluation leakage audit |
| Registry Name | `mmlongbench_doc_train` |
| Conversion Script | `scripts/convert/train/convert_mmlongbench_doc_to_lance.py` |
| Runtime | `src/vlm2emb/data/datasets/mmlongbench_doc_train.py` |

## Raw Data

| Property | Value |
|------|-----|
| Original Name | MMLongBench-Doc |
| HuggingFace | [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) |
| Local Root | `./data/converted` |
| Parquet | `data/train-00000-of-00001.parquet` |
| PDF Directory | `documents/` |

The primary Lance tables preserve these raw parquet fields and do not add derived columns:

| Field | Type | Description |
|------|------|------|
| `doc_id` | string | PDF filename |
| `doc_type` | string | Document type |
| `question` | string | Question text |
| `answer` | string | Answer text |
| `evidence_pages` | string | Stringified page-index list |
| `evidence_sources` | string | Evidence source |
| `answer_format` | string | Answer format |

## Converted Layout

The recommended converted root is `./data/converted`.

The conversion command must pass the MMEB-V2 `MMLongBench-doc` eval root explicitly. This input is used to create the default leak-free training split; if it is missing or does not contain `queries.lance`, conversion fails instead of silently producing an eval-overlapping train table.

| Path | Rows | Description |
|------|------|------|
| `data/official_train_raw.lance` | 1091 | All raw parquet rows |
| `data/official_train.lance` | 473 | Archive-usable rows with one evidence page and a renderable PDF page |
| `data/official_train_without_mmeb_v2_eval.lance` | 4 | Default train split after excluding MMEB-V2 `MMLongBench-doc` eval query overlap |
| `data/pages.lance` | 418 | Rendered PDF page image table, deduplicated by `{doc_id}#page={page_index}` |
| `data/exclusions/unusable_rows.lance` | 618 | Non-single-page, missing-PDF, or render-failed rows |
| `data/exclusions/mmeb_v2_eval.lance` | 469 | Usable rows overlapping MMEB-V2 eval queries |

`data/pages.lance` schema:

| Field | Type | Description |
|------|------|------|
| `path` | string | Join key formatted as `{doc_id}#page={page_index}` |
| `doc_id` | string | PDF filename |
| `page_index` | int32 | Page index used by the archive parser |
| `image` | binary | Rendered JPEG bytes |

## Runtime Output

Default config:

```yaml
type: mmlongbench_doc_train
path: ./data/converted
split: official_train_without_mmeb_v2_eval
transform:
  query:
    instruction: "Find a document image that matches the given query:"
    instruction_body_separator: space
    trailing_newline: ensure_single
  positive:
    instruction: "Understand the content of the provided document image."
    visual_token_placement: own_line
    trailing_newline: ensure_single
  negative:
    trailing_newline: ensure_single
    empty: empty_multimodal_input
```

One real default-split example:

- Raw question: `List all the different icons about networks that can be found in Status Bar`
- Query text: `Find a document image that matches the given query: List all the different icons about networks that can be found in Status Bar\n`
- Positive text: `<|image_pad|>\nUnderstand the content of the provided document image.\n`
- Positive media: `mi_phone.pdf#page=10`

## Audit Conclusion

This dataset must not be enabled with the archive-full split by default. Among the 473 archive-usable rows in `official_train`, 469 overlap MMEB-V2 `MMLongBench-doc` eval queries by exact question text. The default training config must use `official_train_without_mmeb_v2_eval`; use `official_train` only when reproducing the archive training recipe.

The converter requires PyMuPDF to render PDF pages and a readable MMEB-V2 eval root for overlap exclusion. If either requirement is missing, the conversion script fails explicitly. Runtime loading does not depend on PyMuPDF because it reads the rendered JPEG bytes from Lance.
