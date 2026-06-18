"""MMLongBench-Doc training conversion utilities."""

from __future__ import annotations

import ast
import shutil
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

RAW_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("evidence_pages", pa.string()),
        pa.field("evidence_sources", pa.string()),
        pa.field("answer_format", pa.string()),
    ]
)
EXCLUSION_SCHEMA = RAW_SCHEMA.append(pa.field("exclusion_reason", pa.string()))
PAGE_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("page_index", pa.int32()),
        pa.field("image", pa.binary()),
    ]
)
RenderPage = Callable[[Path, int], bytes]


def _parse_single_evidence_page(value: Any) -> int | None:
    """Parse the archive-supported single evidence-page value."""

    try:
        pages = ast.literal_eval(str(value or "[]"))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(pages, list) or len(pages) != 1:
        return None
    page = pages[0]
    if not isinstance(page, int):
        return None
    return page


def mmlongbench_doc_page_key(doc_id: str, page_index: int) -> str:
    """Build the stable page-image lookup key."""

    return f"{doc_id}#page={page_index}"


def _raw_parquet_paths(input_root: Path) -> list[Path]:
    """Return the MMLongBench-Doc parquet shards in deterministic order."""

    data_root = input_root / "data"
    paths = sorted(data_root.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing MMLongBench-Doc train parquet files under: {data_root}")
    return paths


def _load_raw_rows(input_root: Path) -> list[dict[str, Any]]:
    """Load raw parquet rows while preserving source columns."""

    rows: list[dict[str, Any]] = []
    for parquet_path in _raw_parquet_paths(input_root):
        table = pq.read_table(parquet_path, schema=RAW_SCHEMA)
        rows.extend(table.to_pylist())
    return rows


def _batched(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield fixed-size row batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _record_batch_reader(
    rows: Iterable[dict[str, Any]],
    *,
    schema: pa.Schema,
    batch_size: int,
) -> pa.RecordBatchReader:
    """Create an Arrow reader that also represents empty tables."""

    batches = [
        pa.RecordBatch.from_pylist(batch, schema=schema)
        for batch in _batched(rows, batch_size)
    ]
    if not batches:
        batches = [pa.RecordBatch.from_pylist([], schema=schema)]
    return pa.RecordBatchReader.from_batches(schema, batches)


def _render_pdf_page_jpeg(pdf_path: Path, page_index: int) -> bytes:
    """Render one PDF page to JPEG bytes with PyMuPDF."""

    try:
        import fitz
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
        raise ModuleNotFoundError(
            "MMLongBench-Doc conversion requires PyMuPDF. Install `PyMuPDF` "
            "before running the full converter."
        ) from exc

    with fitz.open(pdf_path) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
        return bytes(pixmap.tobytes("jpeg"))


def _render_page_task(args: tuple[str, str, int]) -> tuple[str, str, int, bytes | str, bool]:
    """Render one page in a worker process."""

    pdf_path_str, doc_id, page_index = args
    pdf_path = Path(pdf_path_str)
    page_key = mmlongbench_doc_page_key(doc_id, page_index)
    try:
        return page_key, doc_id, page_index, _render_pdf_page_jpeg(pdf_path, page_index), True
    except Exception as exc:  # pragma: no cover - exercised by integration conversion.
        return page_key, pdf_path.name, page_index, repr(exc), False


def load_eval_overlap_questions(eval_root: Path | None) -> set[str]:
    """Load MMEB-V2 MMLongBench-doc eval questions for leakage exclusion."""

    if eval_root is None:
        raise ValueError("MMLongBench-Doc conversion requires eval_root for MMEB-V2 leakage exclusion.")
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing MMLongBench-Doc eval root: {eval_root}")
    queries_path = eval_root / "queries.lance"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing MMLongBench-Doc eval queries table: {queries_path}")
    dataset = lance.dataset(str(queries_path))
    for column in ("query", "qry_text", "text"):
        if column in dataset.schema.names:
            return {
                str(row[column] or "").strip()
                for row in dataset.to_table(columns=[column]).to_pylist()
            }
    raise ValueError(f"MMLongBench-Doc eval queries table has no query text column: {queries_path}")


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_mmlongbench_doc_train_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the MMLongBench-Doc runtime."""

    pages_path = output_root / "data" / "pages.lance"
    if not pages_path.exists():
        raise FileNotFoundError(f"Missing MMLongBench-Doc page table: {pages_path}")
    _create_scalar_index(pages_path, "path")


def _classify_rows(
    rows: list[dict[str, Any]],
    *,
    input_root: Path,
) -> tuple[list[tuple[dict[str, Any], int, str]], list[dict[str, Any]]]:
    """Split raw rows into archive-usable candidates and unusable rows."""

    documents_root = input_root / "documents"
    candidates: list[tuple[dict[str, Any], int, str]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        page_index = _parse_single_evidence_page(row.get("evidence_pages"))
        if page_index is None:
            exclusions.append({**row, "exclusion_reason": "non_single_evidence_page"})
            continue
        doc_id = str(row.get("doc_id") or "")
        pdf_path = documents_root / doc_id
        if not pdf_path.is_file():
            exclusions.append({**row, "exclusion_reason": "missing_pdf"})
            continue
        candidates.append((row, page_index, mmlongbench_doc_page_key(doc_id, page_index)))
    return candidates, exclusions


def _render_pages(
    candidates: list[tuple[dict[str, Any], int, str]],
    *,
    input_root: Path,
    render_page: RenderPage,
    num_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Render unique candidate pages and report failed page keys."""

    documents_root = input_root / "documents"
    unique_pages: dict[str, tuple[str, int]] = {}
    for row, page_index, page_key in candidates:
        unique_pages.setdefault(page_key, (str(row.get("doc_id") or ""), page_index))

    page_rows: list[dict[str, Any]] = []
    failed_pages: dict[str, str] = {}
    if num_workers <= 1 or render_page is not _render_pdf_page_jpeg:
        for page_key, (doc_id, page_index) in unique_pages.items():
            try:
                image = render_page(documents_root / doc_id, page_index)
            except Exception as exc:
                failed_pages[page_key] = repr(exc)
                continue
            page_rows.append(
                {
                    "path": page_key,
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "image": image,
                }
            )
        return page_rows, failed_pages

    tasks = [
        (str(documents_root / doc_id), doc_id, page_index)
        for doc_id, page_index in unique_pages.values()
    ]
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn")) as pool:
        for page_key, doc_id, page_index, payload, ok in pool.map(_render_page_task, tasks):
            if ok:
                page_rows.append(
                    {
                        "path": page_key,
                        "doc_id": doc_id,
                        "page_index": page_index,
                        "image": payload,
                    }
                )
            else:
                failed_pages[page_key] = str(payload)
    return page_rows, failed_pages


def write_mmlongbench_doc_train_root(
    *,
    input_root: Path,
    output_root: Path,
    eval_root: Path | None = None,
    overwrite: bool = False,
    batch_size: int = 1024,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    num_workers: int = 1,
    render_page: RenderPage = _render_pdf_page_jpeg,
) -> dict[str, Any]:
    """Convert MMLongBench-Doc into raw-preserving Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    rows = _load_raw_rows(input_root)
    candidates, unusable_rows = _classify_rows(rows, input_root=input_root)
    page_rows, failed_pages = _render_pages(
        candidates,
        input_root=input_root,
        render_page=render_page,
        num_workers=num_workers,
    )

    official_train: list[dict[str, Any]] = []
    for row, _page_index, page_key in candidates:
        render_error = failed_pages.get(page_key)
        if render_error is not None:
            unusable_rows.append({**row, "exclusion_reason": f"render_error:{render_error}"})
            continue
        official_train.append(row)

    eval_questions = load_eval_overlap_questions(eval_root)
    train_without_eval = [
        row for row in official_train if str(row.get("question") or "").strip() not in eval_questions
    ]
    eval_exclusions = [
        {**row, "exclusion_reason": "mmeb_v2_eval_query_overlap"}
        for row in official_train
        if str(row.get("question") or "").strip() in eval_questions
    ]

    split_rows = {
        "official_train_raw": rows,
        "official_train": official_train,
        "official_train_without_mmeb_v2_eval": train_without_eval,
    }
    for split_name, split in split_rows.items():
        lance.write_dataset(
            _record_batch_reader(split, schema=RAW_SCHEMA, batch_size=batch_size),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=RAW_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    lance.write_dataset(
        _record_batch_reader(unusable_rows, schema=EXCLUSION_SCHEMA, batch_size=batch_size),
        str(output_root / "data" / "exclusions" / "unusable_rows.lance"),
        schema=EXCLUSION_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        _record_batch_reader(eval_exclusions, schema=EXCLUSION_SCHEMA, batch_size=batch_size),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=EXCLUSION_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        _record_batch_reader(page_rows, schema=PAGE_SCHEMA, batch_size=batch_size),
        str(output_root / "data" / "pages.lance"),
        schema=PAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_mmlongbench_doc_train_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "mmlongbench_doc_train_raw_lance_v1",
        "splits": {name: len(split) for name, split in split_rows.items()},
        "unusable_rows": len(unusable_rows),
        "mmeb_v2_eval_exclusion_rows": len(eval_exclusions),
        "page_rows": len(page_rows),
    }


__all__ = [
    "EXCLUSION_SCHEMA",
    "PAGE_SCHEMA",
    "RAW_SCHEMA",
    "ensure_mmlongbench_doc_train_indices",
    "load_eval_overlap_questions",
    "mmlongbench_doc_page_key",
    "write_mmlongbench_doc_train_root",
]
