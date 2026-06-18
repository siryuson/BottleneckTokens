"""Tests for MMLongBench-Doc conversion and runtime."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.conversion.mmlongbench_doc_train import (
    RAW_SCHEMA,
    write_mmlongbench_doc_train_root,
)


def _jpeg_bytes() -> bytes:
    """Return a tiny valid JPEG payload."""

    image = Image.new("RGB", (4, 4), color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_lance_rows(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Write a small Lance table."""

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            schema,
            [pa.RecordBatch.from_pylist(rows, schema=schema)],
        ),
        str(path),
        schema=schema,
        mode="create",
    )


def test_mmlongbench_doc_conversion_filters_eval_overlap_and_runtime(tmp_path: Path) -> None:
    """Conversion keeps archive-usable rows and defaults runtime to leak-free split."""

    raw_root = tmp_path / "raw"
    data_root = raw_root / "data"
    documents_root = raw_root / "documents"
    data_root.mkdir(parents=True)
    documents_root.mkdir()
    (documents_root / "doc-a.pdf").write_bytes(b"%PDF-1.4")

    rows = [
        {
            "doc_id": "doc-a.pdf",
            "doc_type": "manual",
            "question": "Eval overlap question?",
            "answer": "yes",
            "evidence_pages": "[0]",
            "evidence_sources": "[]",
            "answer_format": "short",
        },
        {
            "doc_id": "doc-a.pdf",
            "doc_type": "manual",
            "question": "Safe training question?",
            "answer": "safe",
            "evidence_pages": "[1]",
            "evidence_sources": "[]",
            "answer_format": "short",
        },
        {
            "doc_id": "doc-a.pdf",
            "doc_type": "manual",
            "question": "Multi page question?",
            "answer": "multi",
            "evidence_pages": "[0, 1]",
            "evidence_sources": "[]",
            "answer_format": "short",
        },
        {
            "doc_id": "missing.pdf",
            "doc_type": "manual",
            "question": "Missing PDF question?",
            "answer": "missing",
            "evidence_pages": "[0]",
            "evidence_sources": "[]",
            "answer_format": "short",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA), data_root / "train-00000-of-00001.parquet")

    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _write_lance_rows(
        eval_root / "queries.lance",
        [{"id": 0, "query": "Eval overlap question?", "corpus_range": [0]}],
        pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("query", pa.string()),
                pa.field("corpus_range", pa.list_(pa.int64())),
            ]
        ),
    )

    output_root = tmp_path / "converted"
    summary = write_mmlongbench_doc_train_root(
        input_root=raw_root,
        output_root=output_root,
        eval_root=eval_root,
        render_page=lambda _path, _page: _jpeg_bytes(),
    )

    assert summary["splits"] == {
        "official_train_raw": 4,
        "official_train": 2,
        "official_train_without_mmeb_v2_eval": 1,
    }
    assert summary["unusable_rows"] == 2
    assert summary["mmeb_v2_eval_exclusion_rows"] == 1
    assert summary["page_rows"] == 2

    dataset = AutoDataset.from_config(
        {
            "type": "mmlongbench_doc_train",
            "path": str(output_root),
            "transform": {
                "query": {
                    "instruction": "Find a document image that matches the given query:",
                    "instruction_body_separator": "space",
                    "trailing_newline": "ensure_single",
                },
                "positive": {
                    "instruction": "Understand the content of the provided document image.",
                    "visual_token_placement": "own_line",
                    "trailing_newline": "ensure_single",
                },
                "negative": {"empty": "empty_multimodal_input"},
            },
        }
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["query"]["text"] == (
        "Find a document image that matches the given query: Safe training question?\n"
    )
    assert sample["positive"]["text"] == (
        "<|image_pad|>\nUnderstand the content of the provided document image.\n"
    )
    assert sample["positive"]["media"][0]["content"].size == (4, 4)
    assert sample["metadata"]["doc_id"] == "doc-a.pdf"
    assert sample["metadata"]["page_index"] == 1


def test_mmlongbench_doc_conversion_requires_eval_root(tmp_path: Path) -> None:
    """Conversion refuses to silently skip MMEB-V2 leakage exclusion."""

    raw_root = tmp_path / "raw"
    data_root = raw_root / "data"
    documents_root = raw_root / "documents"
    data_root.mkdir(parents=True)
    documents_root.mkdir()
    (documents_root / "doc-a.pdf").write_bytes(b"%PDF-1.4")
    rows = [
        {
            "doc_id": "doc-a.pdf",
            "doc_type": "manual",
            "question": "Could overlap eval?",
            "answer": "yes",
            "evidence_pages": "[0]",
            "evidence_sources": "[]",
            "answer_format": "short",
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA), data_root / "train-00000-of-00001.parquet")

    with pytest.raises(ValueError, match="requires eval_root"):
        write_mmlongbench_doc_train_root(
            input_root=raw_root,
            output_root=tmp_path / "converted",
            eval_root=None,
            render_page=lambda _path, _page: _jpeg_bytes(),
        )
