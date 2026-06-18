from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vlm2emb.data.conversion.visrag_train import VISRAG_SUBSET_SOURCES
from vlm2emb.data.conversion.visrag_train import convert_visrag_root
from vlm2emb.data.conversion.vidore_train import VIDORE_RAG_SUBSET_SOURCES
from vlm2emb.data.conversion.vidore_train import convert_vidore_root


def _write_raw_root(raw_root: Path, rows: list[dict]) -> None:
    (raw_root / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), raw_root / "data" / "train-00000-of-00001.parquet")
    (raw_root / "README.md").write_text("# raw\n", encoding="utf-8")
    (raw_root / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")


def test_convert_vidore_family_preserves_raw_fields_and_writes_subset_roots(tmp_path: Path):
    raw_root = tmp_path / "raw_vidore"
    converted_root = tmp_path / "vidore_colpali_train_set"
    output_base = tmp_path / "vlm2emb"
    image = {"bytes": b"img", "path": "page.jpg"}
    rows = [
        {
            "query": f"query-{source}",
            "answer": f"answer-{source}",
            "prompt": "",
            "source": source,
            "image": image,
            "image_filename": f"{source}.jpg",
            "options": "",
            "page": "1",
            "model": "test",
            "answer_type": "text",
        }
        for source in VIDORE_RAG_SUBSET_SOURCES
    ]
    _write_raw_root(raw_root, rows)

    summary = convert_vidore_root(
        raw_root,
        converted_root,
        rag_output_base=output_base,
    )

    assert summary.full.rows == len(VIDORE_RAG_SUBSET_SOURCES)
    assert not (converted_root / "manifest.json").exists()
    assert not (converted_root / "README.md").exists()
    assert not (converted_root / "dataset_infos.json").exists()

    full_dataset = lance.dataset(str(converted_root / "data" / "train.lance"))
    assert full_dataset.count_rows() == len(VIDORE_RAG_SUBSET_SOURCES)
    assert {
        "query",
        "answer",
        "prompt",
        "source",
        "image",
        "image_filename",
        "options",
        "page",
        "model",
        "answer_type",
    }.issubset(set(full_dataset.schema.names))

    subset_root = output_base / "vidore_rag_docvqa"
    assert subset_root.exists()
    assert not (subset_root / "manifest.json").exists()
    assert not (subset_root / "README.md").exists()
    assert not (subset_root / "dataset_infos.json").exists()

    subset_dataset = lance.dataset(str(subset_root / "data" / "train.lance"))
    assert subset_dataset.count_rows() == 1
    row = subset_dataset.to_table().to_pylist()[0]
    assert row["source"] == "docvqa"
    assert row["query"] == "query-docvqa"
    assert row["image"]["path"] == "page.jpg"


def test_convert_visrag_family_preserves_raw_fields_and_writes_subset_roots(tmp_path: Path):
    raw_root = tmp_path / "raw_visrag"
    converted_root = tmp_path / "openbmb_VisRAG-Ret-Train-In-domain-data"
    output_base = tmp_path / "vlm2emb"
    image = {"bytes": b"img", "path": "page.jpg"}
    rows = [
        {
            "query": f"query-{source}",
            "source": source,
            "image": image,
        }
        for source in VISRAG_SUBSET_SOURCES
    ]
    _write_raw_root(raw_root, rows)

    summary = convert_visrag_root(
        raw_root,
        converted_root,
        source_output_base=output_base,
    )

    assert summary.full.rows == len(VISRAG_SUBSET_SOURCES)
    assert not (converted_root / "manifest.json").exists()
    assert not (converted_root / "README.md").exists()
    assert not (converted_root / "dataset_infos.json").exists()

    full_dataset = lance.dataset(str(converted_root / "data" / "train.lance"))
    assert full_dataset.count_rows() == len(VISRAG_SUBSET_SOURCES)
    assert {"query", "source", "image"}.issubset(set(full_dataset.schema.names))

    subset_root = output_base / "visrag_indomain_InfoVQA"
    assert subset_root.exists()
    assert not (subset_root / "manifest.json").exists()
    subset_dataset = lance.dataset(str(subset_root / "data" / "train.lance"))
    assert subset_dataset.count_rows() == 1
    row = subset_dataset.to_table().to_pylist()[0]
    assert row["source"] == "InfoVQA"
    assert row["query"] == "query-InfoVQA"


def test_document_retrieval_conversion_rejects_missing_requested_source(tmp_path: Path):
    raw_root = tmp_path / "raw_vidore"
    converted_root = tmp_path / "vidore_colpali_train_set"
    output_base = tmp_path / "vlm2emb"
    _write_raw_root(
        raw_root,
        [
            {
                "query": "query-docvqa",
                "answer": "answer-docvqa",
                "source": "docvqa",
                "image": {"bytes": b"img", "path": "page.jpg"},
            }
        ],
    )

    with pytest.raises(ValueError, match="No rows were written"):
        convert_vidore_root(
            raw_root,
            converted_root,
            rag_output_base=output_base,
            rag_sources=("pdf",),
        )
