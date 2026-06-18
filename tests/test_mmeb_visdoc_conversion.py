from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vlm2emb.data.conversion.mmeb_v2.visdoc import convert_visdoc_dataset


def _write_beir_table(dataset_dir: Path, table_name: str, table: pa.Table) -> None:
    table_dir = dataset_dir / table_name
    table_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, table_dir / "test-00000-of-00001.parquet")


def test_convert_visdoc_dataset_writes_beir_rerun_artifact(tmp_path: Path):
    dataset_dir = tmp_path / "ViDoRe_biomedical_lectures_v2_multilingual"

    queries = pa.table(
        {
            "query-id": [0, 1],
            "query": ["What does the chart show?", "What is the title?"],
            "language": ["en", "en"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": [27, 28, 29],
            "image": [
                {"bytes": b"img-27", "path": "27.png"},
                {"bytes": b"img-28", "path": None},
                {"bytes": b"img-29", "path": "29.png"},
            ],
            "doc-id": ["doc-a", "doc-a", "doc-b"],
        }
    )
    qrels = pa.table(
        {
            "query-id": [0, 0, 1],
            "corpus-id": [27, 29, 28],
            "is-answerable": ["fully answerable", "partially answerable", "fully answerable"],
            "answer": ["answer-27", "answer-29", "answer-28"],
            "score": [2, 1, 2],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    output_root = tmp_path / "rerun"
    output_path = convert_visdoc_dataset(
        "ViDoRe_biomedical_lectures_v2_multilingual",
        output_root,
        data_dir=dataset_dir,
    )

    assert output_path == output_root / "visdoc-tasks" / "ViDoRe_biomedical_lectures_v2_multilingual"
    assert (output_path / "queries.lance").exists()
    assert (output_path / "candidates.lance").exists()
    assert (output_path / "qrels.lance").exists()
    assert (output_path / "metadata.json").exists()
    assert not (output_path / "manifest.json").exists()

    metadata = json.loads((output_path / "metadata.json").read_text())
    assert metadata == {
        "name": "ViDoRe_biomedical_lectures_v2_multilingual",
        "task_type": "visdoc_vidore_v2",
        "modality": "visdoc",
        "benchmark": "MMEB-V2",
    }

    query_rows = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidate_rows = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrel_rows = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert query_rows == [
        {
            "id": 0,
            "query": "What does the chart show?",
            "language": "en",
            "is-answerable": ["fully answerable", "partially answerable"],
            "answer": ["answer-27", "answer-29"],
        },
        {
            "id": 1,
            "query": "What is the title?",
            "language": "en",
            "is-answerable": ["fully answerable"],
            "answer": ["answer-28"],
        },
    ]

    assert candidate_rows == [
        {"id": 27, "image": b"img-27", "path": "27.png", "doc-id": "doc-a"},
        {"id": 28, "image": b"img-28", "path": None, "doc-id": "doc-a"},
        {"id": 29, "image": b"img-29", "path": "29.png", "doc-id": "doc-b"},
    ]

    assert qrel_rows == [
        {
            "query_id": 0,
            "mode": "sparse",
            "candidate_ids": [27, 29],
            "candidate_scores": [2.0, 1.0],
        },
        {
            "query_id": 1,
            "mode": "sparse",
            "candidate_ids": [28],
            "candidate_scores": [2.0],
        },
    ]


def test_convert_visdoc_dataset_preserves_string_ids_and_flattens_image_struct(tmp_path: Path):
    dataset_dir = tmp_path / "VisRAG_ChartQA"

    queries = pa.table(
        {
            "query-id": ["3960.png-2"],
            "query": ["How many more people felt inspired frequently than depressed frequently?"],
            "answer": ["0.03"],
            "options": [None],
            "is_numerical": [0],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": ["41699051005347.png"],
            "image": [{"bytes": b"chart-bytes", "path": "1.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": ["3960.png-2"],
            "corpus-id": ["41699051005347.png"],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    output_root = tmp_path / "rerun"
    output_path = convert_visdoc_dataset(
        "VisRAG_ChartQA",
        output_root,
        data_dir=dataset_dir,
    )

    query_rows = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidate_rows = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrel_rows = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert query_rows == [
        {
            "id": "3960.png-2",
            "query": "How many more people felt inspired frequently than depressed frequently?",
            "answer": "0.03",
            "options": None,
            "is_numerical": 0,
        }
    ]
    assert candidate_rows == [
        {
            "id": "41699051005347.png",
            "image": b"chart-bytes",
            "path": "1.png",
        }
    ]
    assert qrel_rows == [
        {
            "query_id": "3960.png-2",
            "mode": "sparse",
            "candidate_ids": ["41699051005347.png"],
            "candidate_scores": [1.0],
        }
    ]


def test_convert_visdoc_dataset_fails_on_orphan_query_id_in_qrels(tmp_path: Path):
    dataset_dir = tmp_path / "ViDoRe_biomedical_lectures_v2_multilingual"

    queries = pa.table(
        {
            "query-id": [0],
            "query": ["What does the chart show?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": [27],
            "image": [{"bytes": b"img-27", "path": "27.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": [999],
            "corpus-id": [27],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="unknown query-id"):
        convert_visdoc_dataset(
            "ViDoRe_biomedical_lectures_v2_multilingual",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )


def test_convert_visdoc_dataset_fails_on_unknown_corpus_id_in_qrels(tmp_path: Path):
    dataset_dir = tmp_path / "VisRAG_ChartQA"

    queries = pa.table(
        {
            "query-id": ["q0"],
            "query": ["What is the title?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": ["c0"],
            "image": [{"bytes": b"chart-bytes", "path": "1.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": ["q0"],
            "corpus-id": ["missing-corpus"],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="unknown corpus-id"):
        convert_visdoc_dataset(
            "VisRAG_ChartQA",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )


def test_convert_visdoc_dataset_fails_when_query_has_no_qrels(tmp_path: Path):
    dataset_dir = tmp_path / "ViDoRe_biomedical_lectures_v2_multilingual"

    queries = pa.table(
        {
            "query-id": [0, 1],
            "query": ["What does the chart show?", "What is the title?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": [27],
            "image": [{"bytes": b"img-27", "path": "27.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": [0],
            "corpus-id": [27],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="qrels missing entries for query-id"):
        convert_visdoc_dataset(
            "ViDoRe_biomedical_lectures_v2_multilingual",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )


def test_convert_visdoc_dataset_fails_on_duplicate_query_ids(tmp_path: Path):
    dataset_dir = tmp_path / "ViDoRe_biomedical_lectures_v2_multilingual"

    queries = pa.table(
        {
            "query-id": [0, 0],
            "query": ["What does the chart show?", "What is the title?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": [27],
            "image": [{"bytes": b"img-27", "path": "27.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": [0],
            "corpus-id": [27],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="duplicate query-id"):
        convert_visdoc_dataset(
            "ViDoRe_biomedical_lectures_v2_multilingual",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )


def test_convert_visdoc_dataset_fails_on_duplicate_corpus_ids(tmp_path: Path):
    dataset_dir = tmp_path / "VisRAG_ChartQA"

    queries = pa.table(
        {
            "query-id": ["q0"],
            "query": ["What is the title?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": ["c0", "c0"],
            "image": [
                {"bytes": b"chart-bytes-1", "path": "1.png"},
                {"bytes": b"chart-bytes-2", "path": "2.png"},
            ],
        }
    )
    qrels = pa.table(
        {
            "query-id": ["q0"],
            "corpus-id": ["c0"],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="duplicate corpus-id"):
        convert_visdoc_dataset(
            "VisRAG_ChartQA",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )


def test_convert_visdoc_dataset_fails_on_null_query_id(tmp_path: Path):
    dataset_dir = tmp_path / "ViDoRe_biomedical_lectures_v2_multilingual"

    queries = pa.table(
        {
            "query-id": [None],
            "query": ["What does the chart show?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": [27],
            "image": [{"bytes": b"img-27", "path": "27.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": [None],
            "corpus-id": [27],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="null query-id"):
        convert_visdoc_dataset(
            "ViDoRe_biomedical_lectures_v2_multilingual",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )


def test_convert_visdoc_dataset_fails_on_null_corpus_id(tmp_path: Path):
    dataset_dir = tmp_path / "VisRAG_ChartQA"

    queries = pa.table(
        {
            "query-id": ["q0"],
            "query": ["What is the title?"],
        }
    )
    corpus = pa.table(
        {
            "corpus-id": [None],
            "image": [{"bytes": b"chart-bytes", "path": "1.png"}],
        }
    )
    qrels = pa.table(
        {
            "query-id": ["q0"],
            "corpus-id": [None],
            "score": [1],
        }
    )

    _write_beir_table(dataset_dir, "queries", queries)
    _write_beir_table(dataset_dir, "corpus", corpus)
    _write_beir_table(dataset_dir, "qrels", qrels)

    with pytest.raises(ValueError, match="null corpus-id"):
        convert_visdoc_dataset(
            "VisRAG_ChartQA",
            tmp_path / "rerun",
            data_dir=dataset_dir,
        )
