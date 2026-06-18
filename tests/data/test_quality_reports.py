from __future__ import annotations

import json

import pyarrow as pa
import lance

from scripts.convert.clean_dataset import main as clean_dataset_main
from scripts.convert.validate_dataset import main as validate_dataset_main
from vlm2emb.data.quality.cleaning import classify_validation_result, serialize_validation_result
from vlm2emb.data.quality.reporting import build_quality_audit, render_quality_report

from tests.data.fixtures.quality_artifacts import make_retrieval_artifacts, make_validation_result


def test_quality_audit_builder_summarizes_findings_and_decisions():
    result = make_validation_result()
    decisions = classify_validation_result(result)
    audit = build_quality_audit(result, decisions=decisions)

    assert audit["dataset_name"] == "demo"
    assert audit["summary"]["finding_count"] == 3
    assert audit["summary"]["decision_counts"]["drop"] == 1
    assert audit["summary"]["decision_counts"]["quarantine"] == 2
    assert not audit["ready"]


def test_quality_report_renders_reason_counts_and_representative_findings():
    result = make_validation_result()
    report = render_quality_report(build_quality_audit(result, decisions=classify_validation_result(result)))
    assert "Quality Report: demo" in report
    assert "`empty_text`" in report
    assert "Representative Findings" in report


def test_validate_dataset_cli_writes_audit_and_report(tmp_path, monkeypatch):
    dataset_path = tmp_path / "ImageNet-1K"
    dataset_path.mkdir()
    queries, candidates, qrels = make_retrieval_artifacts()
    lance.write_dataset(pa.Table.from_pylist(queries), str(dataset_path / "queries.lance"), mode="create")
    lance.write_dataset(pa.Table.from_pylist(candidates), str(dataset_path / "candidates.lance"), mode="create")
    lance.write_dataset(pa.Table.from_pylist(qrels), str(dataset_path / "qrels.lance"), mode="create")
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ImageNet-1K", "task_type": "image_classification"}, file_handle)
    with (dataset_path / "manifest.json").open("w") as file_handle:
        json.dump({"dataset_id": "dataset_ImageNet-1K"}, file_handle)

    monkeypatch.setattr(
        "sys.argv",
        ["validate_dataset.py", "--path", str(dataset_path), "--output-dir", str(dataset_path)],
    )
    assert validate_dataset_main() == 0
    assert (dataset_path / "quality.audit.json").exists()
    assert (dataset_path / "quality.report.md").exists()
    assert (dataset_path / "quality.validation.json").exists()


def test_clean_dataset_cli_writes_audit_and_report(tmp_path, monkeypatch):
    validation_path = tmp_path / "validation.json"
    output_dir = tmp_path / "cleaning"
    with validation_path.open("w") as file_handle:
        json.dump(serialize_validation_result(make_validation_result()), file_handle, ensure_ascii=False)

    monkeypatch.setattr(
        "sys.argv",
        ["clean_dataset.py", "--findings", str(validation_path), "--output-dir", str(output_dir)],
    )
    assert clean_dataset_main() == 0
    assert (output_dir / "quality.audit.json").exists()
    assert (output_dir / "quality.report.md").exists()


def test_validate_dataset_cli_supports_document_train_lance_root(tmp_path, monkeypatch):
    dataset_path = tmp_path / "vidore-rag"
    (dataset_path / "data").mkdir(parents=True)
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                {
                    "query": "find the matching page",
                    "answer": "answer",
                    "source": "docvqa",
                    "image": {
                        "bytes": (
                            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
                            b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
                        ),
                        "path": "page.png",
                    },
                }
            ]
        ),
        str(dataset_path / "data" / "train.lance"),
        mode="create",
    )
    with (dataset_path / "manifest.json").open("w") as file_handle:
        json.dump(
            {
                "dataset_name": "vidore-rag",
                "task_type": "document_retrieval",
                "metadata": {"loader_type": "vidore_rag_train"},
            },
            file_handle,
        )

    monkeypatch.setattr(
        "sys.argv",
        ["validate_dataset.py", "--path", str(dataset_path), "--output-dir", str(dataset_path), "--media-sample-limit", "4"],
    )
    assert validate_dataset_main() == 0
    assert (dataset_path / "quality.audit.json").exists()
    assert (dataset_path / "quality.report.md").exists()
    assert (dataset_path / "quality.validation.json").exists()
