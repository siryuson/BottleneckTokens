from __future__ import annotations

import json

from scripts.convert.clean_dataset import main as clean_dataset_main
from vlm2emb.data.quality.cleaning import (
    ACTION_BY_DECISION_HINT,
    bucket_decisions,
    classify_validation_result,
    serialize_validation_result,
)

from tests.data.fixtures.quality_artifacts import make_validation_result


def test_decision_engine_is_deterministic_for_same_findings():
    result = make_validation_result()
    assert classify_validation_result(result) == classify_validation_result(result)


def test_block_findings_are_not_downgraded_to_keep():
    assert ACTION_BY_DECISION_HINT["block"] == "quarantine"


def test_quarantine_is_distinct_from_drop():
    decisions = classify_validation_result(make_validation_result())
    decision_by_target = {decision.target_id: decision for decision in decisions}
    assert decision_by_target["sample_query_1"].action == "drop"
    assert decision_by_target["asset_candidate_1"].action == "quarantine"
    assert decision_by_target["relation_qrels_1"].action == "quarantine"


def test_bucketed_outputs_keep_quarantine_separate_from_drop():
    buckets = bucket_decisions(classify_validation_result(make_validation_result()))
    assert buckets["drop"]["sample"] == ["sample_query_1"]
    assert buckets["quarantine"]["asset"] == ["asset_candidate_1"]
    assert buckets["quarantine"]["relation"] == ["relation_qrels_1"]
    assert buckets["drop"]["asset"] == []


def test_clean_dataset_cli_writes_decision_ledger_and_buckets(tmp_path, monkeypatch):
    validation_path = tmp_path / "validation.json"
    output_dir = tmp_path / "cleaning"
    with validation_path.open("w") as file_handle:
        json.dump(serialize_validation_result(make_validation_result()), file_handle, ensure_ascii=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "clean_dataset.py",
            "--findings",
            str(validation_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert clean_dataset_main() == 0
    assert (output_dir / "quality.decisions.jsonl").exists()
    assert (output_dir / "quality.keep.json").exists()
    assert (output_dir / "quality.drop.json").exists()
    assert (output_dir / "quality.quarantine.json").exists()

    with (output_dir / "quality.drop.json").open() as file_handle:
        drop_payload = json.load(file_handle)
    with (output_dir / "quality.quarantine.json").open() as file_handle:
        quarantine_payload = json.load(file_handle)
    decision_lines = (output_dir / "quality.decisions.jsonl").read_text().strip().splitlines()

    assert drop_payload["sample"] == ["sample_query_1"]
    assert quarantine_payload["asset"] == ["asset_candidate_1"]
    assert quarantine_payload["relation"] == ["relation_qrels_1"]
    assert len(decision_lines) == 3
