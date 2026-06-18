"""Tests for scripts/eval.py helpers."""

from __future__ import annotations

from pathlib import Path


def test_resolve_output_dir_prefers_local_checkpoint(tmp_path):
    from vlm2emb.evaluation.evaluate import _resolve_output_dir

    output_dir = _resolve_output_dir(str(tmp_path), None)
    assert output_dir.parent == tmp_path / "eval"


def test_resolve_output_dir_uses_default_eval_dir_for_nonlocal_checkpoint():
    from vlm2emb.evaluation.evaluate import _resolve_output_dir

    output_dir = _resolve_output_dir("org/model", None)
    assert output_dir.parent == Path("./eval")


def test_resolve_output_dir_honors_explicit_output_dir(tmp_path):
    from vlm2emb.evaluation.evaluate import _resolve_output_dir

    explicit = tmp_path / "custom"
    output_dir = _resolve_output_dir("org/model", str(explicit))
    assert output_dir == explicit


def test_prepare_log_file_creates_parent_directory(tmp_path):
    from scripts.eval import _prepare_log_file

    log_path = tmp_path / "nested" / "eval.log"

    resolved = _prepare_log_file(str(log_path))

    assert resolved == log_path
    assert log_path.parent.exists()
