from __future__ import annotations

from pathlib import Path

import pytest

import vlm2emb.cli.convert_mmeb_v2 as cli
from vlm2emb.data.conversion.mmeb_v2.bindings import resolve_source_path
from vlm2emb.data.conversion.mmeb_v2.common import get_metadata_task_type
from vlm2emb.data.conversion.mmeb_v2.datasets import SourceSpec
from vlm2emb.data.conversion.mmeb_v2.datasets import get_pipeline
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import BENCHMARK_INDEX_ROWS


def test_parse_source_overrides_supports_repeatable_key_value_pairs():
    overrides = cli.parse_source_overrides(
        [
            "hf:siyrus/BToks-ViDoSeek-page-fixed=/data/vidoseek-fixed",
            "hf:siyrus/BToks-MSR-VTT::test_1k/test=/data/msrvtt",
        ]
    )

    assert overrides == {
        "hf:siyrus/BToks-ViDoSeek-page-fixed": Path("/data/vidoseek-fixed"),
        "hf:siyrus/BToks-MSR-VTT::test_1k/test": Path("/data/msrvtt"),
    }


def test_parse_datasets_csv_dedupes_while_preserving_order():
    datasets = cli.parse_datasets_arg("MSR-VTT,ImageNet-1K,MSR-VTT,ViDoSeek-page")

    assert datasets == ["MSR-VTT", "ImageNet-1K", "ViDoSeek-page"]


def test_resolve_source_path_prefers_explicit_source_override():
    resolved = resolve_source_path(
        SourceSpec(
            role="annotation",
            kind="local_dir",
            root="mmeb_v2",
            ref="visdoc-tasks/data/ViDoSeek",
        ),
        source_overrides={
            "mmeb_v2:visdoc-tasks/data/ViDoSeek": Path("/custom/vidoseek"),
        },
    )

    assert resolved == Path("/custom/vidoseek")


def test_convert_selected_datasets_routes_roots_and_overrides(monkeypatch, tmp_path):
    captured: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _capture_image_cls(*args, **kwargs):
        captured.append(("image_cls", args, kwargs))
        return Path("/tmp/image")

    def _capture_video_ret(*args, **kwargs):
        captured.append(("video_ret", args, kwargs))
        return Path("/tmp/video")

    def _capture_visdoc(*args, **kwargs):
        captured.append(("visdoc_beir", args, kwargs))
        return Path("/tmp/visdoc")

    monkeypatch.setattr(cli, "convert_image_cls_dataset", _capture_image_cls)
    monkeypatch.setattr(cli, "convert_video_ret_dataset", _capture_video_ret)
    monkeypatch.setattr(cli, "convert_visdoc_dataset", _capture_visdoc)

    outputs = cli.convert_selected_datasets(
        ["ImageNet-1K", "MSR-VTT", "ViDoSeek-page"],
        output_root=tmp_path / "out",
        mmeb_v2_root=tmp_path / "mmeb-v2",
        mmeb_eval_root=tmp_path / "mmeb-eval",
        source_overrides={
            "hf:siyrus/BToks-ViDoSeek-page-fixed": tmp_path / "fixed" / "vidoseek",
        },
        max_workers=3,
    )

    assert outputs == [Path("/tmp/image"), Path("/tmp/video"), Path("/tmp/visdoc")]
    assert captured[0] == (
        "image_cls",
        ("ImageNet-1K", tmp_path / "out"),
        {
            "mmeb_eval_root": tmp_path / "mmeb-eval",
            "mmeb_v2_root": tmp_path / "mmeb-v2",
            "source_overrides": {
                "hf:siyrus/BToks-ViDoSeek-page-fixed": tmp_path / "fixed" / "vidoseek",
            },
            "max_workers": 3,
        },
    )
    assert captured[1] == (
        "video_ret",
        ("MSR-VTT", tmp_path / "out"),
        {
            "mmeb_v2_root": tmp_path / "mmeb-v2",
            "source_overrides": {
                "hf:siyrus/BToks-ViDoSeek-page-fixed": tmp_path / "fixed" / "vidoseek",
            },
            "max_workers": 3,
        },
    )
    assert captured[2] == (
        "visdoc_beir",
        ("ViDoSeek-page", tmp_path / "out"),
        {
            "mmeb_v2_root": tmp_path / "mmeb-v2",
            "source_overrides": {
                "hf:siyrus/BToks-ViDoSeek-page-fixed": tmp_path / "fixed" / "vidoseek",
            },
        },
    )


def test_main_requires_output_root_when_not_listing():
    with pytest.raises(SystemExit):
        cli.main(["--datasets", "ImageNet-1K"])


def test_main_requires_dataset_selection_when_not_listing(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--output-root", "/tmp/out"])

    captured = capsys.readouterr()
    assert "one of --datasets or --all is required" in captured.err


def test_conversion_metadata_task_type_matches_benchmark_index() -> None:
    for dataset_name, _, _, task_type in BENCHMARK_INDEX_ROWS:
        assert get_metadata_task_type(dataset_name, get_pipeline(dataset_name)) == task_type
