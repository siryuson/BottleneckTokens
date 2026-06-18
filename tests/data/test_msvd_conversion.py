from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np

from vlm2emb.data.conversion.msvd_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_msvd_indices,
    load_msvd_split_rows,
    write_msvd_root,
)
from vlm2emb.data.datasets.msvd_train import MsvdTrainDataset, build_msvd_query_text


def _make_avi_bytes() -> bytes:
    buffer = BytesIO()
    with av.open(buffer, mode="w", format="avi") as container:
        stream = container.add_stream("mpeg4", rate=3)
        stream.width = 8
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
            array = np.full((8, 8, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _row(video_id: str, video: str, captions: list[str]) -> dict[str, object]:
    return {
        "video_id": video_id,
        "video": video,
        "caption": captions,
        "source": "MSVD",
    }


def _write_msvd_fixture(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "raw_videos").mkdir()
    video_bytes = _make_avi_bytes()
    for video in ("train.avi", "val.avi", "test.avi"):
        (source_root / "raw_videos" / video).write_bytes(video_bytes)

    (source_root / "msvd_train.json").write_text(
        json.dumps([_row("train", "train.avi", ["a woman breaks an egg", "a person cooks"])]),
        encoding="utf-8",
    )
    (source_root / "msvd_val.json").write_text(
        json.dumps([_row("val", "val.avi", ["a lion jumps"])]),
        encoding="utf-8",
    )
    (source_root / "msvd_test.json").write_text(
        json.dumps([_row("test", "test.avi", ["two men play ping pong"])]),
        encoding="utf-8",
    )


def test_load_msvd_split_rows_preserves_source_fields(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_msvd_fixture(source_root)

    split_rows = load_msvd_split_rows(source_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["train"][0] == {
        "video_id": "train",
        "video": "train.avi",
        "caption": ["a woman breaks an egg", "a person cooks"],
        "source": "MSVD",
    }


def test_collect_video_refs_deduplicates_all_splits(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_msvd_fixture(source_root)

    video_refs, split_counts = collect_video_refs(source_root)

    assert list(video_refs) == ["train.avi", "val.avi", "test.avi"]
    assert split_counts == {
        "train": 1,
        "val": 1,
        "test": 1,
        "train_without_mmeb_v2_eval": 1,
    }


def test_write_msvd_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "MSVD"
    _write_msvd_fixture(source_root)

    payload = write_msvd_root(
        source_root=source_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["splits"]["train_without_mmeb_v2_eval"] == 1
    assert payload["exclusion_rows"] == 1
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "train_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 3
    assert train.schema.names == ["video_id", "video", "caption", "source"]
    assert exclusions.count_rows() == 1
    assert bad_media.count_rows() == 0
    assert any("video" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = MsvdTrainDataset(path=str(output_root), num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "Find the video snippet that corresponds to the given summary: a woman breaks an egg\n"
    )
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "MSVD"
    assert sample["metadata"]["video"] == "train.avi"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["positive"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_build_msvd_query_text_exposes_caption_selection():
    text = build_msvd_query_text(["first caption", "second caption"], caption_selection="first")

    assert text == "Find the video snippet that corresponds to the given summary: first caption\n"

    try:
        build_msvd_query_text(["first caption"], caption_selection="latest")
    except ValueError as error:
        assert "caption_selection" in str(error)
    else:
        raise AssertionError("Expected invalid MSVD caption_selection to fail")


def test_ensure_msvd_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_msvd_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_msvd_indices should fail when frames.lance is missing")
