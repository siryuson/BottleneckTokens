from __future__ import annotations

import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import lance
import numpy as np

from vlm2emb.data.datasets import msrvtt_train as msrvtt_runtime
from vlm2emb.data.conversion.msrvtt_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_msrvtt_indices,
    load_msrvtt_split_rows,
    write_msrvtt_root,
)
from vlm2emb.data.datasets.msrvtt_train import MsrvttTrainDataset, build_msrvtt_query_text


def _make_mp4_bytes() -> bytes:
    buffer = BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
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


def _train_row(video_id: str, video: str, index: int, captions: list[str]) -> dict[str, object]:
    return {
        "video_id": video_id,
        "video": video,
        "caption": captions,
        "source": "MSR-VTT",
        "category": 9,
        "url": "https://example.com",
        "start time": 1.0,
        "end time": 2.0,
        "id": index,
    }


def _test_row(video_id: str, video: str, index: int, caption: str) -> dict[str, object]:
    return {
        "video_id": video_id,
        "video": video,
        "caption": caption,
        "source": "MSR-VTT",
        "category": 10,
        "url": "https://example.com/test",
        "start time": 3.0,
        "end time": 4.0,
        "id": index,
    }


def _write_msrvtt_fixture(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "raw_videos").mkdir()
    video_bytes = _make_mp4_bytes()
    for video in ("video0.mp4", "video1.mp4", "video2.mp4"):
        (source_root / "raw_videos" / video).write_bytes(video_bytes)

    (source_root / "msrvtt_train_9k.json").write_text(
        json.dumps(
            [
                _train_row("video0", "video0.mp4", 0, ["a car is shown", "a person drives"]),
                _train_row("video1", "video1.mp4", 1, ["a dog runs", "a dog is outside"]),
            ]
        ),
        encoding="utf-8",
    )
    (source_root / "msrvtt_train_7k.json").write_text(
        json.dumps([_train_row("video0", "video0.mp4", 0, ["a car is shown"])]),
        encoding="utf-8",
    )
    (source_root / "msrvtt_test_1k.json").write_text(
        json.dumps([_test_row("video2", "video2.mp4", 2, "a test caption")]),
        encoding="utf-8",
    )


def test_load_msrvtt_split_rows_preserves_source_fields(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_msrvtt_fixture(source_root)

    split_rows = load_msrvtt_split_rows(source_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["train_9k"][0] == {
        "video_id": "video0",
        "video": "video0.mp4",
        "caption": ["a car is shown", "a person drives"],
        "source": "MSR-VTT",
        "category": 9,
        "url": "https://example.com",
        "start time": 1.0,
        "end time": 2.0,
        "id": 0,
    }
    assert rows_by_split["test_1k"][0]["caption"] == "a test caption"


def test_collect_video_refs_deduplicates_all_splits(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_msrvtt_fixture(source_root)

    video_refs, split_counts = collect_video_refs(source_root)

    assert list(video_refs) == ["video0.mp4", "video1.mp4", "video2.mp4"]
    assert split_counts == {
        "train_9k": 2,
        "train_7k": 1,
        "test_1k": 1,
        "train_9k_without_mmeb_v2_eval": 2,
    }


def test_write_msrvtt_root_reads_official_zip_without_unpacked_videos(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "MSR-VTT"
    _write_msrvtt_fixture(source_root)
    with zipfile.ZipFile(source_root / "MSRVTT_Videos.zip", mode="w") as archive:
        for raw_video in sorted((source_root / "raw_videos").glob("*.mp4")):
            archive.write(raw_video, arcname=f"video/{raw_video.name}")
    shutil.rmtree(source_root / "raw_videos")

    payload = write_msrvtt_root(
        source_root=source_root,
        output_root=output_root,
        num_workers=1,
        video_batch_size=2,
    )

    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    assert payload["frame_unit_rows"] == 3
    assert frame_units.count_rows() == 3


def test_write_msrvtt_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "MSR-VTT"
    _write_msrvtt_fixture(source_root)

    payload = write_msrvtt_root(
        source_root=source_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["splits"]["train_9k_without_mmeb_v2_eval"] == 2
    assert payload["exclusion_rows"] == 1
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "train_9k_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 3
    assert frame_units.to_table().to_pylist()[0]["frame_unit_type"] == "full_video"
    assert train.schema.names == [
        "video_id",
        "video",
        "caption",
        "source",
        "category",
        "url",
        "start time",
        "end time",
        "id",
    ]
    assert exclusions.count_rows() == 1
    assert bad_media.count_rows() == 0
    assert any("video" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = MsrvttTrainDataset(path=str(output_root), num_frames=2)
    with (
        mock.patch(
            "vlm2emb.data.utils.video.sample_video_bytes",
            side_effect=AssertionError("runtime must not raw-decode source videos"),
        ),
        mock.patch(
            "vlm2emb.data.utils.video.sample_video_bytes_segment",
            side_effect=AssertionError("runtime must not raw-decode source video segments"),
        ),
        mock.patch(
            "vlm2emb.data.datasets.msrvtt_train.sample_preextracted_frame_unit",
            wraps=msrvtt_runtime.sample_preextracted_frame_unit,
        ) as sample_frame_unit,
    ):
        sample = dataset[0]

    assert sample_frame_unit.call_count == 1
    assert sample["query"]["text"] == "Find a video that contains the following visual content: a car is shown\n"
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "MSR-VTT"
    assert sample["metadata"]["video"] == "video0.mp4"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"
    assert sample["positive"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["positive"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["sampled_indices"] == [0, 2]


def test_build_msrvtt_query_text_exposes_caption_selection():
    text = build_msrvtt_query_text(["first caption", "second caption"], caption_selection="first")

    assert text == "Find a video that contains the following visual content: first caption\n"

    try:
        build_msrvtt_query_text(["first caption"], caption_selection="latest")
    except ValueError as error:
        assert "caption_selection" in str(error)
    else:
        raise AssertionError("Expected invalid MSR-VTT caption_selection to fail")


def test_ensure_msrvtt_indices_requires_video_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_msrvtt_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_msrvtt_indices should fail when videos.lance is missing")
