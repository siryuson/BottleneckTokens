from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import lance
import numpy as np

from vlm2emb.data.datasets import didemo_train as didemo_runtime
from vlm2emb.data.conversion.didemo_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_didemo_indices,
    load_didemo_split_rows,
    write_didemo_root,
)
from vlm2emb.data.datasets.didemo_train import DidemoTrainDataset


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


def _row(video: str, caption: str) -> dict[str, object]:
    return {
        "video": video,
        "caption": caption,
        "source": "DiDeMo",
    }


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return buffer.getvalue()


def _write_didemo_fixture(source_root: Path) -> tuple[list[Path], Path]:
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "didemo_train.json").write_text(
        json.dumps(
            [
                _row("train/train_a.mp4", "camera zooms in on baby"),
                _row("train/train_b.mp4", "a person walks across the room"),
            ]
        ),
        encoding="utf-8",
    )
    (source_root / "didemo_test.json").write_text(
        json.dumps([_row("test/test_a.mp4", "we pan right off the mural")]),
        encoding="utf-8",
    )
    video_bytes = _make_mp4_bytes()
    train_tar = _tar_bytes(
        {
            "video/train/train_a.mp4": video_bytes,
            "video/train/train_b.mp4": video_bytes,
        }
    )
    midpoint = len(train_tar) // 2
    train_part_0 = source_root / "train.tar.part-000"
    train_part_1 = source_root / "train.tar.part-001"
    train_part_0.write_bytes(train_tar[:midpoint])
    train_part_1.write_bytes(train_tar[midpoint:])
    test_archive = source_root / "test.tar"
    test_archive.write_bytes(_tar_bytes({"video/test/test_a.mp4": video_bytes}))
    return [train_part_0, train_part_1], test_archive


def test_load_didemo_split_rows_preserves_source_fields(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_didemo_fixture(source_root)

    split_rows = load_didemo_split_rows(source_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["train"][0] == {
        "video": "train/train_a.mp4",
        "caption": "camera zooms in on baby",
        "source": "DiDeMo",
    }


def test_collect_video_refs_deduplicates_all_splits(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_didemo_fixture(source_root)

    video_refs, split_counts = collect_video_refs(source_root)

    assert list(video_refs) == ["train/train_a.mp4", "train/train_b.mp4", "test/test_a.mp4"]
    assert split_counts == {
        "train": 2,
        "test": 1,
        "train_without_mmeb_v2_eval": 2,
    }


def test_write_didemo_root_streams_split_tar_and_runtime_sample(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "DiDeMo"
    train_archives, test_archive = _write_didemo_fixture(source_root)

    payload = write_didemo_root(
        source_root=source_root,
        train_archives=train_archives,
        test_archive=test_archive,
        output_root=output_root,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["splits"]["train_without_mmeb_v2_eval"] == 2
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
    assert frame_units.to_table().to_pylist()[0]["frame_unit_type"] == "full_video"
    assert train.schema.names == ["video", "caption", "source"]
    assert exclusions.count_rows() == 1
    assert bad_media.count_rows() == 0
    assert any("video" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = DidemoTrainDataset(path=str(output_root), num_frames=2)
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
            "vlm2emb.data.datasets.didemo_train.sample_preextracted_frame_unit",
            wraps=didemo_runtime.sample_preextracted_frame_unit,
        ) as sample_frame_unit,
    ):
        sample = dataset[0]

    assert sample_frame_unit.call_count == 1
    assert sample["query"]["text"] == (
        "Find a video that includes the following described scenes: camera zooms in on baby\n"
    )
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "DiDeMo"
    assert sample["metadata"]["video"] == "train/train_a.mp4"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"
    assert sample["positive"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["positive"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["sampled_indices"] == [0, 2]


def test_ensure_didemo_indices_requires_video_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_didemo_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_didemo_indices should fail when videos.lance is missing")
