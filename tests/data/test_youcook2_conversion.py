from __future__ import annotations

import shutil
import tarfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets import youcook2_train as youcook2_runtime
from vlm2emb.data.conversion.youcook2_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_youcook2_indices,
    load_youcook2_split_rows,
    normalize_youcook2_video_path,
    write_youcook2_root,
)
from vlm2emb.data.datasets.youcook2_train import Youcook2TrainDataset


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


def _make_rgb_frame_bytes(num_frames: int, *, height: int = 4, width: int = 5) -> bytes:
    frames = []
    for index in range(num_frames):
        frame = np.full((height, width, 3), index * 20, dtype=np.uint8)
        frame[:, :, 1] = 255 - index * 10
        frames.append(frame)
    return np.stack(frames, axis=0).tobytes()


def _add_tar_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, BytesIO(payload))


def _write_youcook2_fixture(root: Path) -> tuple[Path, list[Path]]:
    parquet_root = root / "parquet"
    data_root = parquet_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "video_path": "./youcook2/data//raw_videos/training/106/XsALTvYUTI8.mkv",
            "caption": "pour water half and half and macaroni to the pan",
            "recipe_type": 106,
            "id": 0,
            "binary_frames": _make_rgb_frame_bytes(9),
            "num_frames": 9,
            "height": 4,
            "width": 5,
            "channels": 3,
        },
        {
            "video_path": "./youcook2/data//raw_videos/training/103/zaJrdbEGBj0.mp4",
            "caption": "slice the bread into small pieces",
            "recipe_type": 103,
            "id": 1,
            "binary_frames": _make_rgb_frame_bytes(4),
            "num_frames": 4,
            "height": 4,
            "width": 5,
            "channels": 3,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), data_root / "train-00000-of-00001.parquet")

    video_bytes = _make_mp4_bytes()
    archive = root / "raw_videos.partaa"
    with tarfile.open(archive, "w") as tar:
        _add_tar_member(tar, "raw_videos/training/106/XsALTvYUTI8.mkv", video_bytes)
        _add_tar_member(tar, "raw_videos/training/103/zaJrdbEGBj0.mp4", video_bytes)
    return parquet_root, [archive]


def test_normalize_youcook2_video_path_matches_archive_parser():
    assert (
        normalize_youcook2_video_path("./youcook2/data//raw_videos/training/106/XsALTvYUTI8.mkv")
        == "raw_videos/training/106/XsALTvYUTI8.mkv"
    )


def test_load_youcook2_split_rows_preserves_train_view_without_binary_frames(tmp_path: Path):
    parquet_root, _ = _write_youcook2_fixture(tmp_path)

    rows_by_split = load_youcook2_split_rows(parquet_root)

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_train"][0] == {
        "video_path": "./youcook2/data//raw_videos/training/106/XsALTvYUTI8.mkv",
        "caption": "pour water half and half and macaroni to the pan",
        "recipe_type": 106,
        "id": 0,
        "num_frames": 9,
        "height": 4,
        "width": 5,
        "channels": 3,
        "frame_unit_key": "raw_videos/training/106/XsALTvYUTI8.mkv#row=0",
    }


def test_collect_video_refs_deduplicates_normalized_video_paths(tmp_path: Path):
    parquet_root, _ = _write_youcook2_fixture(tmp_path)

    video_refs, split_counts = collect_video_refs(parquet_root)

    assert list(video_refs.items()) == [
        ("raw_videos/training/106/XsALTvYUTI8.mkv", "XsALTvYUTI8"),
        ("raw_videos/training/103/zaJrdbEGBj0.mp4", "zaJrdbEGBj0"),
    ]
    assert split_counts == {
        "official_train": 2,
        "official_train_without_mmeb_v2_eval": 2,
    }


def test_write_youcook2_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    parquet_root, archives = _write_youcook2_fixture(tmp_path / "source")
    output_root = tmp_path / "YouCook2"

    payload = write_youcook2_root(
        parquet_root=parquet_root,
        video_archives=archives,
        output_root=output_root,
        video_batch_size=1,
    )

    assert payload["video_rows"] == 2
    assert payload["frame_unit_rows"] == 2
    assert payload["splits"]["official_train_without_mmeb_v2_eval"] == 2
    assert payload["exclusion_rows"] == 0
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 2
    assert frame_units.count_rows() == 2
    frame_unit_row = frame_units.to_table().to_pylist()[0]
    assert frame_unit_row["frame_unit_type"] == "row_segment"
    assert len(frame_unit_row["frames"]) == 8
    assert frame_unit_row["source_total_num_frames"] == 9
    assert frame_unit_row["sampled_indices"] == [0, 1, 2, 3, 4, 5, 6, 8]
    assert train.schema.names == [
        "video_path",
        "caption",
        "recipe_type",
        "id",
        "num_frames",
        "height",
        "width",
        "channels",
        "frame_unit_key",
    ]
    assert exclusions.count_rows() == 0
    assert bad_media.count_rows() == 0
    assert any("video_path" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = Youcook2TrainDataset(path=str(output_root), num_frames=2)
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
            "vlm2emb.data.datasets.youcook2_train.sample_preextracted_frame_unit",
            wraps=youcook2_runtime.sample_preextracted_frame_unit,
        ) as sample_frame_unit,
    ):
        sample = dataset[0]

    assert sample_frame_unit.call_count == 1
    assert sample["query"]["text"] == (
        "Find a video that demonstrates the following action while making a recipe: "
        "pour water half and half and macaroni to the pan\n"
    )
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["query"]["media"] == []
    assert sample["negative"]["media"] == []
    assert sample["metadata"]["dataset_name"] == "YouCook2"
    assert sample["metadata"]["normalized_video_path"] == "raw_videos/training/106/XsALTvYUTI8.mkv"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "row_segment"
    assert sample["positive"]["media"][0]["metadata"]["total_num_frames"] == 9
    assert sample["positive"]["media"][0]["metadata"]["sampled_indices"] == [0, 8]


def test_write_youcook2_root_excludes_invalid_row_frames(tmp_path: Path):
    parquet_root, archives = _write_youcook2_fixture(tmp_path / "source")
    parquet_file = parquet_root / "data" / "train-00000-of-00001.parquet"
    rows = pq.read_table(parquet_file).to_pylist()
    rows[0]["binary_frames"] = b"invalid"
    pq.write_table(pa.Table.from_pylist([rows[0]]), parquet_file)
    output_root = tmp_path / "YouCook2-bad"

    payload = write_youcook2_root(
        parquet_root=parquet_root,
        video_archives=archives,
        output_root=output_root,
        video_batch_size=1,
    )

    assert payload["frame_unit_rows"] == 0
    assert payload["bad_media_rows"] == 1
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert train.count_rows() == 0
    assert bad_media.count_rows() == 1


def test_ensure_youcook2_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_youcook2_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_youcook2_indices should fail when frames.lance is missing")


def test_ensure_youcook2_indices_accepts_frames_only_root(tmp_path: Path):
    parquet_root, archives = _write_youcook2_fixture(tmp_path / "source")
    output_root = tmp_path / "YouCook2"
    write_youcook2_root(
        parquet_root=parquet_root,
        video_archives=archives,
        output_root=output_root,
        video_batch_size=1,
    )
    shutil.rmtree(output_root / "data" / "videos.lance")

    ensure_youcook2_indices(output_root)

    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())
