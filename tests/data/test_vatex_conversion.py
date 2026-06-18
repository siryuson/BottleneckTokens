from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np

from vlm2emb.data.conversion.vatex_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_vatex_indices,
    load_vatex_split_rows,
    normalize_vatex_video_member,
    write_vatex_root,
)
from vlm2emb.data.datasets.vatex_train import VatexTrainDataset


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


def _add_tar_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, BytesIO(payload))


def _write_vatex_fixture(root: Path) -> tuple[Path, Path]:
    source_root = root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    annotation_path = source_root / "vatex_training_v1.0.json"
    rows = [
        {
            "videoID": "abc123_000001_000011",
            "enCap": ["a person opens a door", "someone enters a room"],
            "chCap": ["一个人打开门", "有人进入房间"],
        },
        {
            "videoID": "xyz789_000002_000012",
            "enCap": ["a dog runs outside"],
            "chCap": ["一只狗在外面跑"],
        },
    ]
    annotation_path.write_text(json.dumps(rows), encoding="utf-8")

    video_bytes = _make_mp4_bytes()
    archive_path = source_root / "vatex-dataset.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        _add_tar_member(tar, "./vatex-dataset/abc123_000001_000011.mp4", video_bytes)
        _add_tar_member(tar, "vatex-dataset/xyz789_000002_000012.mp4", video_bytes)
    return annotation_path, archive_path


def test_normalize_vatex_video_member_matches_official_tar_layout():
    assert normalize_vatex_video_member("./vatex-dataset/abc123_000001_000011.mp4") == (
        "vatex-dataset/abc123_000001_000011.mp4"
    )
    assert normalize_vatex_video_member("abc123_000001_000011") == "vatex-dataset/abc123_000001_000011.mp4"


def test_load_vatex_split_rows_preserves_official_training_view(tmp_path: Path):
    annotation_path, _ = _write_vatex_fixture(tmp_path)

    rows_by_split = load_vatex_split_rows(annotation_path)

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_train"][0] == {
        "video_id": "abc123_000001_000011",
        "video": "vatex-dataset/abc123_000001_000011.mp4",
        "en_caption": ["a person opens a door", "someone enters a room"],
        "ch_caption": ["一个人打开门", "有人进入房间"],
    }


def test_collect_video_refs_deduplicates_video_ids(tmp_path: Path):
    annotation_path, _ = _write_vatex_fixture(tmp_path)

    video_refs, split_counts = collect_video_refs(annotation_path)

    assert list(video_refs.items()) == [
        ("abc123_000001_000011", "vatex-dataset/abc123_000001_000011.mp4"),
        ("xyz789_000002_000012", "vatex-dataset/xyz789_000002_000012.mp4"),
    ]
    assert split_counts == {
        "official_train": 2,
        "official_train_without_mmeb_v2_eval": 2,
    }


def test_write_vatex_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    annotation_path, archive_path = _write_vatex_fixture(tmp_path)
    output_root = tmp_path / "VATEX"

    payload = write_vatex_root(
        annotation_path=annotation_path,
        video_archive=archive_path,
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
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 2
    assert frame_units.count_rows() == 2
    assert train.schema.names == ["video_id", "video", "en_caption", "ch_caption"]
    assert exclusions.count_rows() == 0
    assert bad_media.count_rows() == 0
    assert any("video_id" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = VatexTrainDataset(path=str(output_root), num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == "Find a video that matches the following caption: a person opens a door\n"
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["query"]["media"] == []
    assert sample["negative"]["media"] == []
    assert sample["metadata"]["dataset_name"] == "VATEX"
    assert sample["metadata"]["video_id"] == "abc123_000001_000011"
    assert sample["metadata"]["en_caption"] == ["a person opens a door", "someone enters a room"]
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["positive"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_write_vatex_root_can_skip_videos_table_and_parallelize_frame_units(tmp_path: Path):
    annotation_path, archive_path = _write_vatex_fixture(tmp_path)
    output_root = tmp_path / "VATEX-fast"

    payload = write_vatex_root(
        annotation_path=annotation_path,
        video_archive=archive_path,
        output_root=output_root,
        video_batch_size=1,
        frame_unit_workers=2,
        write_videos_table=False,
    )

    assert payload["video_rows"] == 2
    assert payload["frame_unit_rows"] == 2
    assert payload["videos_table_written"] is False
    assert payload["frame_unit_workers"] == 2
    assert not (output_root / "data" / "videos.lance").exists()
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    assert frame_units.count_rows() == 2
    assert train.count_rows() == 2

    dataset = VatexTrainDataset(path=str(output_root), num_frames=2)
    sample = dataset[0]

    assert sample["metadata"]["dataset_name"] == "VATEX"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2


def test_write_vatex_root_records_and_excludes_bad_media(tmp_path: Path):
    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    broken_annotation = broken_root / "vatex_training_v1.0.json"
    broken_annotation.write_text(
        json.dumps(
            [
                {
                    "videoID": "bad_000001_000011",
                    "enCap": ["this video cannot be decoded"],
                    "chCap": ["坏视频"],
                }
            ]
        ),
        encoding="utf-8",
    )
    broken_archive = broken_root / "vatex-dataset.tar"
    with tarfile.open(broken_archive, "w") as tar:
        _add_tar_member(tar, "vatex-dataset/bad_000001_000011.mp4", b"not a video")

    output_root = tmp_path / "VATEX-bad"
    payload = write_vatex_root(
        annotation_path=broken_annotation,
        video_archive=broken_archive,
        output_root=output_root,
        video_batch_size=1,
    )

    assert payload["video_rows"] == 0
    assert payload["frame_unit_rows"] == 0
    assert payload["bad_media_rows"] == 1
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert train.count_rows() == 0
    assert bad_media.count_rows() == 1


def test_ensure_vatex_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_vatex_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_vatex_indices should fail when frames.lance is missing")
