from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np

from vlm2emb.data.conversion.kinetics700_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    load_kinetics700_split_rows,
    load_mmeb_v2_eval_exclusion_rows,
    write_kinetics700_root,
    write_kinetics700_root_sharded,
)
from vlm2emb.data.datasets.kinetics700_train import Kinetics700TrainDataset


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
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, BytesIO(payload))


def _write_kinetics700_fixture(raw_root: Path, vlm2vec_root: Path) -> None:
    (raw_root / "annotations").mkdir(parents=True, exist_ok=True)
    (raw_root / "train_tars").mkdir(parents=True, exist_ok=True)
    (raw_root / "annotations" / "train.csv").write_text(
        "\n".join(
            [
                "label,youtube_id,time_start,time_end,split",
                "abseiling,videoa,0,10,train",
                "lifting hat,videob,11,21,train",
                "zumba,videoc,22,32,train",
                "zumba,videoc,22,32,train",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    video_bytes = _make_mp4_bytes()
    with tarfile.open(raw_root / "train_tars" / "k700_train_001.tar.gz", mode="w:gz") as tar:
        _add_tar_member(tar, "abseiling/videoa_000000_000010.mp4", video_bytes)
        _add_tar_member(tar, "lifting hat/videob_000011_000021.mp4", video_bytes)
    with tarfile.open(raw_root / "train_tars" / "k700_train_002.tar.gz", mode="w:gz") as tar:
        _add_tar_member(tar, "abseiling/videoa_000000_000010.mp4", video_bytes)
        _add_tar_member(tar, "zumba/videoc_000022_000032.mp4", video_bytes)

    (vlm2vec_root / "data").mkdir(parents=True, exist_ok=True)
    (vlm2vec_root / "data" / "test.jsonl").write_text(
        json.dumps(
            {
                "video_id": "videob_000011_000021",
                "pos_text": "lifting hat",
                "video_path": "lifting hat/videob_000011_000021.mp4",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_kinetics700_split_rows_deduplicates_and_excludes_mmeb_v2_eval(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_kinetics700_fixture(raw_root, vlm2vec_root)

    split_rows = load_kinetics700_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}
    exclusion_rows = load_mmeb_v2_eval_exclusion_rows(vlm2vec_root)

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert len(rows_by_split["official_train"]) == 3
    assert len(rows_by_split["train_without_mmeb_v2_eval"]) == 2
    assert exclusion_rows == [
        {
            "video_id": "videob_000011_000021",
            "video_path": "lifting hat/videob_000011_000021.mp4",
            "pos_text": "lifting hat",
        }
    ]
    assert rows_by_split["official_train"][0] == {
        "video_id": "videoa_000000_000010",
        "video_path": "abseiling/videoa_000000_000010.mp4",
        "label": "abseiling",
        "youtube_id": "videoa",
        "time_start": 0,
        "time_end": 10,
        "split": "train",
    }
    assert [row["label"] for row in rows_by_split["train_without_mmeb_v2_eval"]] == ["abseiling", "zumba"]


def test_collect_kinetics700_video_refs_uses_official_train_rows(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_kinetics700_fixture(raw_root, vlm2vec_root)

    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)

    assert list(video_refs) == [
        "abseiling/videoa_000000_000010.mp4",
        "lifting hat/videob_000011_000021.mp4",
        "zumba/videoc_000022_000032.mp4",
    ]
    assert split_counts == {
        "official_train": 3,
        "train_without_mmeb_v2_eval": 2,
    }


def test_write_kinetics700_root_streams_tar_videos_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    output_root = tmp_path / "Kinetics-700"
    _write_kinetics700_fixture(raw_root, vlm2vec_root)

    payload = write_kinetics700_root(
        raw_root=raw_root,
        vlm2vec_root=vlm2vec_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=1,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["excluded_rows"] == 1
    assert payload["bad_media_rows"] == 0
    assert payload["splits"]["train_without_mmeb_v2_eval"] == 2
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
    assert train.count_rows() == 2
    assert exclusions.count_rows() == 1
    assert bad_media.count_rows() == 0
    assert train.schema.names == ["video_id", "video_path", "label", "youtube_id", "time_start", "time_end", "split"]
    assert any("video_path" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = Kinetics700TrainDataset(path=str(output_root), split="train_without_mmeb_v2_eval", num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == "<|video_pad|>\nRecognize the category of the video content.\n"
    assert sample["positive"]["text"] == "abseiling"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "Kinetics-700"
    assert sample["metadata"]["video_path"] == "abseiling/videoa_000000_000010.mp4"
    assert sample["query"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["query"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["query"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_kinetics700_runtime_supports_config_sample_cap(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    output_root = tmp_path / "Kinetics-700"
    _write_kinetics700_fixture(raw_root, vlm2vec_root)
    write_kinetics700_root(raw_root=raw_root, vlm2vec_root=vlm2vec_root, output_root=output_root)

    dataset = Kinetics700TrainDataset(
        path=str(output_root),
        split="train_without_mmeb_v2_eval",
        num_frames=1,
        max_samples=1,
        sample_seed=42,
    )

    assert len(dataset) == 1
    assert dataset[0]["positive"]["text"] in {"abseiling", "zumba"}


def test_write_kinetics700_root_sharded_merges_tar_shards(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    output_root = tmp_path / "Kinetics-700"
    _write_kinetics700_fixture(raw_root, vlm2vec_root)

    payload = write_kinetics700_root_sharded(
        raw_root=raw_root,
        vlm2vec_root=vlm2vec_root,
        output_root=output_root,
        shard_workers=2,
        threads_per_shard=1,
        video_batch_size=1,
        merge_batch_size=1,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["splits"]["train_without_mmeb_v2_eval"] == 2
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "_shards").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 3

    dataset = Kinetics700TrainDataset(path=str(output_root), split="train_without_mmeb_v2_eval", num_frames=2)
    sample = dataset[0]
    assert sample["query"]["media"][0]["kind"] == "video"
