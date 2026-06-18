from __future__ import annotations

from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.conversion.breakfast_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_breakfast_indices,
    load_breakfast_split_rows,
    write_breakfast_root,
)
from vlm2emb.data.datasets.breakfast_train import BreakfastTrainDataset


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


def _write_breakfast_split(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.table({
        "video_path": [str(row["video_path"]) for row in rows],
        "participant": [str(row["participant"]) for row in rows],
        "camera": [str(row["camera"]) for row in rows],
        "video": [str(row["video"]) for row in rows],
        "labels": [row["labels"] for row in rows],
    })
    pq.write_table(table, path)


def _write_breakfast_fixture(raw_root: Path) -> None:
    (raw_root / "data").mkdir(parents=True, exist_ok=True)
    video_bytes = _make_mp4_bytes()

    splits = {
        "s1": [
            {
                "video_path": "videos/P03/cam01/P03_cereals.avi",
                "participant": "P03",
                "camera": "cam01",
                "video": "P03_cereals",
                "labels": [{"start": 1, "end": 10, "label": "SIL"}],
            },
            {
                "video_path": "videos/P03/webcam01/P03_coffee.avi",
                "participant": "P03",
                "camera": "webcam01",
                "video": "P03_coffee",
                "labels": [{"start": 1, "end": 10, "label": "SIL"}],
            },
        ],
        "s2": [
            {
                "video_path": "videos/P16/stereo/P16_friedegg.avi",
                "participant": "P16",
                "camera": "stereo",
                "video": "P16_friedegg",
                "labels": [{"start": 1, "end": 10, "label": "SIL"}],
            }
        ],
        "s3": [
            {
                "video_path": "videos/P29/cam01/P29_tea.avi",
                "participant": "P29",
                "camera": "cam01",
                "video": "P29_tea",
                "labels": [{"start": 1, "end": 10, "label": "SIL"}],
            }
        ],
        "s4": [
            {
                "video_path": "videos/P42/webcam02/P42_milk.avi",
                "participant": "P42",
                "camera": "webcam02",
                "video": "P42_milk",
                "labels": [{"start": 1, "end": 10, "label": "SIL"}],
            }
        ],
    }

    for split_name, rows in splits.items():
        _write_breakfast_split(raw_root / "data" / f"{split_name}-00000-of-00001.parquet", rows)
        for row in rows:
            video_path = raw_root / str(row["video_path"])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(video_bytes)


def _write_breakfast_vlm2vec_fixture(vlm2vec_root: Path) -> None:
    (vlm2vec_root / "data").mkdir(parents=True, exist_ok=True)
    (vlm2vec_root / "data" / "test.jsonl").write_text(
        '{"pos_text":"cereal","video_id":"P03_cereal","video_path":"P03/cam01/P03_cereal.avi"}\n'
        '{"pos_text":"tea","video_id":"P29_tea","video_path":"P29/cam01/P29_tea.avi"}\n',
        encoding="utf-8",
    )


def test_load_breakfast_split_rows_collects_official_and_archive_views(tmp_path: Path):
    raw_root = tmp_path / "Breakfast-Actions"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_breakfast_fixture(raw_root)
    _write_breakfast_vlm2vec_fixture(vlm2vec_root)

    split_rows = load_breakfast_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)

    assert tuple(split_rows) == SUPPORTED_SPLITS
    assert len(split_rows["official_s1"]) == 2
    assert len(split_rows["official_s2"]) == 1
    assert len(split_rows["archive_train_non_cam01"]) == 3
    assert len(split_rows["vlm2vec_test_cam01_433"]) == 2
    assert split_rows["archive_train_non_cam01"][0]["label"] == "coffee"
    assert split_rows["vlm2vec_test_cam01_433"][0]["label"] == "cereal"
    assert split_rows["vlm2vec_test_cam01_433"][0]["raw_label"] == "cereals"


def test_collect_breakfast_video_refs_deduplicates_all_split_videos(tmp_path: Path):
    raw_root = tmp_path / "Breakfast-Actions"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_breakfast_fixture(raw_root)
    _write_breakfast_vlm2vec_fixture(vlm2vec_root)

    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)

    assert list(video_refs) == [
        "videos/P03/cam01/P03_cereals.avi",
        "videos/P03/webcam01/P03_coffee.avi",
        "videos/P16/stereo/P16_friedegg.avi",
        "videos/P29/cam01/P29_tea.avi",
        "videos/P42/webcam02/P42_milk.avi",
    ]
    assert split_counts["archive_train_non_cam01"] == 3
    assert split_counts["vlm2vec_test_cam01_433"] == 2


def test_write_breakfast_root_writes_shared_tables_and_loader_reads_archive_split(tmp_path: Path):
    raw_root = tmp_path / "Breakfast-Actions"
    vlm2vec_root = tmp_path / "vlm2vec"
    output_root = tmp_path / "Breakfast"
    _write_breakfast_fixture(raw_root)
    _write_breakfast_vlm2vec_fixture(vlm2vec_root)

    payload = write_breakfast_root(raw_root=raw_root, vlm2vec_root=vlm2vec_root, output_root=output_root)

    assert payload["video_rows"] == 5
    assert payload["frame_unit_rows"] == 5
    assert payload["splits"]["archive_train_non_cam01"] == 3
    assert payload["splits"]["vlm2vec_test_cam01_433"] == 2
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "archive_train_non_cam01.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 5
    assert frame_units.count_rows() == 5
    assert train.count_rows() == 3
    assert exclusions.count_rows() == 2
    assert bad_media.count_rows() == 0
    assert train.schema.names == [
        "video_id",
        "video_path",
        "label",
        "raw_label",
        "participant",
        "camera",
        "source_split",
        "labels",
    ]
    assert any("video_path" in idx.field_names for idx in videos.describe_indices())
    assert any("frame_unit_key" in idx.field_names for idx in frame_units.describe_indices())

    dataset = BreakfastTrainDataset(
        path=str(output_root),
        dataset_name="Breakfast",
        split="archive_train_non_cam01",
        num_frames=2,
    )
    sample = dataset[0]

    assert sample["metadata"]["dataset_name"] == "Breakfast"
    assert sample["query"]["text"] == "<|video_pad|>\nRecognize the breakfast type that the person is cooking in the video.\n"
    assert sample["positive"]["text"] == "coffee"
    assert sample["negative"]["text"] == ""
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["query"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["query"]["media"][0]["metadata"]["sampled_indices"] == [0, 2]
    assert sample["query"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"

    eval_dataset = BreakfastTrainDataset(
        path=str(output_root),
        dataset_name="Breakfast",
        split="vlm2vec_test_cam01_433",
        num_frames=2,
    )
    eval_sample = eval_dataset[0]
    assert eval_sample["metadata"]["label"] == "cereal"
    assert eval_sample["metadata"]["raw_label"] == "cereals"
    assert eval_sample["positive"]["text"] == "cereal"


def test_ensure_breakfast_indices_adds_missing_persistent_indices(tmp_path: Path):
    output_root = tmp_path / "Breakfast"
    (output_root / "data").mkdir(parents=True, exist_ok=True)

    frame_rows = [
        {
            "frame_unit_key": "videos/P03/cam01/P03_cereals.avi",
            "frame_unit_type": "full_video",
            "source_key": "P03_cereals",
            "source_path": "videos/P03/cam01/P03_cereals.avi",
            "frames": [],
            "sampled_frame_count": 0,
            "source_total_num_frames": 3,
            "clip_total_num_frames": 3,
            "fps": 3.0,
            "duration": 1.0,
            "clip_start": 0.0,
            "clip_end": 0.0,
            "sampled_indices": [],
            "sampled_timestamps": [],
            "decode_backend": "test",
            "warning_count": 0,
        }
    ]

    lance.write_dataset(pa.Table.from_pylist(frame_rows), str(output_root / "data" / "frames.lance"), mode="create")

    assert lance.dataset(str(output_root / "data" / "frames.lance")).describe_indices() == []

    ensure_breakfast_indices(output_root)

    assert any(
        "frame_unit_key" in idx.field_names
        for idx in lance.dataset(str(output_root / "data" / "frames.lance")).describe_indices()
    )
