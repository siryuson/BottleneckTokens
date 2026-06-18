from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import lance
import numpy as np
import pytest

from vlm2emb.data.datasets import charades_sta_train as charades_runtime
from vlm2emb.data.conversion.charades_sta_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_charades_sta_indices,
    load_charades_sta_split_rows,
    write_charades_sta_root,
)
from vlm2emb.data.datasets.charades_sta_train import CharadesStaTrainDataset


def _make_mp4_bytes() -> bytes:
    buffer = BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 8
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)):
            array = np.full((8, 8, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _write_fixture(raw_root: Path) -> None:
    annotation_root = raw_root / "annotations" / "charades_sta"
    video_root = raw_root / "videos"
    annotation_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    (annotation_root / "Charades_sta_train.txt").write_text(
        "\n".join(
            [
                "AO8RW 0.0 0.7##a person is putting a book on a shelf.",
                "Y6R7T 0.0 0.6##person begins to play on a phone.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (annotation_root / "Charades_sta_test.txt").write_text(
        "3MSZA 0.0 0.5##person turn a light on.\n",
        encoding="utf-8",
    )
    video_bytes = _make_mp4_bytes()
    with zipfile.ZipFile(video_root / "Charades_v1.zip", mode="w") as archive:
        for video_id in ("AO8RW", "Y6R7T", "3MSZA"):
            archive.writestr(f"Charades_v1/{video_id}.mp4", video_bytes)


def test_load_charades_sta_split_rows_preserves_annotation_fields(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root)

    split_rows = load_charades_sta_split_rows(raw_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_train"][0] == {
        "id": "AO8RW",
        "start_time": 0.0,
        "end_time": 0.7,
        "duration": 0.7,
        "description": "a person is putting a book on a shelf.",
        "query_frame_unit_key": "AO8RW#full",
        "positive_frame_unit_key": "AO8RW#segment=0.000-0.700",
    }


def test_collect_video_refs_deduplicates_all_splits(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root)

    video_refs, split_counts = collect_video_refs(raw_root)

    assert list(video_refs.items()) == [
        ("AO8RW", "Charades_v1/AO8RW.mp4"),
        ("Y6R7T", "Charades_v1/Y6R7T.mp4"),
        ("3MSZA", "Charades_v1/3MSZA.mp4"),
    ]
    assert split_counts == {
        "official_train": 2,
        "official_test": 1,
        "official_train_without_mmeb_v2_eval": 2,
    }


def test_write_charades_sta_root_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "Charades-STA"
    _write_fixture(raw_root)

    payload = write_charades_sta_root(
        raw_root=raw_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 6
    assert payload["splits"]["official_train_without_mmeb_v2_eval"] == 2
    assert payload["exclusion_rows"] == 1
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 6
    assert train.schema.names == [
        "id",
        "start_time",
        "end_time",
        "duration",
        "description",
        "query_frame_unit_key",
        "positive_frame_unit_key",
    ]
    assert exclusions.count_rows() == 1
    assert bad_media.count_rows() == 0
    assert any("video_id" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = CharadesStaTrainDataset(path=str(output_root), num_frames=2)
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
            "vlm2emb.data.datasets.charades_sta_train.sample_prefixed_preextracted_frame_unit",
            wraps=charades_runtime.sample_prefixed_preextracted_frame_unit,
        ) as sample_frame_unit,
    ):
        sample = dataset[0]

    assert sample_frame_unit.call_count == 2
    assert sample["query"]["text"] == (
        "<|video_pad|>\n"
        "Find the clip that corresponds to the described scene in the given video: "
        "a person is putting a book on a shelf.\n"
    )
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "Charades-STA"
    assert sample["metadata"]["id"] == "AO8RW"
    assert sample["query"]["media"][0]["kind"] == "video"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["clip_start"] == 0.0
    assert sample["positive"]["media"][0]["metadata"]["clip_end"] == pytest.approx(0.7)
    assert sample["positive"]["media"][0]["metadata"]["clip_total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["source_total_num_frames"] == 4
    assert sample["positive"]["media"][0]["metadata"]["sampled_indices"] == [0, 2]
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "segment"


def test_ensure_charades_sta_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_charades_sta_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_charades_sta_indices should fail when frames.lance is missing")
