from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import cv2
import lance
import numpy as np

from vlm2emb.data.conversion.ucf101_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_ucf101_indices,
    load_ucf101_split_rows,
    write_ucf101_root,
)
from vlm2emb.data.datasets.ucf101_train import Ucf101TrainDataset, _label_text
from vlm2emb.data.utils.video import (
    _source_video_suffix,
    probe_video_bytes,
    sample_video_bytes,
    sample_video_bytes_segment,
)

_REAL_CV2_VIDEO_CAPTURE = cv2.VideoCapture


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


def _write_ucf101_fixture(raw_root: Path, vlm2vec_root: Path) -> None:
    split_root = raw_root / "splits" / "ucfTrainTestlist"
    split_root.mkdir(parents=True, exist_ok=True)
    videos_root = raw_root / "videos" / "UCF-101"
    for class_name in ("ApplyEyeMakeup", "Knitting"):
        (videos_root / class_name).mkdir(parents=True, exist_ok=True)

    (split_root / "classInd.txt").write_text("1 ApplyEyeMakeup\n2 Knitting\n", encoding="utf-8")
    (split_root / "trainlist01.txt").write_text(
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1\nKnitting/v_Knitting_g02_c02.avi 2\n",
        encoding="utf-8",
    )
    (split_root / "testlist01.txt").write_text(
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi\n",
        encoding="utf-8",
    )
    (split_root / "trainlist02.txt").write_text(
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1\nApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1\n",
        encoding="utf-8",
    )
    (split_root / "testlist02.txt").write_text(
        "Knitting/v_Knitting_g02_c02.avi\n",
        encoding="utf-8",
    )
    (split_root / "trainlist03.txt").write_text(
        "Knitting/v_Knitting_g02_c02.avi 2\nApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1\n",
        encoding="utf-8",
    )
    (split_root / "testlist03.txt").write_text(
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi\n",
        encoding="utf-8",
    )

    (vlm2vec_root / "data").mkdir(parents=True, exist_ok=True)
    (vlm2vec_root / "data" / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "video_id": "v_ApplyEyeMakeup_g08_c01",
                        "pos_text": "ApplyEyeMakeup",
                        "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi",
                    }
                ),
                json.dumps(
                    {
                        "video_id": "v_ApplyEyeMakeup_g01_c01",
                        "pos_text": "ApplyEyeMakeup",
                        "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
                    }
                ),
                json.dumps(
                    {
                        "video_id": "v_Knitting_g02_c02",
                        "pos_text": "Knitting",
                        "video_path": "Knitting/v_Knitting_g02_c02.avi",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (vlm2vec_root / "data" / "test.jsonl").write_text(
        json.dumps(
            {
                "video_id": "v_Knitting_g02_c02",
                "pos_text": "Knitting",
                "video_path": "Knitting/v_Knitting_g02_c02.avi",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    video_bytes = _make_mp4_bytes()
    (videos_root / "ApplyEyeMakeup" / "v_ApplyEyeMakeup_g08_c01.avi").write_bytes(video_bytes)
    (videos_root / "ApplyEyeMakeup" / "v_ApplyEyeMakeup_g01_c01.avi").write_bytes(video_bytes)
    (videos_root / "Knitting" / "v_Knitting_g02_c02.avi").write_bytes(video_bytes)


def test_load_ucf101_split_rows_preserves_official_and_vlm2vec_fields(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_ucf101_fixture(raw_root, vlm2vec_root)

    split_rows = load_ucf101_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_split1_train"][0] == {
        "video_id": "v_ApplyEyeMakeup_g08_c01",
        "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi",
        "label": "ApplyEyeMakeup",
        "class_index": 1,
        "split_file": "trainlist01.txt",
    }
    assert rows_by_split["official_split1_test"][0] == {
        "video_id": "v_ApplyEyeMakeup_g01_c01",
        "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
        "label": "ApplyEyeMakeup",
        "class_index": None,
        "split_file": "testlist01.txt",
    }
    assert rows_by_split["vlm2vec_train"][0] == {
        "video_id": "v_ApplyEyeMakeup_g08_c01",
        "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi",
        "pos_text": "ApplyEyeMakeup",
    }


def test_ucf101_label_text_prefers_canonical_label_when_present():
    assert _label_text({"label": "CanonicalAction", "pos_text": "LegacyAction"}) == "CanonicalAction"


def test_collect_video_refs_deduplicates_all_split_videos(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_ucf101_fixture(raw_root, vlm2vec_root)

    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)

    assert list(video_refs) == [
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi",
        "Knitting/v_Knitting_g02_c02.avi",
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
    ]
    assert split_counts == {
        "official_split1_train": 2,
        "official_split1_test": 1,
        "official_split2_train": 2,
        "official_split2_test": 1,
        "official_split3_train": 2,
        "official_split3_test": 1,
        "vlm2vec_train": 3,
        "vlm2vec_test_1k": 1,
    }


def test_write_ucf101_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    output_root = tmp_path / "UCF101"
    _write_ucf101_fixture(raw_root, vlm2vec_root)

    payload = write_ucf101_root(
        raw_root=raw_root,
        vlm2vec_root=vlm2vec_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["splits"]["vlm2vec_train"] == 3
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "vlm2vec_train.lance"))
    official = lance.dataset(str(output_root / "data" / "official_split1_train.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 3
    assert train.schema.names == ["video_id", "video_path", "pos_text"]
    assert official.schema.names == ["video_id", "video_path", "label", "class_index", "split_file"]
    assert bad_media.count_rows() == 0
    assert any("video_path" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = Ucf101TrainDataset(path=str(output_root), split="vlm2vec_train", num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|video_pad|>\n"
        "What activities or sports are being performed by the person in the video?\n"
    )
    assert sample["positive"]["text"] == "ApplyEyeMakeup"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "UCF101"
    assert sample["metadata"]["video_path"] == "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi"
    assert sample["query"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["query"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["query"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_ucf101_runtime_reads_existing_legacy_layout(tmp_path: Path):
    output_root = tmp_path / "UCF101"
    (output_root / "data" / "splits").mkdir(parents=True)
    video_bytes = _make_mp4_bytes()

    lance.write_dataset(
        [
            {
                "sample_id": "sample_a",
                "membership_index": 0,
            }
        ],
        str(output_root / "data" / "splits" / "vlm2vec_train.lance"),
        mode="create",
    )
    lance.write_dataset(
        [
            {
                "sample_id": "sample_a",
                "asset_id": "asset_a",
                "video_id": "v_ApplyEyeMakeup_g08_c01",
                "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi",
                "label_text": "ApplyEyeMakeup",
                "negative_text": "",
                "class_name": "ApplyEyeMakeup",
            }
        ],
        str(output_root / "data" / "samples.lance"),
        mode="create",
    )
    lance.write_dataset(
        [
            {
                "asset_id": "asset_a",
                "video_id": "v_ApplyEyeMakeup_g08_c01",
                "video_path": "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi",
                "video_bytes": video_bytes,
                "fps": 3.0,
                "total_num_frames": 3,
            }
        ],
        str(output_root / "data" / "media.lance"),
        mode="create",
    )
    ensure_ucf101_indices(output_root)

    try:
        Ucf101TrainDataset(path=str(output_root), split="vlm2vec_train", num_frames=2)
    except FileNotFoundError as error:
        assert "allow_raw_video=True" in str(error)
    else:
        raise AssertionError("Legacy raw-video layout should require explicit opt-in")

    dataset = Ucf101TrainDataset(
        path=str(output_root),
        split="vlm2vec_train",
        num_frames=2,
        allow_raw_video=True,
    )
    sample = dataset[0]

    assert dataset.storage_layout == "legacy"
    assert sample["positive"]["text"] == "ApplyEyeMakeup"
    assert sample["negative"]["text"] == ""
    assert len(sample["query"]["media"][0]["content"]) == 2




def test_video_utils_fallback_to_cv2_when_pyav_probe_fails():
    video_bytes = _make_mp4_bytes()
    with mock.patch("vlm2emb.data.utils.video.av.open", side_effect=UnicodeDecodeError("utf-8", b"\xd3", 0, 1, "bad")):
        metadata = probe_video_bytes(video_bytes)
        images, sampled = sample_video_bytes(video_bytes, num_frames=2)

    assert metadata["fps"] == 3.0
    assert metadata["total_num_frames"] == 3
    assert len(images) == 2
    assert sampled["fps"] == 3.0
    assert sampled["total_num_frames"] == 3
    assert sampled["sampled_indices"] == [0, 2]


def test_video_utils_uses_sparse_path_for_large_video_bytes():
    video_bytes = _make_mp4_bytes()
    with mock.patch("vlm2emb.data.utils.video.SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES", 1):
        images, sampled = sample_video_bytes(video_bytes, num_frames=2)

    assert len(images) == 2
    assert sampled == {
        "fps": 3.0,
        "total_num_frames": 3,
        "sampled_indices": [0, 2],
    }


def test_video_utils_prefers_in_memory_sparse_path_by_default():
    video_bytes = _make_mp4_bytes()
    with mock.patch("vlm2emb.data.utils.video._with_temp_video_file") as with_temp_video_file:
        images, sampled = sample_video_bytes(video_bytes, num_frames=2)

    with_temp_video_file.assert_not_called()
    assert len(images) == 2
    assert sampled == {
        "fps": 3.0,
        "total_num_frames": 3,
        "sampled_indices": [0, 2],
    }


def test_video_utils_preserves_source_video_suffix_for_temp_file():
    assert _source_video_suffix("raw/path/video.webm") == ".webm"
    assert _source_video_suffix("raw/path/video.MKV") == ".mkv"
    assert _source_video_suffix("raw/path/video") == ".mp4"


def test_video_utils_sparse_segment_metadata_matches_decoded_path():
    video_bytes = _make_mp4_bytes()
    with mock.patch("vlm2emb.data.utils.video.SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES", 1):
        images, sampled = sample_video_bytes_segment(
            video_bytes,
            num_frames=2,
            start_time=0.0,
            end_time=0.7,
        )

    assert len(images) == 2
    assert sampled == {
        "fps": 3.0,
        "total_num_frames": 3,
        "sampled_indices": [0, 2],
        "source_total_num_frames": 3,
        "source_sampled_indices": [0, 2],
        "segment_start_frame": 0,
        "segment_end_frame_exclusive": 3,
    }


class _UnknownFrameCountCapture:
    """Wrap OpenCV capture while hiding frame-count metadata."""

    def __init__(self, path: str) -> None:
        self._capture = _REAL_CV2_VIDEO_CAPTURE(path)

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        return self._capture.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        return self._capture.set(prop_id, value)

    def read(self):
        return self._capture.read()

    def release(self) -> None:
        self._capture.release()


def test_video_utils_sparse_cv2_falls_back_to_sequential_decode_when_frame_count_is_unknown():
    video_bytes = _make_mp4_bytes()
    with (
        mock.patch("vlm2emb.data.utils.video.av.open", side_effect=UnicodeDecodeError("utf-8", b"\xd3", 0, 1, "bad")),
        mock.patch("vlm2emb.data.utils.video.SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES", 1),
        mock.patch("vlm2emb.data.utils.video.cv2.VideoCapture", _UnknownFrameCountCapture),
    ):
        images, sampled = sample_video_bytes(video_bytes, num_frames=2)

    assert len(images) == 2
    assert sampled == {
        "fps": 3.0,
        "total_num_frames": 3,
        "sampled_indices": [0, 2],
    }


def test_video_utils_sparse_cv2_segment_falls_back_to_sequential_decode_when_frame_count_is_unknown():
    video_bytes = _make_mp4_bytes()
    with (
        mock.patch("vlm2emb.data.utils.video.av.open", side_effect=UnicodeDecodeError("utf-8", b"\xd3", 0, 1, "bad")),
        mock.patch("vlm2emb.data.utils.video.SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES", 1),
        mock.patch("vlm2emb.data.utils.video.cv2.VideoCapture", _UnknownFrameCountCapture),
    ):
        images, sampled = sample_video_bytes_segment(
            video_bytes,
            num_frames=2,
            start_time=0.0,
            end_time=0.7,
        )

    assert len(images) == 2
    assert sampled == {
        "fps": 3.0,
        "total_num_frames": 3,
        "sampled_indices": [0, 2],
        "source_total_num_frames": 3,
        "source_sampled_indices": [0, 2],
        "segment_start_frame": 0,
        "segment_end_frame_exclusive": 3,
    }
