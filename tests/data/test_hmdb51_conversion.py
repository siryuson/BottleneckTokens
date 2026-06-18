from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np

from vlm2emb.data.conversion.hmdb51_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_hmdb51_indices,
    load_hmdb51_split_rows,
    write_hmdb51_root,
)
from vlm2emb.data.datasets.hmdb51_train import Hmdb51TrainDataset, _label_text


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


def _write_hmdb51_fixture(raw_root: Path, vlm2vec_root: Path, video_root: Path) -> None:
    split_root = raw_root / "test_train_splits" / "testTrainMulti_7030_splits"
    split_root.mkdir(parents=True, exist_ok=True)

    # HMDB51 official split semantics: 1=train, 2=test, 0=unused for that split.
    split_payloads = {
        "brush_hair_test_split1.txt": [
            "brush_hair_a.avi 1",
            "brush_hair_b.avi 2",
        ],
        "brush_hair_test_split2.txt": [
            "brush_hair_a.avi 2",
            "brush_hair_b.avi 1",
        ],
        "brush_hair_test_split3.txt": [
            "brush_hair_a.avi 1",
            "brush_hair_b.avi 0",
        ],
        "kick_test_split1.txt": [
            "kick_a.avi 1",
            "kick_b.avi 0",
        ],
        "kick_test_split2.txt": [
            "kick_a.avi 0",
            "kick_b.avi 2",
        ],
        "kick_test_split3.txt": [
            "kick_a.avi 2",
            "kick_b.avi 1",
        ],
    }
    for name, rows in split_payloads.items():
        (split_root / name).write_text("\n".join(rows) + "\n", encoding="utf-8")

    (vlm2vec_root / "data").mkdir(parents=True, exist_ok=True)
    (vlm2vec_root / "data" / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "video_id": "brush_hair_a",
                        "pos_text": "brush_hair",
                        "video_path": "brush_hair/brush_hair_a.avi",
                    }
                ),
                json.dumps(
                    {
                        "video_id": "kick_a",
                        "pos_text": "kick",
                        "video_path": "kick/kick_a.avi",
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
                "video_id": "brush_hair_b",
                "pos_text": "brush_hair",
                "video_path": "brush_hair/brush_hair_b.avi",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    video_bytes = _make_mp4_bytes()
    for class_name, filename in [
        ("brush_hair", "brush_hair_a.avi"),
        ("brush_hair", "brush_hair_b.avi"),
        ("kick", "kick_a.avi"),
        ("kick", "kick_b.avi"),
    ]:
        target = video_root / class_name
        target.mkdir(parents=True, exist_ok=True)
        (target / filename).write_bytes(video_bytes)


def test_load_hmdb51_split_rows_preserves_official_and_vlm2vec_fields(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    video_root = tmp_path / "videos"
    _write_hmdb51_fixture(raw_root, vlm2vec_root, video_root)

    split_rows = load_hmdb51_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert len(rows_by_split["official_split1_train"]) == 2
    assert rows_by_split["official_split1_train"][0] == {
        "video_id": "brush_hair_a",
        "video_path": "brush_hair/brush_hair_a.avi",
        "label": "brush_hair",
        "split_tag": 1,
        "split_file": "brush_hair_test_split1.txt",
    }
    assert len(rows_by_split["official_split2_test"]) == 2
    assert rows_by_split["vlm2vec_train"][0] == {
        "video_id": "brush_hair_a",
        "video_path": "brush_hair/brush_hair_a.avi",
        "pos_text": "brush_hair",
    }


def test_hmdb51_label_text_prefers_canonical_label_when_present():
    assert _label_text({"label": "canonical_label", "pos_text": "legacy_positive"}) == "canonical_label"


def test_collect_video_refs_deduplicates_all_split_videos(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    video_root = tmp_path / "videos"
    _write_hmdb51_fixture(raw_root, vlm2vec_root, video_root)

    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)

    assert list(video_refs) == [
        "brush_hair/brush_hair_a.avi",
        "kick/kick_a.avi",
        "brush_hair/brush_hair_b.avi",
        "kick/kick_b.avi",
    ]
    assert split_counts == {
        "official_split1_train": 2,
        "official_split1_test": 1,
        "official_split2_train": 1,
        "official_split2_test": 2,
        "official_split3_train": 2,
        "official_split3_test": 1,
        "vlm2vec_train": 2,
        "vlm2vec_test_1k": 1,
    }


def test_write_hmdb51_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    video_root = tmp_path / "videos"
    output_root = tmp_path / "HMDB51"
    _write_hmdb51_fixture(raw_root, vlm2vec_root, video_root)

    payload = write_hmdb51_root(
        raw_root=raw_root,
        vlm2vec_root=vlm2vec_root,
        video_root=video_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 4
    assert payload["frame_unit_rows"] == 4
    assert payload["splits"]["vlm2vec_train"] == 2
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "vlm2vec_train.lance"))
    official = lance.dataset(str(output_root / "data" / "official_split1_train.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 4
    assert frame_units.count_rows() == 4
    assert train.schema.names == ["video_id", "video_path", "pos_text"]
    assert official.schema.names == ["video_id", "video_path", "label", "split_tag", "split_file"]
    assert bad_media.count_rows() == 0
    assert any("video_path" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = Hmdb51TrainDataset(path=str(output_root), split="vlm2vec_train", num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|video_pad|>\n"
        "What actions or objects interactions are the person in the video doing?\n"
    )
    assert sample["positive"]["text"] == "brush_hair"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "HMDB51"
    assert sample["metadata"]["video_path"] == "brush_hair/brush_hair_a.avi"
    assert sample["query"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["query"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["query"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_hmdb51_runtime_reads_existing_legacy_layout(tmp_path: Path):
    output_root = tmp_path / "HMDB51"
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
                "video_id": "brush_hair_a",
                "video_path": "brush_hair/brush_hair_a.avi",
                "label_text": "brush_hair",
                "negative_text": "",
                "class_name": "brush_hair",
            }
        ],
        str(output_root / "data" / "samples.lance"),
        mode="create",
    )
    lance.write_dataset(
        [
            {
                "asset_id": "asset_a",
                "video_id": "brush_hair_a",
                "video_path": "brush_hair/brush_hair_a.avi",
                "video_bytes": video_bytes,
                "fps": 3.0,
                "total_num_frames": 3,
            }
        ],
        str(output_root / "data" / "media.lance"),
        mode="create",
    )
    ensure_hmdb51_indices(output_root)

    try:
        Hmdb51TrainDataset(path=str(output_root), split="vlm2vec_train", num_frames=2)
    except FileNotFoundError as error:
        assert "allow_raw_video=True" in str(error)
    else:
        raise AssertionError("Legacy raw-video layout should require explicit opt-in")

    dataset = Hmdb51TrainDataset(
        path=str(output_root),
        split="vlm2vec_train",
        num_frames=2,
        allow_raw_video=True,
    )
    sample = dataset[0]

    assert dataset.storage_layout == "legacy"
    assert sample["positive"]["text"] == "brush_hair"
    assert sample["negative"]["text"] == ""
    assert len(sample["query"]["media"][0]["content"]) == 2
