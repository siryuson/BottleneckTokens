from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np

from vlm2emb.data.conversion.ssv2_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_ssv2_indices,
    load_ssv2_split_rows,
    write_ssv2_root,
)
from vlm2emb.data.datasets.ssv2_train import Ssv2TrainDataset, _label_text


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


def _write_ssv2_fixture(raw_root: Path, vlm2vec_root: Path) -> None:
    labels_root = raw_root / "annotations" / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    videos_root = raw_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    (labels_root / "train.json").write_text(
        json.dumps(
            [
                {
                    "id": "100",
                    "label": "holding pen",
                    "template": "Holding [something]",
                    "placeholders": ["pen"],
                },
                {
                    "id": "101",
                    "label": "moving cup up",
                    "template": "Moving [something] up",
                    "placeholders": ["cup"],
                },
            ]
        ),
        encoding="utf-8",
    )
    (labels_root / "validation.json").write_text(
        json.dumps(
            [
                {
                    "id": "200",
                    "label": "spinning cube that quickly stops spinning",
                    "template": "Spinning [something] that quickly stops spinning",
                    "placeholders": ["cube"],
                }
            ]
        ),
        encoding="utf-8",
    )

    (vlm2vec_root / "data").mkdir(parents=True, exist_ok=True)
    (vlm2vec_root / "data" / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "video_id": 100,
                        "pos_text": "holding pen",
                        "neg_text": ["moving cup up", "spinning cube that quickly stops spinning"],
                        "video_path": "100.webm",
                    }
                ),
                json.dumps(
                    {
                        "video_id": 101,
                        "pos_text": "moving cup up",
                        "neg_text": ["holding pen", "spinning cube that quickly stops spinning"],
                        "video_path": "101.webm",
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
                "video_id": 200,
                "pos_text": "spinning cube that quickly stops spinning",
                "neg_text": ["holding pen", "moving cup up"],
                "video_path": "200.webm",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    video_bytes = _make_mp4_bytes()
    for filename in ["100.webm", "101.webm", "200.webm"]:
        (videos_root / filename).write_bytes(video_bytes)


def test_load_ssv2_split_rows_preserves_official_and_vlm2vec_fields(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_ssv2_fixture(raw_root, vlm2vec_root)

    split_rows = load_ssv2_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_train"][0] == {
        "video_id": "100",
        "video_path": "100.webm",
        "label": "holding pen",
        "template": "Holding [something]",
        "placeholders": ["pen"],
    }
    assert rows_by_split["vlm2vec_train"][0] == {
        "video_id": "100",
        "video_path": "100.webm",
        "pos_text": "holding pen",
        "neg_text": ["moving cup up", "spinning cube that quickly stops spinning"],
    }


def test_ssv2_label_text_prefers_canonical_label_when_present():
    assert _label_text({"label": "canonical label", "pos_text": "legacy positive"}) == "canonical label"


def test_collect_video_refs_deduplicates_all_split_videos(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    _write_ssv2_fixture(raw_root, vlm2vec_root)

    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)

    assert list(video_refs) == ["100.webm", "101.webm", "200.webm"]
    assert split_counts == {
        "official_train": 2,
        "official_validation": 1,
        "vlm2vec_train": 2,
        "vlm2vec_test_1k": 1,
    }


def test_write_ssv2_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "raw"
    vlm2vec_root = tmp_path / "vlm2vec"
    output_root = tmp_path / "SmthSmthV2"
    _write_ssv2_fixture(raw_root, vlm2vec_root)

    payload = write_ssv2_root(
        raw_root=raw_root,
        vlm2vec_root=vlm2vec_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 3
    assert payload["splits"]["vlm2vec_train"] == 2
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "vlm2vec_train.lance"))
    official = lance.dataset(str(output_root / "data" / "official_train.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 3
    assert train.schema.names == ["video_id", "video_path", "pos_text", "neg_text"]
    assert official.schema.names == ["video_id", "video_path", "label", "template", "placeholders"]
    assert bad_media.count_rows() == 0
    assert any("video_path" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = Ssv2TrainDataset(path=str(output_root), split="vlm2vec_train", num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|video_pad|>\n"
        "What actions or object interactions are being performed by the person in the video?\n"
    )
    assert sample["positive"]["text"] == "holding pen"
    assert sample["negative"]["text"] == "moving cup up"
    assert sample["metadata"]["dataset_name"] == "SmthSmthV2"
    assert sample["metadata"]["video_path"] == "100.webm"
    assert sample["query"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["query"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["query"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_ssv2_runtime_reads_existing_legacy_layout(tmp_path: Path):
    output_root = tmp_path / "SmthSmthV2"
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
                "video_id": "100",
                "video_path": "100.webm",
                "label_text": "holding pen",
                "negative_text": "moving cup up",
                "class_name": "holding pen",
            }
        ],
        str(output_root / "data" / "samples.lance"),
        mode="create",
    )
    lance.write_dataset(
        [
            {
                "asset_id": "asset_a",
                "video_id": "100",
                "video_path": "100.webm",
                "video_bytes": video_bytes,
                "fps": 3.0,
                "total_num_frames": 3,
            }
        ],
        str(output_root / "data" / "media.lance"),
        mode="create",
    )
    ensure_ssv2_indices(output_root)

    try:
        Ssv2TrainDataset(path=str(output_root), split="vlm2vec_train", num_frames=2)
    except FileNotFoundError as error:
        assert "allow_raw_video=True" in str(error)
    else:
        raise AssertionError("Legacy raw-video layout should require explicit opt-in")

    dataset = Ssv2TrainDataset(
        path=str(output_root),
        split="vlm2vec_train",
        num_frames=2,
        allow_raw_video=True,
    )
    sample = dataset[0]

    assert dataset.storage_layout == "legacy"
    assert sample["positive"]["text"] == "holding pen"
    assert sample["negative"]["text"] == "moving cup up"
    assert len(sample["query"]["media"][0]["content"]) == 2
