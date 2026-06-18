from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import lance
import numpy as np
import pytest

from vlm2emb.data.datasets import qvhighlight_train as qvhighlight_runtime
from vlm2emb.data.conversion import qvhighlight_train as qvhighlight_conversion
from vlm2emb.data.conversion.qvhighlight_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_qvhighlight_indices,
    load_qvhighlight_split_rows,
    write_qvhighlight_root,
)
from vlm2emb.data.datasets.qvhighlight_train import QvhighlightTrainDataset


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


def _jsonl_row(qid: int, vid: str, query: str, *, windows: list[list[float]] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "qid": qid,
        "query": query,
        "duration": 1.0,
        "vid": vid,
    }
    if windows is not None:
        row["relevant_clip_ids"] = [0, 1]
        row["saliency_scores"] = [[4, 3, 2], [1, 2, 3]]
        row["relevant_windows"] = windows
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_fixture(raw_root: Path, video_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        raw_root / "highlight_train_release.jsonl",
        [
            _jsonl_row(9769, "train_a", "some military patriots takes us through safety.", windows=[[0.0, 0.6]]),
            _jsonl_row(10016, "train_b", "Man in baseball cap eats before interview.", windows=[[0.1, 0.7]]),
        ],
    )
    _write_jsonl(
        raw_root / "highlight_val_release.jsonl",
        [_jsonl_row(2579, "val_a", "A girl and her mother cooked.", windows=[[0.0, 0.5]])],
    )
    _write_jsonl(
        raw_root / "highlight_test_release.jsonl",
        [_jsonl_row(3158, "test_a", "A video covering hill and water from a boat")],
    )
    video_bytes = _make_mp4_bytes()
    for video_id in ("train_a", "train_b", "val_a", "test_a"):
        (video_root / f"{video_id}.mp4").write_bytes(video_bytes)


def test_load_qvhighlight_split_rows_preserves_json_fields(tmp_path: Path):
    raw_root = tmp_path / "raw"
    video_root = tmp_path / "videos"
    _write_fixture(raw_root, video_root)

    split_rows = load_qvhighlight_split_rows(raw_root)
    rows_by_split = {name: list(rows) for name, rows in split_rows.items()}

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_train"][0] == {
        "qid": 9769,
        "query": "some military patriots takes us through safety.",
        "duration": 1.0,
        "vid": "train_a",
        "relevant_clip_ids": [0, 1],
        "saliency_scores": [[4, 3, 2], [1, 2, 3]],
        "relevant_windows": [[0.0, 0.6]],
        "query_frame_unit_key": "train_a#full",
        "positive_frame_unit_keys": ["train_a#segment=0.000-0.600"],
    }


def test_collect_qvhighlight_video_refs_deduplicates_all_splits(tmp_path: Path):
    raw_root = tmp_path / "raw"
    video_root = tmp_path / "videos"
    _write_fixture(raw_root, video_root)

    video_refs, split_counts = collect_video_refs(raw_root)

    assert list(video_refs.items()) == [
        ("train_a", "train_a.mp4"),
        ("train_b", "train_b.mp4"),
        ("val_a", "val_a.mp4"),
        ("test_a", "test_a.mp4"),
    ]
    assert split_counts == {
        "official_train": 2,
        "official_val": 1,
        "official_test": 1,
        "official_train_without_mmeb_v2_eval": 2,
    }


def test_write_qvhighlight_root_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "raw"
    video_root = tmp_path / "videos"
    output_root = tmp_path / "QVHighlights"
    _write_fixture(raw_root, video_root)

    payload = write_qvhighlight_root(
        raw_root=raw_root,
        video_root=video_root,
        output_root=output_root,
        num_workers=2,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 4
    assert payload["frame_unit_rows"] == 7
    assert payload["splits"]["official_train_without_mmeb_v2_eval"] == 2
    assert payload["exclusion_rows"] == 2
    assert payload["bad_media_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    videos = lance.dataset(str(output_root / "data" / "videos.lance"))
    frame_units = lance.dataset(str(output_root / "data" / "frames.lance"))
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    bad_media = lance.dataset(str(output_root / "data" / "exclusions" / "bad_media.lance"))
    assert videos.count_rows() == 4
    assert frame_units.count_rows() == 7
    assert train.schema.names == [
        "qid",
        "query",
        "duration",
        "vid",
        "relevant_clip_ids",
        "saliency_scores",
        "relevant_windows",
        "query_frame_unit_key",
        "positive_frame_unit_keys",
    ]
    assert exclusions.count_rows() == 2
    assert bad_media.count_rows() == 0
    assert any("video_id" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = QvhighlightTrainDataset(
        path=str(output_root),
        num_frames=2,
        transform_kwargs={"positive": {"window_selection": "first"}},
    )
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
            "vlm2emb.data.datasets.qvhighlight_train.sample_prefixed_preextracted_frame_unit",
            wraps=qvhighlight_runtime.sample_prefixed_preextracted_frame_unit,
        ) as sample_frame_unit,
    ):
        sample = dataset[0]

    assert sample_frame_unit.call_count == 2
    assert sample["query"]["text"] == (
        "<|video_pad|>\n"
        "Find the clip that corresponds to the described scene in the given video: "
        "some military patriots takes us through safety.\n"
    )
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "QVHighlights"
    assert sample["metadata"]["qid"] == 9769
    assert sample["query"]["media"][0]["kind"] == "video"
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["clip_start"] == 0.0
    assert sample["positive"]["media"][0]["metadata"]["clip_end"] == pytest.approx(0.6)
    assert sample["positive"]["media"][0]["metadata"]["clip_total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["source_total_num_frames"] == 4
    assert sample["positive"]["media"][0]["metadata"]["sampled_indices"] == [0, 2]
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "segment"


def test_write_qvhighlight_root_trims_invalid_positive_frame_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_root = tmp_path / "raw"
    video_root = tmp_path / "videos"
    output_root = tmp_path / "QVHighlights"
    _write_fixture(raw_root, video_root)
    _write_jsonl(
        raw_root / "highlight_train_release.jsonl",
        [
            _jsonl_row(
                9769,
                "train_a",
                "some military patriots takes us through safety.",
                windows=[[0.0, 0.6], [0.6, 0.9]],
            ),
            _jsonl_row(10016, "train_b", "Man in baseball cap eats before interview.", windows=[[0.1, 0.7]]),
        ],
    )

    original_build_segment = qvhighlight_conversion.build_segment_frame_unit_from_bytes

    def fail_one_segment(*args, **kwargs):
        if kwargs.get("frame_unit_key") == "train_a#segment=0.600-0.900":
            raise ValueError("invalid segment fixture")
        return original_build_segment(*args, **kwargs)

    monkeypatch.setattr(qvhighlight_conversion, "build_segment_frame_unit_from_bytes", fail_one_segment)

    payload = write_qvhighlight_root(
        raw_root=raw_root,
        video_root=video_root,
        output_root=output_root,
        num_workers=1,
        video_batch_size=2,
    )

    assert payload["bad_media_rows"] == 1
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    train_rows = train.to_table().to_pylist()
    train_a = next(row for row in train_rows if row["vid"] == "train_a")
    assert train_a["relevant_windows"][0][0] == pytest.approx(0.0)
    assert train_a["relevant_windows"][0][1] == pytest.approx(0.6)
    assert train_a["positive_frame_unit_keys"] == ["train_a#segment=0.000-0.600"]

    dataset = QvhighlightTrainDataset(
        path=str(output_root),
        num_frames=2,
        transform_kwargs={"positive": {"window_selection": "random"}},
    )
    for _ in range(10):
        sample = dataset[0]
        assert sample["positive"]["media"][0]["metadata"]["clip_end"] == pytest.approx(0.6)


def test_ensure_qvhighlight_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_qvhighlight_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_qvhighlight_indices should fail when frames.lance is missing")
