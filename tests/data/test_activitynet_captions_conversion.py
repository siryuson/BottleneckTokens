from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import av
import lance
import numpy as np

from vlm2emb.data.datasets import activitynet_captions_train as activitynet_runtime
from vlm2emb.data.conversion.activitynet_captions_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_activitynet_captions_indices,
    load_activitynet_captions_split_rows,
    write_activitynet_captions_root,
)
from vlm2emb.data.datasets.activitynet_captions_train import ActivitynetCaptionsTrainDataset


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


def _video_row(video_id: str, sentences: list[str], timestamps: list[list[float]]) -> dict[str, object]:
    return {
        "video_id": video_id,
        "video": f"{video_id}.mp4",
        "caption": " ".join(sentences),
        "source": "ActivityNet_Captions",
        "duration": 82.73,
        "timestamps": timestamps,
        "sentences": sentences,
    }


def _write_activitynet_fixture(root: Path) -> tuple[Path, list[Path]]:
    source_root = root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "activitynet_captions_train.json").write_text(
        json.dumps(
            [
                _video_row(
                    "v_QOlSCBRmfWY",
                    [
                        "A young woman is seen standing in a room and leads into her dancing.",
                        "The girl dances around the room while the camera captures her movements.",
                    ],
                    [[0.0, 1.0], [0.0, 1.0]],
                )
            ]
        ),
        encoding="utf-8",
    )
    (source_root / "activitynet_captions_val1.json").write_text(
        json.dumps([_video_row("v_uqiMw7tQ1Cc", ["A weight lifting tutorial is given."], [[0.28, 55.15]])]),
        encoding="utf-8",
    )
    (source_root / "activitynet_captions_val2.json").write_text(
        json.dumps([_video_row("v_ZREEgMgSz_o", ["Two people cook food."], [[0.0, 4.14]])]),
        encoding="utf-8",
    )

    video_bytes = _make_mp4_bytes()
    archive = root / "ActivityNet_Videos.tar.part-000"
    with tarfile.open(archive, "w") as tar:
        _add_tar_member(tar, "Activity_Videos/v_QOlSCBRmfWY.mp4", video_bytes)
        _add_tar_member(tar, "Activity_Videos/v_uqiMw7tQ1Cc.mp4", video_bytes)
        _add_tar_member(tar, "Activity_Videos/v_ZREEgMgSz_o.mp4", video_bytes)
    return source_root, [archive]


def test_load_activitynet_captions_split_rows_expands_segments(tmp_path: Path):
    source_root, _ = _write_activitynet_fixture(tmp_path)

    rows_by_split = load_activitynet_captions_split_rows(source_root)

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert len(rows_by_split["official_train"]) == 2
    assert rows_by_split["official_train"][0]["segment_index"] == 0
    assert rows_by_split["official_train"][0]["segment_start"] == 0.0
    assert rows_by_split["official_train"][0]["segment_end"] == 1.0
    assert rows_by_split["official_train"][0]["segment_sentence"] == (
        "A young woman is seen standing in a room and leads into her dancing."
    )


def test_collect_video_refs_deduplicates_all_splits(tmp_path: Path):
    source_root, _ = _write_activitynet_fixture(tmp_path)

    video_refs, split_counts = collect_video_refs(source_root)

    assert list(video_refs.items()) == [
        ("v_QOlSCBRmfWY.mp4", "v_QOlSCBRmfWY"),
        ("v_uqiMw7tQ1Cc.mp4", "v_uqiMw7tQ1Cc"),
        ("v_ZREEgMgSz_o.mp4", "v_ZREEgMgSz_o"),
    ]
    assert split_counts == {
        "official_train": 2,
        "official_val1": 1,
        "official_val2": 1,
        "official_train_without_mmeb_v2_eval": 2,
    }


def test_write_activitynet_captions_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    source_root, archives = _write_activitynet_fixture(tmp_path / "fixture")
    output_root = tmp_path / "ActivityNet-Captions"

    payload = write_activitynet_captions_root(
        source_root=source_root,
        video_archives=archives,
        output_root=output_root,
        video_batch_size=2,
    )

    assert payload["video_rows"] == 3
    assert payload["frame_unit_rows"] == 4
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
    assert videos.count_rows() == 3
    assert frame_units.count_rows() == 4
    assert train.count_rows() == 2
    assert exclusions.count_rows() == 0
    assert bad_media.count_rows() == 0
    assert any("video" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = ActivitynetCaptionsTrainDataset(path=str(output_root), num_frames=2)
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
            "vlm2emb.data.datasets.activitynet_captions_train.sample_preextracted_frame_unit",
            wraps=activitynet_runtime.sample_preextracted_frame_unit,
        ) as sample_frame_unit,
    ):
        sample = dataset[0]

    assert sample_frame_unit.call_count == 1
    assert sample["query"]["text"] == (
        "Find the video segment that corresponds to the following description: "
        "A young woman is seen standing in a room and leads into her dancing.\n"
    )
    assert sample["positive"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video segment.\n"
    assert sample["negative"]["text"] == ""
    assert sample["query"]["media"] == []
    assert sample["negative"]["media"] == []
    assert sample["metadata"]["dataset_name"] == "ActivityNet-Captions"
    assert sample["metadata"]["segment_start"] == 0.0
    assert sample["metadata"]["segment_end"] == 1.0
    assert sample["positive"]["media"][0]["kind"] == "video"
    assert len(sample["positive"]["media"][0]["content"]) == 2
    assert sample["positive"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["positive"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["source_total_num_frames"] == 3
    assert sample["positive"]["media"][0]["metadata"]["sampled_indices"] == [0, 2]
    assert sample["positive"]["media"][0]["metadata"]["frame_unit_type"] == "segment"


def test_ensure_activitynet_captions_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_activitynet_captions_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_activitynet_captions_indices should fail when frames.lance is missing")
