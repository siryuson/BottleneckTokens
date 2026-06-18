from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import av
import lance
import numpy as np

from vlm2emb.data.conversion.nextqa_train import (
    SUPPORTED_SPLITS,
    collect_video_refs,
    ensure_nextqa_indices,
    load_nextqa_split_rows,
    load_nextqa_train_rows,
    write_nextqa_root,
)
from vlm2emb.data.datasets.nextqa_train import NextqaTrainDataset


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


def _row(
    *,
    video: str,
    video_path: str,
    qid: str,
    question: str,
    answer: str,
    options: list[str],
) -> dict[str, object]:
    return {
        "video": video,
        "frame_count": "3",
        "width": "8",
        "height": "8",
        "question": question,
        "answer": answer,
        "qid": qid,
        "type": "DC",
        "options": options,
        "video_path": video_path,
    }


def _write_nextqa_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    annotation = root / "nextqa_train.json"
    video_zip = root / "NExTVideo.zip"
    annotation.write_text(
        json.dumps(
            [
                _row(
                    video="3238737531",
                    video_path="1164/3238737531",
                    qid="2",
                    question="how many children are in the video",
                    answer="3",
                    options=["one", "three", "seven", "two", "five"],
                ),
                _row(
                    video="8968804598",
                    video_path="1006/8968804598",
                    qid="0",
                    question="why is the blue sweater guy looking at the shirtless men",
                    answer="4",
                    options=["sharing with his friends", "found the man funny", "poor vision", "keep hands warm", "training"],
                ),
            ]
        ),
        encoding="utf-8",
    )
    video_bytes = _make_mp4_bytes()
    with zipfile.ZipFile(video_zip, mode="w") as archive:
        archive.writestr("NExTVideo/1164/3238737531.mp4", video_bytes)
        archive.writestr("NExTVideo/1006/8968804598.mp4", video_bytes)
    return annotation, video_zip


def test_load_nextqa_train_rows_preserves_source_fields(tmp_path: Path):
    annotation, _ = _write_nextqa_fixture(tmp_path)

    rows = load_nextqa_train_rows(annotation)
    split_rows = load_nextqa_split_rows(annotation)

    assert tuple(split_rows) == SUPPORTED_SPLITS
    assert rows[0] == {
        "video": "3238737531",
        "frame_count": "3",
        "width": "8",
        "height": "8",
        "question": "how many children are in the video",
        "answer": "3",
        "qid": "2",
        "type": "DC",
        "options": ["one", "three", "seven", "two", "five"],
        "video_path": "1164/3238737531",
    }


def test_collect_video_refs_deduplicates_all_rows(tmp_path: Path):
    annotation, _ = _write_nextqa_fixture(tmp_path)

    video_refs, split_counts = collect_video_refs(annotation)

    assert list(video_refs.items()) == [
        ("1164/3238737531", "3238737531"),
        ("1006/8968804598", "8968804598"),
    ]
    assert split_counts == {
        "official_train": 2,
        "official_train_without_mmeb_v2_eval": 2,
    }


def test_write_nextqa_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "NExTQA"
    annotation, video_zip = _write_nextqa_fixture(source_root)

    payload = write_nextqa_root(
        annotation_path=annotation,
        video_zip=video_zip,
        output_root=output_root,
        num_workers=2,
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
    assert train.schema.names == [
        "video",
        "frame_count",
        "width",
        "height",
        "question",
        "answer",
        "qid",
        "type",
        "options",
        "video_path",
    ]
    assert exclusions.count_rows() == 0
    assert bad_media.count_rows() == 0
    assert any("video_path" in index.field_names for index in videos.describe_indices())
    assert any("frame_unit_key" in index.field_names for index in frame_units.describe_indices())

    dataset = NextqaTrainDataset(path=str(output_root), num_frames=2)
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|video_pad|>\n"
        "Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question: how many children are in the video\n"
        "Options:\n"
        "(A) one\n"
        "(B) three\n"
        "(C) seven\n"
        "(D) two\n"
        "(E) five\n"
    )
    assert sample["positive"]["text"] == "two"
    assert sample["positive"]["media"] == []
    assert sample["negative"]["text"] == ""
    assert sample["negative"]["media"] == []
    assert sample["metadata"]["dataset_name"] == "NExTQA"
    assert sample["metadata"]["video_path"] == "1164/3238737531"
    assert sample["query"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"]["fps"] == 3.0
    assert sample["query"]["media"][0]["metadata"]["total_num_frames"] == 3
    assert sample["query"]["media"][0]["metadata"]["frame_unit_type"] == "full_video"


def test_ensure_nextqa_indices_requires_frame_unit_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_nextqa_indices(output_root)
    except FileNotFoundError as error:
        assert "frames.lance" in str(error)
    else:
        raise AssertionError("ensure_nextqa_indices should fail when frames.lance is missing")
