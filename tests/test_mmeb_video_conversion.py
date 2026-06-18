from __future__ import annotations

import json
from pathlib import Path
import lance
import pytest

import vlm2emb.data.conversion.mmeb_v2.video as video_conversion
from vlm2emb.data.conversion.mmeb_v2.video import (
    convert_video_cls_dataset,
    convert_video_momentseeker_dataset,
    convert_video_mret_dataset,
    convert_video_qa_dataset,
    convert_video_ret_dataset,
)

def _write_frame_dir(path: Path, *, num_frames: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(num_frames):
        (path / f"{index:03d}.jpg").write_bytes(f"frame-{index}".encode())


def test_load_records_from_hf_reads_online_hf_identity(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_load_hf_records(ref: str, **kwargs):
        captured["ref"] = ref
        captured.update(kwargs)
        return [{"video_id": "video_0", "caption": "a dog runs"}]

    monkeypatch.setattr(video_conversion, "load_hf_records", _fake_load_hf_records)

    records = video_conversion._load_records_from_hf("MSR-VTT")

    assert records == [{"video_id": "video_0", "caption": "a dog runs"}]
    assert captured == {
        "ref": "siyrus/BToks-MSR-VTT::test_1k/test",
    }


def test_load_records_from_hf_mvbench_reads_all_online_subsets(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_load_hf_records(ref: str, **kwargs):
        captured["ref"] = ref
        captured.update(kwargs)
        return [
            {
                "video": "clip_0",
                "question": "What happens next?",
                "candidates": ["run", "jump"],
                "answer": "run",
                "_source_file": "action_sequence",
            }
        ]

    monkeypatch.setattr(video_conversion, "load_hf_records", _fake_load_hf_records)

    records = video_conversion._load_records_from_hf("MVBench")

    assert records == [
        {
            "video": "clip_0",
            "question": "What happens next?",
            "candidates": ["run", "jump"],
            "answer": "run",
            "_source_file": "action_sequence",
        }
    ]
    assert captured["ref"] == "siyrus/BToks-MVBench::train"
    assert captured["subset_names"] == video_conversion.MVBENCH_SUBSETS
    assert captured["subset_field_name"] == "_source_file"


def test_convert_video_cls_dataset_writes_minimal_rerun_artifact(tmp_path):
    raw_root = tmp_path / "raw"
    data_dir = raw_root / "video-tasks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_root = raw_root / "video-tasks" / "frames" / "K700"
    _write_frame_dir(frame_root / "video_0")
    _write_frame_dir(frame_root / "video_1")

    records = [
        {
            "video_id": "video_0",
            "video_path": "K700-val/running/video_0.mp4",
            "pos_text": "running",
            "qry_instruction": "Recognize the category of the video content.",
            "qry_text": None,
        },
        {
            "video_id": "video_1",
            "video_path": "K700-val/jumping/video_1.mp4",
            "pos_text": "jumping",
            "qry_instruction": "Recognize the category of the video content.",
            "qry_text": None,
        },
    ]
    with (data_dir / "k700.jsonl").open("w") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record) + "\n")

    output_root = tmp_path / "rerun"
    output_path = convert_video_cls_dataset(
        "K700",
        output_root,
        mmeb_v2_root=raw_root,
        max_workers=2,
    )

    assert output_path == output_root / "video-tasks" / "K700"
    assert (output_path / "queries.lance").exists()
    assert (output_path / "candidates.lance").exists()
    assert (output_path / "qrels.lance").exists()
    assert (output_path / "metadata.json").exists()

    metadata = json.loads((output_path / "metadata.json").read_text())
    assert metadata == {
        "name": "K700",
        "task_type": "video_classification",
        "modality": "video",
        "benchmark": "MMEB-V2",
    }

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert [row["id"] for row in queries] == [0, 1]
    assert queries[0]["video_id"] == "video_0"
    assert queries[0]["pos_text"] == "running"
    assert queries[0]["qry_instruction"] == "Recognize the category of the video content."
    assert queries[0]["video"] == [b"frame-0", b"frame-1"]
    assert "render_metadata_json" not in queries[0]
    assert "text" not in queries[0]

    assert candidates == [
        {"id": 0, "text": "running"},
        {"id": 1, "text": "jumping"},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "sparse",
            "candidate_ids": [0],
            "candidate_scores": [1.0],
        },
        {
            "query_id": 1,
            "mode": "sparse",
            "candidate_ids": [1],
            "candidate_scores": [1.0],
        },
    ]


def test_convert_video_cls_dataset_writes_exhaustive_ssv2_qrels(tmp_path):
    raw_root = tmp_path / "raw"
    data_dir = raw_root / "video-tasks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_root = raw_root / "video-tasks" / "frames" / "SSv2"
    _write_frame_dir(frame_root / "231")

    record = {
        "video_id": 231,
        "pos_text": "holding pen",
        "neg_text": ["approaching adaptor with your camera", "holding pen", "opening book"],
    }
    with (data_dir / "ssv2.jsonl").open("w") as file_handle:
        file_handle.write(json.dumps(record) + "\n")

    output_root = tmp_path / "rerun"
    output_path = convert_video_cls_dataset(
        "SmthSmthV2",
        output_root,
        mmeb_v2_root=raw_root,
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert queries[0]["id"] == 0
    assert queries[0]["video_id"] == 231
    assert queries[0]["video"] == [b"frame-0", b"frame-1"]
    assert [row["text"] for row in candidates] == [
        "holding pen",
        "approaching adaptor with your camera",
        "opening book",
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "exhaustive",
            "candidate_ids": [0, 1, 2],
            "candidate_scores": [1.0, 0.0, 0.0],
        }
    ]


def test_convert_video_cls_dataset_fails_on_blank_positive_label(tmp_path):
    raw_root = tmp_path / "raw"
    data_dir = raw_root / "video-tasks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_root = raw_root / "video-tasks" / "frames" / "K700"
    _write_frame_dir(frame_root / "video_0")

    record = {
        "video_id": "video_0",
        "pos_text": " ",
    }
    with (data_dir / "k700.jsonl").open("w") as file_handle:
        file_handle.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="blank classification candidates"):
        convert_video_cls_dataset(
            "K700",
            tmp_path / "rerun",
            mmeb_v2_root=raw_root,
            max_workers=2,
        )


def test_convert_video_cls_dataset_fails_on_blank_negative_label(tmp_path):
    raw_root = tmp_path / "raw"
    data_dir = raw_root / "video-tasks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_root = raw_root / "video-tasks" / "frames" / "SSv2"
    _write_frame_dir(frame_root / "231")

    record = {
        "video_id": 231,
        "pos_text": "holding pen",
        "neg_text": ["", "opening book"],
    }
    with (data_dir / "ssv2.jsonl").open("w") as file_handle:
        file_handle.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="blank classification candidates"):
        convert_video_cls_dataset(
            "SmthSmthV2",
            tmp_path / "rerun",
            mmeb_v2_root=raw_root,
            max_workers=2,
        )


def test_convert_video_qa_dataset_writes_local_exhaustive_candidates(tmp_path):
    raw_root = tmp_path / "raw"
    frame_root = raw_root / "video-tasks" / "frames" / "video_qa" / "NExTQA"
    _write_frame_dir(frame_root / "123")
    original_loader = video_conversion._load_records_from_hf
    video_conversion._load_records_from_hf = lambda dataset_name: [
        {
            "video": 123,
            "question": "What is the person doing?",
            "answer": 1,
            "a0": "running",
            "a1": "jumping",
            "a2": "sitting",
            "a3": "sleeping",
            "a4": "reading",
            "qid": 7,
            "type": "action",
        }
    ]

    try:
        output_root = tmp_path / "rerun"
        output_path = convert_video_qa_dataset(
            "NExTQA",
            output_root,
            mmeb_v2_root=raw_root,
            max_workers=2,
        )
    finally:
        video_conversion._load_records_from_hf = original_loader

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert queries[0]["id"] == 0
    assert queries[0]["video"] == [b"frame-0", b"frame-1"]
    assert queries[0]["question"] == "What is the person doing?"
    assert queries[0]["a1"] == "jumping"
    assert "render_metadata_json" not in queries[0]

    assert candidates == [
        {"id": 0, "text": "running"},
        {"id": 1, "text": "jumping"},
        {"id": 2, "text": "sitting"},
        {"id": 3, "text": "sleeping"},
        {"id": 4, "text": "reading"},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "exhaustive",
            "candidate_ids": [0, 1, 2, 3, 4],
            "candidate_scores": [0.0, 1.0, 0.0, 0.0, 0.0],
        }
    ]


def test_convert_video_qa_dataset_writes_mvbench_prefixed_candidates_and_sparse_qrels(monkeypatch, tmp_path):
    record = {
        "video": "clip_0",
        "question": "Why did Castle dress like a fairy?",
        "candidates": [
            "To get her to trust him",
            "He secretly loved fairies",
            "He lost a bet with Emily",
        ],
        "answer": "To get her to trust him",
        "_source_file": "action_sequence",
    }

    monkeypatch.setattr(video_conversion, "_resolve_frame_root", lambda *args, **kwargs: Path("/tmp/frames"))
    monkeypatch.setattr(video_conversion, "_load_annotation_records", lambda *args, **kwargs: [record])
    monkeypatch.setattr(
        video_conversion,
        "_load_video_qa_query_frames",
        lambda *args, **kwargs: [b"frame-0", b"frame-1"],
    )

    output_path = convert_video_qa_dataset(
        "MVBench",
        tmp_path / "rerun",
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert queries[0]["id"] == 0
    assert queries[0]["question"] == "Why did Castle dress like a fairy?"
    assert queries[0]["video"] == [b"frame-0", b"frame-1"]
    assert "render_metadata_json" not in queries[0]

    assert candidates == [
        {"id": 0, "text": "(A) To get her to trust him"},
        {"id": 1, "text": "(B) He secretly loved fairies"},
        {"id": 2, "text": "(C) He lost a bet with Emily"},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "sparse",
            "candidate_ids": [0],
            "candidate_scores": [1.0],
        }
    ]


def test_convert_video_qa_dataset_fails_on_blank_option_text(tmp_path):
    raw_root = tmp_path / "raw"
    frame_root = raw_root / "video-tasks" / "frames" / "video_qa" / "NExTQA"
    _write_frame_dir(frame_root / "123")
    original_loader = video_conversion._load_records_from_hf
    video_conversion._load_records_from_hf = lambda dataset_name: [
        {
            "video": 123,
            "question": "What is the person doing?",
            "answer": 1,
            "a0": "running",
            "a1": None,
            "a2": "sitting",
            "a3": "sleeping",
            "a4": "reading",
        }
    ]

    try:
        with pytest.raises(ValueError, match="blank answer options"):
            convert_video_qa_dataset(
                "NExTQA",
                tmp_path / "rerun",
                mmeb_v2_root=raw_root,
                max_workers=2,
            )
    finally:
        video_conversion._load_records_from_hf = original_loader


def test_convert_video_ret_dataset_writes_sparse_global_candidates(tmp_path):
    raw_root = tmp_path / "raw"
    frame_root = raw_root / "video-tasks" / "frames" / "data" / "ziyan" / "video_retrieval" / "MSR-VTT" / "frames"
    _write_frame_dir(frame_root / "video_0")
    _write_frame_dir(frame_root / "video_1")
    original_loader = video_conversion._load_records_from_hf
    video_conversion._load_records_from_hf = lambda dataset_name: [
        {"video_id": "video_0", "caption": "a dog runs"},
        {"video_id": "video_1", "caption": "a cat sleeps"},
    ]

    try:
        output_root = tmp_path / "rerun"
        output_path = convert_video_ret_dataset(
            "MSR-VTT",
            output_root,
            mmeb_v2_root=raw_root,
            max_workers=2,
        )
    finally:
        video_conversion._load_records_from_hf = original_loader

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert [row["id"] for row in queries] == [0, 1]
    assert queries[0]["caption"] == "a dog runs"
    assert "text" not in queries[0]
    assert candidates == [
        {"id": 0, "video_id": "video_0", "video": [b"frame-0", b"frame-1"]},
        {"id": 1, "video_id": "video_1", "video": [b"frame-0", b"frame-1"]},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "sparse",
            "candidate_ids": [0],
            "candidate_scores": [1.0],
        },
        {
            "query_id": 1,
            "mode": "sparse",
            "candidate_ids": [1],
            "candidate_scores": [1.0],
        },
    ]


def test_convert_video_mret_dataset_writes_exhaustive_local_clip_candidates(tmp_path):
    raw_root = tmp_path / "raw"
    data_dir = raw_root / "video-tasks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_root = raw_root / "video-tasks" / "frames" / "video_mret" / "QVHighlight"

    query_root = frame_root / "sample_query" / "query"
    positive_root = frame_root / "sample_query" / "positive_clip"
    negative_root = frame_root / "sample_query" / "negative_clip_0"
    _write_frame_dir(query_root)
    _write_frame_dir(positive_root)
    _write_frame_dir(negative_root)

    record = {
        "query": "Find the matching clip.",
        "clips_dir_path": "/tmp/source/sample_query",
    }
    with (data_dir / "qvhighlight.jsonl").open("w") as file_handle:
        file_handle.write(json.dumps(record) + "\n")

    output_root = tmp_path / "rerun"
    output_path = convert_video_mret_dataset(
        "QVHighlight",
        output_root,
        mmeb_v2_root=raw_root,
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert queries == [
        {
            "id": 0,
            "query": "Find the matching clip.",
            "clips_dir_path": "/tmp/source/sample_query",
            "video": [b"frame-0", b"frame-1"],
        }
    ]
    assert candidates == [
        {"id": 0, "video": [b"frame-0", b"frame-1"]},
        {"id": 1, "video": [b"frame-0", b"frame-1"]},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "exhaustive",
            "candidate_ids": [0, 1],
            "candidate_scores": [1.0, 0.0],
        }
    ]


def test_convert_video_momentseeker_dataset_preserves_query_modalities_and_all_positive_clips(tmp_path):
    raw_root = tmp_path / "raw"
    frame_root = raw_root / "video-tasks" / "frames" / "video_mret" / "MomentSeeker"
    _write_frame_dir(frame_root / "video_frames" / "videos_clip_a")
    _write_frame_dir(frame_root / "video_frames" / "videos_pos_0")
    _write_frame_dir(frame_root / "video_frames" / "videos_pos_1")
    _write_frame_dir(frame_root / "video_frames" / "videos_neg_0")
    image_dir = frame_root / "query_images" / "movie101_77"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "frame_173.jpg").write_bytes(b"query-image")

    original_loader = video_conversion._load_records_from_hf
    video_conversion._load_records_from_hf = lambda dataset_name: [
        {
            "query": "Find the matching clip for this video segment.",
            "input_frames": "videos/clip_a.mp4",
            "positive_frames": [
                {"output_path": "videos/pos_0.mp4", "frame_path": "videos/pos_0_mid_frame.png", "start_time": 0.0, "end_time": 5.0},
                {"output_path": "videos/pos_1.mp4", "frame_path": "videos/pos_1_mid_frame.png", "start_time": 5.0, "end_time": 9.0},
            ],
            "negative_frames": [
                {"output_path": "videos/neg_0.mp4", "frame_path": "videos/neg_0_mid_frame.png", "start_time": 9.0, "end_time": 12.0},
            ],
        },
        {
            "query": "Find the matching clip for this image.",
            "input_frames": "images/movie101_77/frame_173.jpg",
            "positive_frames": [
                {"output_path": "videos/pos_1.mp4", "frame_path": "videos/pos_1_mid_frame.png", "start_time": 5.0, "end_time": 9.0},
            ],
            "negative_frames": [
                {"output_path": "videos/neg_0.mp4", "frame_path": "videos/neg_0_mid_frame.png", "start_time": 9.0, "end_time": 12.0},
            ],
        },
        {
            "query": "Find the matching clip for this text only query.",
            "input_frames": "",
            "positive_frames": [
                {"output_path": "videos/pos_0.mp4", "frame_path": "videos/pos_0_mid_frame.png", "start_time": 0.0, "end_time": 5.0},
            ],
            "negative_frames": [
                {"output_path": "videos/neg_0.mp4", "frame_path": "videos/neg_0_mid_frame.png", "start_time": 9.0, "end_time": 12.0},
            ],
        },
    ]

    try:
        output_root = tmp_path / "rerun"
        output_path = convert_video_momentseeker_dataset(
            "MomentSeeker",
            output_root,
            mmeb_v2_root=raw_root,
            max_workers=2,
        )
    finally:
        video_conversion._load_records_from_hf = original_loader

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert queries[0]["id"] == 0
    assert queries[0]["video"] == [b"frame-0", b"frame-1"]
    assert queries[0]["image"] is None
    assert queries[1]["image"] == b"query-image"
    assert queries[1]["video"] is None
    assert queries[2]["image"] is None
    assert queries[2]["video"] is None
    assert "render_metadata_json" not in queries[0]

    assert candidates[0]["output_path"] == "videos/pos_0.mp4"
    assert candidates[0]["video"] == [b"frame-0", b"frame-1"]
    assert candidates[1]["output_path"] == "videos/pos_1.mp4"
    assert candidates[2]["output_path"] == "videos/neg_0.mp4"

    assert qrels[0] == {
        "query_id": 0,
        "mode": "exhaustive",
        "candidate_ids": [0, 1, 2],
        "candidate_scores": [1.0, 1.0, 0.0],
    }
    assert qrels[1] == {
        "query_id": 1,
        "mode": "exhaustive",
        "candidate_ids": [3, 4],
        "candidate_scores": [1.0, 0.0],
    }
    assert qrels[2] == {
        "query_id": 2,
        "mode": "exhaustive",
        "candidate_ids": [5, 6],
        "candidate_scores": [1.0, 0.0],
    }


def test_convert_video_momentseeker_dataset_fails_when_query_has_no_candidates(tmp_path):
    raw_root = tmp_path / "raw"
    frame_root = raw_root / "video-tasks" / "frames" / "video_mret" / "MomentSeeker"
    _write_frame_dir(frame_root / "video_frames" / "videos_clip_a")

    original_loader = video_conversion._load_records_from_hf
    video_conversion._load_records_from_hf = lambda dataset_name: [
        {
            "query": "Find the matching clip for this video segment.",
            "input_frames": "videos/clip_a.mp4",
            "positive_frames": [],
            "negative_frames": [],
        }
    ]

    try:
        with pytest.raises(ValueError, match="has no positive or negative candidates"):
            convert_video_momentseeker_dataset(
                "MomentSeeker",
                tmp_path / "rerun",
                mmeb_v2_root=raw_root,
                max_workers=2,
            )
    finally:
        video_conversion._load_records_from_hf = original_loader
