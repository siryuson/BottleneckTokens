from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lance
import pandas as pd
import pytest
from PIL import Image

from vlm2emb.data.conversion.mmeb_v2.image import (
    convert_all_image_datasets,
    convert_image_cls_dataset,
    convert_image_i2i_vg_dataset,
    convert_image_i2t_dataset,
    convert_image_qa_dataset,
    convert_image_t2i_dataset,
    convert_imagenet_1k,
)


def _write_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path, format="JPEG")


def test_phase12_imagenet_pilot_entrypoint_uses_rerun_writer(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    def _capture_convert(dataset_name, output_root, **kwargs):
        captured["dataset_name"] = dataset_name
        captured["output_root"] = output_root
        captured["kwargs"] = kwargs
        return output_root

    monkeypatch.setattr(
        "vlm2emb.data.conversion.mmeb_v2.image.convert_image_cls_dataset",
        _capture_convert,
    )

    output = convert_imagenet_1k(
        tmp_path / "out",
        mmeb_eval_root=tmp_path / "annotations",
        mmeb_v2_root=tmp_path,
        max_workers=4,
    )

    assert output == tmp_path / "out"
    assert captured == {
        "dataset_name": "ImageNet-1K",
        "output_root": tmp_path / "out",
        "kwargs": {
            "mmeb_eval_root": tmp_path / "annotations",
            "mmeb_v2_root": tmp_path,
            "source_overrides": None,
            "max_workers": 4,
        },
    }


def test_convert_imagenet_1k_accepts_explicit_source_overrides(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    def _capture_convert(dataset_name, output_root, **kwargs):
        captured["dataset_name"] = dataset_name
        captured["output_root"] = output_root
        captured["kwargs"] = kwargs
        return output_root

    overrides = {
        "mmeb_eval:ImageNet-1K": tmp_path / "alt-annotations" / "ImageNet-1K",
        "mmeb_v2:image-tasks/MMEB": tmp_path / "alt-images",
    }
    monkeypatch.setattr(
        "vlm2emb.data.conversion.mmeb_v2.image.convert_image_cls_dataset",
        _capture_convert,
    )

    output = convert_imagenet_1k(
        tmp_path / "out",
        mmeb_eval_root=tmp_path / "annotations",
        mmeb_v2_root=tmp_path,
        source_overrides=overrides,
        max_workers=4,
    )

    assert output == tmp_path / "out"
    assert captured == {
        "dataset_name": "ImageNet-1K",
        "output_root": tmp_path / "out",
        "kwargs": {
            "mmeb_eval_root": tmp_path / "annotations",
            "mmeb_v2_root": tmp_path,
            "source_overrides": overrides,
            "max_workers": 4,
        },
    }


def test_convert_image_cls_dataset_accepts_explicit_source_overrides_without_default_roots(tmp_path):
    annotation_dir = tmp_path / "custom-annotations" / "ImageNet-1K"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "qry_text": ["Classify image one."],
            "qry_img_path": ["ImageNet-1K/image_0.jpg"],
            "qry_inst": ["<|image_1|> classify"],
            "tgt_text": [["tench"]],
        }
    )
    df.to_parquet(annotation_dir / "test-00000-of-00001.parquet")

    image_root = tmp_path / "custom-images"
    _write_test_image(image_root / "ImageNet-1K" / "image_0.jpg")

    output_root = tmp_path / "rerun"
    output_path = convert_image_cls_dataset(
        "ImageNet-1K",
        output_root,
        source_overrides={
            "mmeb_eval:ImageNet-1K": annotation_dir,
            "mmeb_v2:image-tasks/MMEB": image_root,
        },
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    assert output_path == output_root / "image-tasks" / "ImageNet-1K"
    assert queries[0]["qry_img_path"] == "ImageNet-1K/image_0.jpg"
    assert isinstance(queries[0]["image"], bytes)


def test_convert_image_cls_dataset_writes_minimal_rerun_artifact(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "ImageNet-1K"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": [
                "Classify image one.",
                "Classify image two.",
            ],
            "qry_img_path": [
                "ImageNet-1K/image_0.jpg",
                "ImageNet-1K/image_1.jpg",
            ],
            "qry_inst": [
                "<|image_1|> classify",
                "<|image_1|> classify",
            ],
            "tgt_text": [
                ["tench", "goldfish"],
                ["goldfish", "tench"],
            ],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "ImageNet-1K" / "image_0.jpg")
    _write_test_image(image_root / "ImageNet-1K" / "image_1.jpg")

    output_root = tmp_path / "rerun"
    output_path = convert_image_cls_dataset(
        "ImageNet-1K",
        output_root,
        mmeb_eval_root=annotation_root,
        mmeb_v2_root=mmeb_v2_root,
        max_workers=2,
    )

    assert output_path == output_root / "image-tasks" / "ImageNet-1K"
    assert (output_path / "queries.lance").exists()
    assert (output_path / "candidates.lance").exists()
    assert (output_path / "qrels.lance").exists()
    assert (output_path / "metadata.json").exists()
    assert not (output_path / "manifest.json").exists()

    metadata = json.loads((output_path / "metadata.json").read_text())
    assert metadata == {
        "name": "ImageNet-1K",
        "task_type": "image_classification",
        "modality": "image",
        "benchmark": "MMEB-V2",
    }

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert [row["id"] for row in queries] == [0, 1]
    assert queries[0]["qry_img_path"] == "ImageNet-1K/image_0.jpg"
    assert isinstance(queries[0]["image"], bytes)
    assert "images" not in queries[0]
    assert "text" not in queries[0]
    assert "render_metadata_json" not in queries[0]

    assert candidates == [
        {"id": 0, "text": "tench"},
        {"id": 1, "text": "goldfish"},
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


def test_convert_image_qa_dataset_writes_exhaustive_rerun_artifact(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "OK-VQA"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": [
                "<|image_1|>\nRepresent the given image with the following question: Question one?\n",
                "<|image_1|>\nRepresent the given image with the following question: Question two?\n",
            ],
            "qry_img_path": [
                "OK-VQA/image_0.jpg",
                "OK-VQA/image_1.jpg",
            ],
            "tgt_text": [
                ["answer-a", "answer-d", "answer-b"],
                ["answer-c", "answer-e", "answer-b"],
            ],
            "tgt_img_path": [
                ["", "", ""],
                ["", "", ""],
            ],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "OK-VQA" / "image_0.jpg")
    _write_test_image(image_root / "OK-VQA" / "image_1.jpg")

    output_root = tmp_path / "rerun"
    output_path = convert_image_qa_dataset(
        "OK-VQA",
        output_root,
        mmeb_eval_root=annotation_root,
        mmeb_v2_root=mmeb_v2_root,
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert [row["id"] for row in queries] == [0, 1]
    assert queries[0]["qry_inst"] is None
    assert isinstance(queries[0]["image"], bytes)

    assert candidates == [
        {"id": 0, "text": "answer-a"},
        {"id": 1, "text": "answer-d"},
        {"id": 2, "text": "answer-b"},
        {"id": 3, "text": "answer-c"},
        {"id": 4, "text": "answer-e"},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "exhaustive",
            "candidate_ids": [0, 1, 2],
            "candidate_scores": [1.0, 0.0, 0.0],
        },
        {
            "query_id": 1,
            "mode": "exhaustive",
            "candidate_ids": [3, 4, 2],
            "candidate_scores": [1.0, 0.0, 0.0],
        },
    ]


def test_convert_image_cls_dataset_fails_on_blank_text_candidate(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "ImageNet-1K"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": ["Classify image one."],
            "qry_img_path": ["ImageNet-1K/image_0.jpg"],
            "qry_inst": ["<|image_1|> classify"],
            "tgt_text": [["tench", ""]],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "ImageNet-1K" / "image_0.jpg")

    with pytest.raises(ValueError, match="contains blank text candidates"):
        convert_image_cls_dataset(
            "ImageNet-1K",
            tmp_path / "rerun",
            mmeb_eval_root=annotation_root,
            mmeb_v2_root=mmeb_v2_root,
            max_workers=2,
        )


def test_convert_image_qa_dataset_filters_blank_text_candidates_for_ok_vqa(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "OK-VQA"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": ["<|image_1|>\nRepresent the given image with the following question: Question one?\n"],
            "qry_img_path": ["OK-VQA/image_0.jpg"],
            "tgt_text": [["answer-a", ""]],
            "tgt_img_path": [["", ""]],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "OK-VQA" / "image_0.jpg")

    output_path = convert_image_qa_dataset(
        "OK-VQA",
        tmp_path / "rerun",
        mmeb_eval_root=annotation_root,
        mmeb_v2_root=mmeb_v2_root,
        max_workers=2,
    )

    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert candidates == [{"id": 0, "text": "answer-a"}]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "sparse",
            "candidate_ids": [0],
            "candidate_scores": [1.0],
        }
    ]


def test_convert_image_qa_dataset_fails_when_ok_vqa_positive_candidate_is_blank(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "OK-VQA"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": ["<|image_1|>\nRepresent the given image with the following question: Question one?\n"],
            "qry_img_path": ["OK-VQA/image_0.jpg"],
            "tgt_text": [["", "answer-a"]],
            "tgt_img_path": [["", ""]],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "OK-VQA" / "image_0.jpg")

    with pytest.raises(ValueError, match="loses the positive text candidate"):
        convert_image_qa_dataset(
            "OK-VQA",
            tmp_path / "rerun",
            mmeb_eval_root=annotation_root,
            mmeb_v2_root=mmeb_v2_root,
            max_workers=2,
        )


def test_convert_image_i2t_dataset_writes_conditional_rerun_artifact(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "MSCOCO_i2t"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": [
                "<|image_1|>\nFind caption one.\n",
                "<|image_1|>\nFind caption two.\n",
            ],
            "qry_img_path": [
                "MSCOCO_i2t/image_0.jpg",
                "MSCOCO_i2t/image_1.jpg",
            ],
            "tgt_text": [
                ["caption-a", "caption-b"],
                ["caption-b", "caption-c"],
            ],
            "tgt_img_path": [
                ["", ""],
                ["", ""],
            ],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "MSCOCO_i2t" / "image_0.jpg")
    _write_test_image(image_root / "MSCOCO_i2t" / "image_1.jpg")

    output_root = tmp_path / "rerun"
    output_path = convert_image_i2t_dataset(
        "MSCOCO_i2t",
        output_root,
        mmeb_eval_root=annotation_root,
        mmeb_v2_root=mmeb_v2_root,
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert [row["id"] for row in queries] == [0, 1]
    assert queries[0]["qry_img_path"] == "MSCOCO_i2t/image_0.jpg"
    assert isinstance(queries[0]["image"], bytes)

    assert candidates == [
        {"id": 0, "text": "caption-a"},
        {"id": 1, "text": "caption-b"},
        {"id": 2, "text": "caption-c"},
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "exhaustive",
            "candidate_ids": [0, 1],
            "candidate_scores": [1.0, 0.0],
        },
        {
            "query_id": 1,
            "mode": "exhaustive",
            "candidate_ids": [1, 2],
            "candidate_scores": [1.0, 0.0],
        },
    ]


def test_convert_image_t2i_dataset_writes_image_candidates(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "MSCOCO_t2i"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": [
                "Find me image one.\n",
                "Find me image two.\n",
            ],
            "qry_img_path": [
                "",
                "",
            ],
            "tgt_text": [
                ["<|image_1|>\nRepresent image one.\n", "<|image_1|>\nRepresent image two.\n"],
                ["<|image_1|>\nRepresent image two.\n", "<|image_1|>\nRepresent image one.\n"],
            ],
            "tgt_img_path": [
                ["MSCOCO_t2i/image_0.jpg", "MSCOCO_t2i/image_1.jpg"],
                ["MSCOCO_t2i/image_1.jpg", "MSCOCO_t2i/image_0.jpg"],
            ],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "MSCOCO_t2i" / "image_0.jpg")
    _write_test_image(image_root / "MSCOCO_t2i" / "image_1.jpg")

    output_root = tmp_path / "rerun"
    output_path = convert_image_t2i_dataset(
        "MSCOCO_t2i",
        output_root,
        mmeb_eval_root=annotation_root,
        mmeb_v2_root=mmeb_v2_root,
        max_workers=2,
    )

    queries = lance.dataset(str(output_path / "queries.lance")).to_table().to_pylist()
    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert queries[0]["image"] is None
    assert queries[0]["qry_img_path"] == ""
    assert candidates == [
        {
            "id": 0,
            "tgt_text": "<|image_1|>\nRepresent image one.\n",
            "tgt_inst": None,
            "tgt_img_path": "MSCOCO_t2i/image_0.jpg",
            "image": candidates[0]["image"],
        },
        {
            "id": 1,
            "tgt_text": "<|image_1|>\nRepresent image two.\n",
            "tgt_inst": None,
            "tgt_img_path": "MSCOCO_t2i/image_1.jpg",
            "image": candidates[1]["image"],
        },
    ]
    assert isinstance(candidates[0]["image"], bytes)
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


def test_convert_image_i2i_vg_dataset_uses_image_text_candidate_key(tmp_path):
    annotation_root = tmp_path / "annotations"
    dataset_dir = annotation_root / "RefCOCO"
    dataset_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "qry_text": [
                "<|image_1|>\nFind object one.\n",
                "<|image_1|>\nFind object two.\n",
            ],
            "qry_img_path": [
                "RefCOCO/query_0.jpg",
                "RefCOCO/query_1.jpg",
            ],
            "tgt_text": [
                ['<|image_1|>\nCrop "a"\n', '<|image_1|>\nCrop "b"\n'],
                ['<|image_1|>\nCrop "b"\n', '<|image_1|>\nCrop "c"\n'],
            ],
            "tgt_img_path": [
                ["RefCOCO/cand_shared.jpg", "RefCOCO/cand_shared.jpg"],
                ["RefCOCO/cand_shared.jpg", "RefCOCO/cand_2.jpg"],
            ],
        }
    )
    df.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

    mmeb_v2_root = tmp_path / "mmeb-v2"
    image_root = mmeb_v2_root / "image-tasks" / "MMEB"
    _write_test_image(image_root / "RefCOCO" / "query_0.jpg")
    _write_test_image(image_root / "RefCOCO" / "query_1.jpg")
    _write_test_image(image_root / "RefCOCO" / "cand_shared.jpg")
    _write_test_image(image_root / "RefCOCO" / "cand_2.jpg")

    output_root = tmp_path / "rerun"
    output_path = convert_image_i2i_vg_dataset(
        "RefCOCO",
        output_root,
        mmeb_eval_root=annotation_root,
        mmeb_v2_root=mmeb_v2_root,
        max_workers=2,
    )

    candidates = lance.dataset(str(output_path / "candidates.lance")).to_table().to_pylist()
    qrels = lance.dataset(str(output_path / "qrels.lance")).to_table().to_pylist()

    assert [(row["id"], row["tgt_img_path"], row["tgt_text"]) for row in candidates] == [
        (0, "RefCOCO/cand_shared.jpg", '<|image_1|>\nCrop "a"\n'),
        (1, "RefCOCO/cand_shared.jpg", '<|image_1|>\nCrop "b"\n'),
        (2, "RefCOCO/cand_2.jpg", '<|image_1|>\nCrop "c"\n'),
    ]
    assert qrels == [
        {
            "query_id": 0,
            "mode": "exhaustive",
            "candidate_ids": [0, 1],
            "candidate_scores": [1.0, 0.0],
        },
        {
            "query_id": 1,
            "mode": "exhaustive",
            "candidate_ids": [1, 2],
            "candidate_scores": [1.0, 0.0],
        },
    ]


def test_convert_all_image_datasets_runs_all_known_image_families(monkeypatch, tmp_path):
    calls: list[str] = []

    def _capture(dataset_name, output_root, **kwargs):
        del output_root, kwargs
        calls.append(dataset_name)
        return tmp_path / dataset_name

    monkeypatch.setattr("vlm2emb.data.conversion.mmeb_v2.image.convert_image_cls_dataset", _capture)
    monkeypatch.setattr("vlm2emb.data.conversion.mmeb_v2.image.convert_image_qa_dataset", _capture)
    monkeypatch.setattr("vlm2emb.data.conversion.mmeb_v2.image.convert_image_i2t_dataset", _capture)
    monkeypatch.setattr("vlm2emb.data.conversion.mmeb_v2.image.convert_image_t2i_dataset", _capture)
    monkeypatch.setattr("vlm2emb.data.conversion.mmeb_v2.image.convert_image_i2i_vg_dataset", _capture)

    outputs = convert_all_image_datasets(tmp_path / "rerun")

    assert calls
    assert len(calls) == len(outputs)
