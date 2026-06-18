from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from vlm2emb.data.conversion.refcoco_train import (
    SUPPORTED_SPLITS,
    collect_image_ids,
    ensure_refcoco_indices,
    load_refcoco_split_rows,
    write_refcoco_root,
)
from vlm2emb.data.datasets.refcoco_train import RefcocoTrainDataset


def _make_image_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (12, 10), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_refcoco_fixture(root: Path) -> tuple[Path, Path]:
    parquet_root = root / "refcoco"
    data_root = parquet_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    base_row = {
        "sent_ids": [1, 2],
        "file_name": "COCO_train2014_000000000042_1.jpg",
        "ann_id": 1001,
        "ref_id": 2001,
        "image_id": 42,
        "split": "train",
        "sentences": [{"raw": "red square", "sent": "red square", "sent_id": 1, "tokens": ["red", "square"]}],
        "category_id": 1,
        "raw_anns": "{}",
        "raw_image_info": "{}",
        "raw_sentences": "[]",
        "image_path": "coco/train2014/COCO_train2014_000000000042.jpg",
        "bbox": [2.0, 1.0, 8.0, 7.0],
        "captions": ["red square", "red object"],
        "global_image_id": "coco.42",
        "anns_id": "refcoco.1001",
    }
    split_files = {
        "train-00000-of-00001.parquet": [base_row],
        "validation-00000-of-00001.parquet": [{**base_row, "ann_id": 1002, "ref_id": 2002, "split": "val"}],
        "test-00000-of-00001.parquet": [{**base_row, "ann_id": 1003, "ref_id": 2003, "split": "testA"}],
        "testB-00000-of-00001.parquet": [{**base_row, "ann_id": 1004, "ref_id": 2004, "split": "testB"}],
    }
    for filename, rows in split_files.items():
        pq.write_table(pa.Table.from_pylist(rows), data_root / filename)

    coco_root = root / "coco"
    coco_data_root = coco_root / "data"
    coco_data_root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": 42,
                    "image": {"bytes": _make_image_bytes(), "path": "COCO_train2014_000000000042.jpg"},
                }
            ]
        ),
        coco_data_root / "train-00000-of-00001.parquet",
    )
    return parquet_root, coco_root


def test_load_refcoco_split_rows_preserves_hf_train_view(tmp_path: Path):
    parquet_root, _ = _write_refcoco_fixture(tmp_path)

    rows_by_split = load_refcoco_split_rows(parquet_root)

    assert tuple(rows_by_split) == SUPPORTED_SPLITS
    assert rows_by_split["official_train"][0]["ann_id"] == 1001
    assert rows_by_split["official_train"][0]["captions"] == ["red square", "red object"]
    assert rows_by_split["official_train_without_mmeb_v2_eval"][0]["bbox"] == [2.0, 1.0, 8.0, 7.0]


def test_collect_image_ids_uses_all_split_rows(tmp_path: Path):
    parquet_root, _ = _write_refcoco_fixture(tmp_path)

    rows_by_split = load_refcoco_split_rows(parquet_root)

    assert collect_image_ids(rows_by_split) == {42}


def test_write_refcoco_root_writes_raw_layout_and_runtime_sample(tmp_path: Path):
    parquet_root, coco_root = _write_refcoco_fixture(tmp_path)
    output_root = tmp_path / "RefCOCO"

    payload = write_refcoco_root(
        parquet_root=parquet_root,
        coco_images_root=coco_root,
        output_root=output_root,
        image_batch_size=1,
    )

    assert payload["image_rows"] == 1
    assert payload["splits"]["official_train_without_mmeb_v2_eval"] == 1
    assert payload["exclusion_rows"] == 0
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    images = lance.dataset(str(output_root / "data" / "images.lance"))
    train = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    exclusions = lance.dataset(str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"))
    assert images.count_rows() == 1
    assert train.schema.names == [
        "ann_id",
        "ref_id",
        "image_id",
        "split",
        "sent_ids",
        "captions",
        "bbox",
        "category_id",
        "file_name",
        "image_path",
        "global_image_id",
        "anns_id",
        "raw_anns",
        "raw_image_info",
        "raw_sentences",
    ]
    assert exclusions.count_rows() == 0
    assert any("image_id" in index.field_names for index in images.describe_indices())

    dataset = RefcocoTrainDataset(path=str(output_root))
    sample = dataset[0]

    assert sample["query"]["text"] == (
        '<|image_pad|>\nSelect the portion of the image that follows the language expressions: "red square"\n'
    )
    assert sample["positive"]["text"] == "<|image_pad|>\nRepresent the given cropped image of the object.\n"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["dataset_name"] == "RefCOCO"
    assert sample["metadata"]["ann_id"] == 1001
    assert sample["query"]["media"][0]["content"].size == (12, 10)
    assert sample["positive"]["media"][0]["content"].size == (6, 6)

    matching_dataset = RefcocoTrainDataset(
        path=str(output_root),
        dataset_name="RefCOCO-Matching",
        transform_kwargs={
            "query": {
                "instruction": "Select the portion of the image that follows the language expressions:",
                "caption_selection": "first",
            },
            "positive": {
                "instruction": "Select the portion of the image that follows the language expressions:",
                "caption_selection": "first",
            },
        },
    )
    matching_sample = matching_dataset[0]

    assert matching_sample["positive"]["text"] == (
        '<|image_pad|>\nSelect the portion of the image that follows the language expressions: "red square"\n'
    )
    assert matching_sample["metadata"]["dataset_name"] == "RefCOCO-Matching"
    assert matching_sample["positive"]["media"][0]["content"].size == (6, 6)


def test_write_refcoco_root_can_read_hf_refcoco_tar_media(tmp_path: Path):
    parquet_root, coco_root = _write_refcoco_fixture(tmp_path)
    output_root = tmp_path / "RefCOCO"
    tar_path = coco_root / "refcoco.tar.gz"
    payload = _make_image_bytes(color=(0, 255, 0))
    info = tarfile.TarInfo("refcoco/images/train2014/COCO_train2014_000000000042.jpg")
    info.size = len(payload)
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.addfile(info, BytesIO(payload))

    summary = write_refcoco_root(
        parquet_root=parquet_root,
        coco_images_root=coco_root,
        output_root=output_root,
        image_batch_size=1,
    )

    assert summary["image_rows"] == 1
    images = lance.dataset(str(output_root / "data" / "images.lance"))
    row = images.to_table().to_pylist()[0]
    assert row["image_id"] == 42
    assert row["image_path"] == "refcoco/images/train2014/COCO_train2014_000000000042.jpg"


def test_ensure_refcoco_indices_requires_image_table(tmp_path: Path):
    output_root = tmp_path / "missing"
    try:
        ensure_refcoco_indices(output_root)
    except FileNotFoundError as error:
        assert "images.lance" in str(error)
    else:
        raise AssertionError("ensure_refcoco_indices should fail when images.lance is missing")
