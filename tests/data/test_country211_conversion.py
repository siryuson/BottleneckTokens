from __future__ import annotations

import io
import tarfile
from pathlib import Path

import lance
import numpy as np
from PIL import Image

from vlm2emb.data.conversion.country211_train import (
    SUPPORTED_SPLITS,
    load_country211_split_rows,
    write_country211_root,
)
from vlm2emb.data.datasets.country211_train import Country211TrainDataset


def _make_jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _add_tar_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _write_country211_fixture(raw_root: Path) -> None:
    (raw_root / "train").mkdir(parents=True, exist_ok=True)
    (raw_root / "test").mkdir(parents=True, exist_ok=True)
    (raw_root / "classnames.txt").write_text("Andorra\nBrazil\n", encoding="utf-8")
    (raw_root / "train" / "nshards.txt").write_text("1\n", encoding="utf-8")
    (raw_root / "test" / "nshards.txt").write_text("1\n", encoding="utf-8")
    with tarfile.open(raw_root / "train" / "0.tar", "w") as tar:
        _add_tar_bytes(tar, "s0000000.cls", b"0")
        _add_tar_bytes(tar, "s0000000.jpg", _make_jpeg_bytes((255, 0, 0)))
        _add_tar_bytes(tar, "s0000001.cls", b"1")
        _add_tar_bytes(tar, "s0000001.jpg", _make_jpeg_bytes((0, 255, 0)))
    with tarfile.open(raw_root / "test" / "0.tar", "w") as tar:
        _add_tar_bytes(tar, "s0000000.cls", b"1")
        _add_tar_bytes(tar, "s0000000.jpg", _make_jpeg_bytes((0, 0, 255)))


def test_load_country211_split_rows_keeps_tar_metadata(tmp_path: Path):
    raw_root = tmp_path / "country211"
    _write_country211_fixture(raw_root)

    split_rows = load_country211_split_rows(raw_root)

    assert tuple(split_rows) == SUPPORTED_SPLITS
    assert len(split_rows["official_train"]) == 2
    assert len(split_rows["official_test"]) == 1
    assert split_rows["official_train"][1]["class_name"] == "Brazil"
    assert split_rows["official_train"][1]["source_tar"] == "train/0.tar"
    assert split_rows["official_train_without_mmeb_v2_eval"] == split_rows["official_train"]


def test_write_country211_root_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "country211"
    output_root = tmp_path / "Country211"
    _write_country211_fixture(raw_root)

    summary = write_country211_root(raw_root=raw_root, output_root=output_root)

    assert summary["image_rows"] == 3
    assert summary["splits"]["official_train"] == 2
    assert summary["splits"]["official_test"] == 1
    assert not (output_root / "manifest.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    assert image_ds.schema.names == ["path", "image"]
    assert train_ds.schema.names == ["image_path", "source_tar", "sample_key", "class_index", "class_name"]
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = Country211TrainDataset(path=str(output_root), split="official_train_without_mmeb_v2_eval")
    sample = dataset[1]

    assert sample["query"]["text"] == "<|image_pad|>\nIdentify the country depicted in the image\n"
    assert sample["positive"]["text"] == "Brazil"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["class_index"] == 1
    assert sample["metadata"]["source_tar"] == "train/0.tar"
    assert sample["query"]["media"][0]["kind"] == "image"
