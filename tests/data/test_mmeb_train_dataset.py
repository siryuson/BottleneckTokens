from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.collators.training_collator import TrainingCollator
from vlm2emb.data.datasets.mmeb_train import MmebTrainDataset


def _make_png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (2, 2), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_mmeb_train_artifact(
    root: Path,
    *,
    subset: str,
    rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
) -> None:
    sample_path = root / "data" / subset / "train.lance"
    image_path = root / "data" / "images" / f"{subset}.lance"
    sample_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    lance.write_dataset(pa.Table.from_pylist(rows), str(sample_path))
    lance.write_dataset(
        pa.Table.from_pylist(
            image_rows,
            schema=pa.schema([
                pa.field("path", pa.string()),
                pa.field("image", pa.binary()),
            ]),
        ),
        str(image_path),
    )


def test_mmeb_train_dataset_joins_path_image_table_and_outputs_train_sample(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    dataset = MmebTrainDataset(path=str(tmp_path), subset="ImageNet_1K")
    sample = dataset[0]

    assert sample["query"]["text"] == "<|image_pad|>\nRepresent the given image for classification\n"
    assert sample["query"]["media"][0]["kind"] == "image"
    assert sample["query"]["media"][0]["metadata"]["path"] == "images/ImageNet_1K/Train/image_0.jpg"
    assert sample["query"]["media"][0]["content"].mode == "RGB"
    assert sample["positive"] == {"text": "plane", "media": []}
    assert sample["negative"] == {"text": "", "media": []}
    assert sample["metadata"]["dataset_name"] == "mmeb_train/ImageNet_1K"
    assert sample["metadata"]["subset"] == "ImageNet_1K"
    assert sample["metadata"]["split"] == "train"


def test_mmeb_train_dataset_skips_empty_image_paths_without_lookup(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="WebQA",
        rows=[
            {
                "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: Which body part?\n",
                "qry_image_path": "",
                "pos_text": "<|image_1|>\nRepresent the given Wikipedia image.\n",
                "pos_image_path": "images/WebQA/Train/30272327.jpg",
            }
        ],
        image_rows=[
            {
                "path": "images/WebQA/Train/30272327.jpg",
                "image": _make_png_bytes((0, 255, 0)),
            }
        ],
    )

    sample = MmebTrainDataset(path=str(tmp_path), subset="WebQA")[0]

    assert sample["query"]["text"] == "Find a Wikipedia image that answers this question: Which body part?\n"
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == "<|image_pad|>\nRepresent the given Wikipedia image.\n"
    assert len(sample["positive"]["media"]) == 1
    assert sample["negative"] == {"text": "", "media": []}


def test_mmeb_train_dataset_trims_chartqa_answer_tail_spaces(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ChartQA",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image with the following question: Value?\n",
                "qry_image_path": "images/ChartQA/Train/ChartQA_image_0.jpg",
                "pos_text": "10.5 ",
                "pos_image_path": "",
                "neg_text": "7.2 ",
                "neg_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ChartQA/Train/ChartQA_image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    sample = MmebTrainDataset(path=str(tmp_path), subset="ChartQA")[0]

    assert sample["positive"] == {"text": "10.5", "media": []}
    assert sample["negative"] == {"text": "7.2", "media": []}


def test_mmeb_train_dataset_custom_transform_replaces_default_transform(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    def custom_transform(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": {"text": f"custom {record['qry']}", "media": []},
            "positive": {"text": "custom positive", "media": []},
            "negative": {"text": "", "media": []},
        }

    sample = MmebTrainDataset(
        path=str(tmp_path),
        subset="ImageNet_1K",
        transform=custom_transform,
    )[0]

    assert sample["query"]["text"].startswith("custom <|image_1|>")
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == "custom positive"


def test_mmeb_train_dataset_falsy_custom_transform_still_replaces_default(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    class FalsyTransform:
        def __bool__(self) -> bool:
            return False

        def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
            return {
                "query": {"text": f"falsy {record['qry']}", "media": []},
                "positive": {"text": "falsy positive", "media": []},
                "negative": {"text": "", "media": []},
            }

    sample = MmebTrainDataset(
        path=str(tmp_path),
        subset="ImageNet_1K",
        transform=FalsyTransform(),
    )[0]

    assert sample["query"]["text"].startswith("falsy <|image_1|>")
    assert sample["positive"]["text"] == "falsy positive"


def test_mmeb_train_dataset_read_columns_override_default_read_set(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
                "extra": "kept",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    dataset = MmebTrainDataset(
        path=str(tmp_path),
        subset="ImageNet_1K",
        read_columns=["extra"],
        transform=lambda row: {
            "query": {"text": row["extra"], "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )
    sample = dataset[0]

    assert dataset._read_columns == ["extra"]
    assert sample["query"] == {"text": "kept", "media": []}
    assert sample["positive"] == {"text": "custom", "media": []}


def test_mmeb_train_custom_transform_can_use_joined_image_side_table(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    sample = MmebTrainDataset(
        path=str(tmp_path),
        subset="ImageNet_1K",
        read_columns=["qry_image_path"],
        transform=lambda row: {
            "query": {"text": "<|image_pad|>", "media": [{"kind": "image", "content": row["query_image"]}]},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )[0]

    assert isinstance(sample["query"]["media"][0]["content"], bytes)


def test_mmeb_train_dataset_sample_metadata_overrides_dataset_metadata(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    sample = MmebTrainDataset(
        path=str(tmp_path),
        subset="ImageNet_1K",
        metadata={
            "dataset_name": "override",
            "subset": "override",
            "custom": "dataset",
        },
    )[0]

    assert sample["metadata"]["dataset_name"] == "mmeb_train/ImageNet_1K"
    assert sample["metadata"]["subset"] == "ImageNet_1K"
    assert sample["metadata"]["custom"] == "dataset"


def test_mmeb_train_dataset_reports_missing_joined_image(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/missing.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[],
    )

    dataset = MmebTrainDataset(path=str(tmp_path), subset="ImageNet_1K")

    try:
        dataset[0]
    except FileNotFoundError as err:
        assert "path=images/ImageNet_1K/Train/missing.jpg" in str(err)
    else:
        raise AssertionError("Expected missing joined image to fail during transform")


def test_mmeb_train_dataset_is_registered_without_lance_prefix(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    dataset = AutoDataset.from_config(
        {
            "type": "mmeb_train",
            "path": str(tmp_path),
            "subset": "ImageNet_1K",
        }
    )

    assert isinstance(dataset, MmebTrainDataset)
    assert dataset[0]["metadata"]["dataset_name"] == "mmeb_train/ImageNet_1K"


def test_mmeb_train_dataset_accepts_declarative_transform_kwargs(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="WebQA",
        rows=[
            {
                "qry": "<|image_1|>\nFind a Wikipedia image that answers this question.\n",
                "qry_image_path": "",
                "pos_text": "<|image_1|>\nRepresent the given Wikipedia image.\n",
                "pos_image_path": "images/WebQA/Train/image_0.jpg",
            }
        ],
        image_rows=[
            {
                "path": "images/WebQA/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    dataset = AutoDataset.from_config(
        {
            "type": "mmeb_train",
            "path": str(tmp_path),
            "subset": "WebQA",
            "transform": {
                "positive": {
                    "trailing_newline": "strip",
                },
            },
        }
    )

    assert dataset[0]["positive"]["text"] == "<|image_pad|>\nRepresent the given Wikipedia image."


def test_mmeb_train_dataset_rejects_non_callable_runtime_transform(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    try:
        MmebTrainDataset(
            path=str(tmp_path),
            subset="ImageNet_1K",
            transform={"query": {"trailing_newline": "strip"}},
        )
    except TypeError as err:
        assert "transform must be a callable SampleTransform or None" in str(err)
    else:
        raise AssertionError("Expected non-callable runtime transform to fail fast")


def test_mmeb_train_dataset_combined_config_treats_transform_mapping_as_kwargs(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="WebQA",
        rows=[
            {
                "qry": "<|image_1|>\nFind a Wikipedia image that answers this question.\n",
                "qry_image_path": "",
                "pos_text": "<|image_1|>\nRepresent the given Wikipedia image.\n",
                "pos_image_path": "images/WebQA/Train/image_0.jpg",
            }
        ],
        image_rows=[
            {
                "path": "images/WebQA/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    combined = AutoDataset.build(
        "combined",
        datasets={
            "webqa": {
                "type": "mmeb_train",
                "path": str(tmp_path),
                "subset": "WebQA",
                "transform": {
                    "positive": {
                        "trailing_newline": "strip",
                    },
                },
            }
        },
    )

    assert combined[0]["positive"]["text"] == "<|image_pad|>\nRepresent the given Wikipedia image."


def test_mmeb_train_dataset_config_callable_transform_replaces_default(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )

    def custom_transform(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": {"text": f"configured {record['qry']}", "media": []},
            "positive": {"text": "configured positive", "media": []},
            "negative": {"text": "", "media": []},
        }

    dataset = AutoDataset.from_config(
        {
            "type": "mmeb_train",
            "path": str(tmp_path),
            "subset": "ImageNet_1K",
            "transform": custom_transform,
        }
    )

    sample = dataset[0]

    assert sample["query"]["text"].startswith("configured <|image_1|>")
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == "configured positive"


def test_mmeb_train_dataset_empty_transform_clears_negative(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="WebQA",
        rows=[
            {
                "qry": "<|image_1|>\nFind a Wikipedia image that answers this question.\n",
                "qry_image_path": "",
                "pos_text": "<|image_1|>\nRepresent the given Wikipedia image.\n",
                "pos_image_path": "images/WebQA/Train/image_0.jpg",
                "neg_text": "should be cleared",
                "neg_image_path": "images/WebQA/Train/negative.jpg",
            }
        ],
        image_rows=[
            {
                "path": "images/WebQA/Train/image_0.jpg",
                "image": _make_png_bytes(),
            },
            {
                "path": "images/WebQA/Train/negative.jpg",
                "image": _make_png_bytes((0, 0, 255)),
            },
        ],
    )

    sample = MmebTrainDataset(path=str(tmp_path), subset="WebQA")[0]

    assert sample["negative"] == {"text": "", "media": []}


def test_mmeb_train_dataset_media_reaches_training_collator(tmp_path) -> None:
    _write_mmeb_train_artifact(
        tmp_path,
        subset="ImageNet_1K",
        rows=[
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            }
        ],
        image_rows=[
            {
                "path": "images/ImageNet_1K/Train/image_0.jpg",
                "image": _make_png_bytes(),
            }
        ],
    )
    captured: list[list[dict[str, Any]]] = []

    class DummyWrapper:
        def process_multimodal_batch(self, payloads):
            captured.append(payloads)
            return [
                {"text": payload["text"], "media_count": len(payload.get("media", []))}
                for payload in payloads
            ]

    sample = MmebTrainDataset(path=str(tmp_path), subset="ImageNet_1K")[0]
    outputs = TrainingCollator(wrapper=DummyWrapper())([sample])

    assert outputs["query"][0]["media_count"] == 1
    assert outputs["positive"][0]["media_count"] == 0
    assert outputs["negative"] == [None]
    assert captured[0][0]["text"] == "<|image_pad|>\nRepresent the given image for classification\n"
    assert isinstance(captured[0][0]["media"][0]["content"], Image.Image)
