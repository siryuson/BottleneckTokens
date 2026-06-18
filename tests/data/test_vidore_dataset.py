from __future__ import annotations

from io import BytesIO

import lance
import pyarrow as pa
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN
from vlm2emb.data.datasets.vidore import VidoreTrainDataset
from vlm2emb.data.datasets.vidore_rag import VidoreRagTrainDataset


class _FalsyTransform:
    def __call__(self, row):
        return {
            "query": {"text": row["query"], "media": []},
            "positive": {"text": "falsy", "media": []},
            "negative": {"text": "", "media": []},
        }

    def __bool__(self):
        return False


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(0, 255, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_vidore_train_artifact(root):
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    lance.write_dataset(
        pa.table(
            {
                "image": [{"bytes": _make_png_bytes(), "path": "page.png"}],
                "query": ["find the relevant passage"],
                "answer": ["The answer is on this page."],
                "source": ["docvqa"],
                "options": [None],
                "image_filename": ["page.png"],
                "page": ["1"],
                "model": ["test"],
                "prompt": [None],
                "answer_type": ["text"],
            }
        ),
        data_dir / "train.lance",
    )


def _write_minimal_vidore_train_artifact(root):
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    lance.write_dataset(
        pa.table(
            {
                "image": [{"bytes": _make_png_bytes(), "path": "page.png"}],
                "query": ["find the relevant passage"],
                "answer": ["The answer is on this page."],
            }
        ),
        data_dir / "train.lance",
    )


def test_vidore_train_dataset_builds_from_direct_init(tmp_path):
    _write_vidore_train_artifact(tmp_path)

    dataset = VidoreTrainDataset(path=str(tmp_path))
    sample = dataset[0]

    assert sample["query"]["text"].startswith(STANDARD_IMAGE_TOKEN)
    assert "Options:" not in sample["query"]["text"]
    assert len(sample["query"]["media"]) == 1
    assert sample["positive"]["text"] == "The answer is on this page."
    assert sample["negative"] == {"text": "", "media": []}
    assert sample["metadata"]["dataset_name"] == "vidore_train"
    assert sample["metadata"]["source"] == "docvqa"
    assert dataset.path == str(tmp_path)
    assert dataset.split == "train"


def test_vidore_train_dataset_from_config_keeps_public_entry(tmp_path):
    _write_vidore_train_artifact(tmp_path)

    dataset = AutoDataset.from_config(
        {
            "type": "vidore_train",
            "path": str(tmp_path),
        }
    )
    sample = dataset[0]

    assert isinstance(dataset, VidoreTrainDataset)
    assert sample["metadata"]["dataset_name"] == "vidore_train"


def test_vidore_train_transform_replaces_default_transform(tmp_path):
    _write_vidore_train_artifact(tmp_path)

    dataset = VidoreTrainDataset(
        path=str(tmp_path),
        transform=lambda row: {
            "query": {"text": str(row["query"]), "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )

    assert dataset[0]["query"]["text"] == "find the relevant passage"
    assert dataset[0]["positive"]["text"] == "custom"


def test_vidore_train_falsy_callable_transform_replaces_default(tmp_path):
    _write_vidore_train_artifact(tmp_path)

    dataset = VidoreTrainDataset(path=str(tmp_path), transform=_FalsyTransform())

    assert dataset[0]["positive"]["text"] == "falsy"


def test_vidore_rag_config_transform_mapping_builds_default_transform(tmp_path):
    _write_vidore_train_artifact(tmp_path)

    dataset = AutoDataset.from_config(
        {
            "type": "vidore_rag_train",
            "path": str(tmp_path),
            "transform": {
                "query": {
                    "instruction": "Retrieve:",
                    "instruction_body_separator": "space",
                },
                "positive": {
                    "instruction": "Read the document.",
                    "visual_token_placement": "own_line",
                },
                "negative": {"empty": "empty_multimodal_input"},
            },
        }
    )
    sample = dataset[0]

    assert sample["query"]["text"] == "Retrieve: find the relevant passage\n"
    assert sample["positive"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nRead the document.\n"


def test_vidore_full_config_transform_mapping_uses_side_scoped_rules(tmp_path):
    _write_vidore_train_artifact(tmp_path)

    dataset = AutoDataset.from_config(
        {
            "type": "vidore_train",
            "path": str(tmp_path),
            "transform": {
                "query": {
                    "visual_token_placement": "own_line",
                    "trailing_newline": "ensure_single",
                },
                "positive": {"trailing_newline": "ensure_single"},
                "negative": {"empty": "empty_multimodal_input"},
            },
        }
    )
    sample = dataset[0]

    assert sample["query"]["text"].endswith("\n")
    assert sample["positive"]["text"] == "The answer is on this page.\n"


def test_vidore_train_read_columns_override_default_read_set(tmp_path):
    _write_minimal_vidore_train_artifact(tmp_path)

    dataset = VidoreTrainDataset(
        path=str(tmp_path),
        read_columns=["query"],
        transform=lambda row: {
            "query": {"text": row["query"], "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )
    sample = dataset[0]

    assert dataset._read_columns == ["query"]
    assert sample["query"] == {"text": "find the relevant passage", "media": []}
    assert sample["positive"] == {"text": "custom", "media": []}


def test_vidore_rag_read_columns_override_default_read_set(tmp_path):
    _write_minimal_vidore_train_artifact(tmp_path)

    dataset = VidoreRagTrainDataset(
        path=str(tmp_path),
        read_columns=["query"],
        transform=lambda row: {
            "query": {"text": row["query"], "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )
    sample = dataset[0]

    assert dataset._read_columns == ["query"]
    assert sample["query"] == {"text": "find the relevant passage", "media": []}
    assert sample["positive"] == {"text": "custom", "media": []}
