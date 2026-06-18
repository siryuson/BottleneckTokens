from __future__ import annotations

from io import BytesIO

import lance
import pyarrow as pa
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN
from vlm2emb.data.datasets.visrag import QUERY_SOURCE_PROMPTS, TARGET_SOURCE_PROMPTS, VisragTrainDataset


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
    image = Image.new("RGB", (2, 2), color=(0, 0, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_visrag_train_artifact(root):
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    lance.write_dataset(
        pa.table(
            {
                "image": [{"bytes": _make_png_bytes(), "path": "page.png"}],
                "query": ["What does the chart show?"],
                "source": ["ChartQA"],
            }
        ),
        data_dir / "train.lance",
    )


def test_visrag_train_dataset_builds_from_config(tmp_path):
    _write_visrag_train_artifact(tmp_path)

    dataset = AutoDataset.from_config(
        {
            "type": "visrag_train",
            "path": str(tmp_path),
        }
    )
    sample = dataset[0]

    assert isinstance(dataset, VisragTrainDataset)
    assert sample["query"]["text"] == f"{QUERY_SOURCE_PROMPTS['ChartQA']}What does the chart show?\n"
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == f"{STANDARD_IMAGE_TOKEN}\n{TARGET_SOURCE_PROMPTS['ChartQA']}\n"
    assert len(sample["positive"]["media"]) == 1
    assert sample["negative"] == {"text": "", "media": []}
    assert sample["metadata"]["dataset_name"] == "visrag_train"
    assert sample["metadata"]["source"] == "ChartQA"


def test_visrag_train_config_transform_mapping_builds_default_transform(tmp_path):
    _write_visrag_train_artifact(tmp_path)

    dataset = AutoDataset.from_config(
        {
            "type": "visrag_train",
            "path": str(tmp_path),
            "transform": {
                "query": {
                    "instructions": {
                        "by_source": {"ChartQA": "Custom query: "},
                        "fallback": "",
                    },
                    "instruction_body_separator": "none",
                },
                "positive": {
                    "instructions": {
                        "by_source": {"ChartQA": "Custom page."},
                        "fallback": "",
                    },
                    "visual_token_placement": "own_line",
                },
                "negative": {"empty": "empty_multimodal_input"},
            },
        }
    )
    sample = dataset[0]

    assert sample["query"]["text"] == "Custom query: What does the chart show?\n"
    assert sample["positive"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nCustom page.\n"


def test_visrag_train_callable_transform_replaces_default_transform(tmp_path):
    _write_visrag_train_artifact(tmp_path)

    dataset = VisragTrainDataset(
        path=str(tmp_path),
        transform=lambda row: {
            "query": {"text": row["query"], "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )

    assert dataset[0]["query"]["text"] == "What does the chart show?"
    assert dataset[0]["positive"]["text"] == "custom"


def test_visrag_train_falsy_callable_transform_replaces_default(tmp_path):
    _write_visrag_train_artifact(tmp_path)

    dataset = VisragTrainDataset(path=str(tmp_path), transform=_FalsyTransform())

    assert dataset[0]["positive"]["text"] == "falsy"


def test_visrag_train_read_columns_override_default_read_set(tmp_path):
    _write_visrag_train_artifact(tmp_path)

    dataset = VisragTrainDataset(
        path=str(tmp_path),
        read_columns=["query"],
        transform=lambda row: {
            "query": {"text": row["query"], "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )

    assert dataset._read_columns == ["query"]
    assert dataset[0]["query"]["text"] == "What does the chart show?"
