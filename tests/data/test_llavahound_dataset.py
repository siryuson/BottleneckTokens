from __future__ import annotations

from io import BytesIO

import lance
import pyarrow as pa
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.conversion.llavahound_train import (
    convert_llavahound_root,
    discover_instruction_subsets,
)
from vlm2emb.data.datasets.const import STANDARD_VIDEO_TOKEN
from vlm2emb.data.datasets.llavahound import LlavaHoundTrainDataset


class _FalsyTransform:
    def __call__(self, row):
        return {
            "query": {"text": row["id"], "media": []},
            "positive": {"text": "falsy", "media": []},
            "negative": {"text": "", "media": []},
        }

    def __bool__(self):
        return False


def _make_png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (2, 2), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_llavahound_fixture(root):
    frame_rows = [
        {"video_id": "video-a", "frame_idx": "frame_0002.jpg", "image": _make_png_bytes((0, 0, 255))},
        {"video_id": "video-a", "frame_idx": "frame_0000.jpg", "image": _make_png_bytes((255, 0, 0))},
        {"video_id": "video-a", "frame_idx": "frame_0001.jpg", "image": _make_png_bytes((0, 255, 0))},
        {"video_id": "video-b", "frame_idx": "frame_0000.jpg", "image": _make_png_bytes((255, 255, 0))},
    ]
    frame_path = root / "data" / "train_300k.lance"
    lance.write_dataset(pa.Table.from_pylist(frame_rows), str(frame_path), mode="overwrite")
    lance.dataset(str(frame_path)).create_scalar_index("video_id", "BTREE")

    instruction_rows = [
        {
            "id": "sample-a",
            "video": "video-a",
            "conversations": [
                {"from": "human", "value": "<video>\nDescribe this video."},
                {"from": "gpt", "value": "A person is cooking."},
            ],
        },
        {
            "id": "sample-b",
            "video": "video-b",
            "conversations": [
                {"from": "human", "value": "<video>\nWhat appears?"},
                {"from": "gpt", "value": "A yellow frame."},
            ],
        },
        {
            "id": "sample-c",
            "video": "video-a",
            "conversations": [
                {"from": "human", "value": "<video>\nDescribe this video again."},
                {"from": "gpt", "value": "A person is still cooking."},
            ],
        },
    ]
    instruction_path = root / "data" / "video_instruction" / "train" / "sft" / "video_caption_300k.lance"
    lance.write_dataset(pa.Table.from_pylist(instruction_rows), str(instruction_path), mode="overwrite")


def test_llavahound_caption_retrieval_runtime_resolves_sampled_frames(tmp_path):
    _write_llavahound_fixture(tmp_path)
    dataset = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
        num_frames=2,
    )

    sample = dataset[0]

    assert sample["query"]["text"] == f"{STANDARD_VIDEO_TOKEN}\nDescribe this video.\n"
    assert len(sample["query"]["media"]) == 1
    assert sample["query"]["media"][0]["kind"] == "video"
    assert len(sample["query"]["media"][0]["content"]) == 2
    assert sample["query"]["media"][0]["metadata"] == {
        "total_num_frames": 3,
        "sampled_indices": [0, 1],
    }
    assert sample["positive"]["text"] == "A person is cooking.\n"
    assert sample["positive"]["media"] == []
    assert sample["negative"] == {"text": "", "media": []}
    assert sample["metadata"]["dataset_name"] == "llavahound/train/sft/video_caption_300k"


def test_llavahound_video_retrieval_runtime_places_video_on_positive_side(tmp_path):
    _write_llavahound_fixture(tmp_path)
    dataset = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
        num_frames=2,
        data_mode="video_retrieval",
    )

    sample = dataset[0]

    assert sample["query"]["text"] == "Find a video that contains the following visual content: A person is cooking.\n"
    assert sample["query"]["media"] == []
    assert sample["positive"]["text"] == f"Understand the content of the provided video: {STANDARD_VIDEO_TOKEN}\n"
    assert len(sample["positive"]["media"][0]["content"]) == 2


def test_llavahound_config_transform_mapping_and_callable_replacement(tmp_path):
    _write_llavahound_fixture(tmp_path)
    dataset = AutoDataset.from_config(
        {
            "type": "llavahound_train",
            "path": str(tmp_path),
            "subset": "train/sft/video_caption_300k",
            "num_frames": 1,
            "transform": {
                "query": {
                    "instruction": "Answer a question based on the content of a video.",
                    "instruction_body_separator": "newline",
                },
                "negative": {"empty": "empty_multimodal_input"},
            },
        }
    )
    mapped_sample = dataset[1]
    assert mapped_sample["query"]["text"] == (
        f"{STANDARD_VIDEO_TOKEN}\nAnswer a question based on the content of a video.\nWhat appears?\n"
    )

    def custom_transform(record):
        return {
            "query": {"text": "custom query", "media": []},
            "positive": {"text": record["id"], "media": []},
            "negative": {"text": "", "media": []},
        }

    replaced = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
        transform=custom_transform,
    )
    assert replaced[0]["query"]["text"] == "custom query"
    assert replaced[0]["positive"]["text"] == "sample-a"


def test_llavahound_falsy_callable_transform_replaces_default(tmp_path):
    _write_llavahound_fixture(tmp_path)

    dataset = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
        transform=_FalsyTransform(),
    )

    assert dataset[0]["positive"]["text"] == "falsy"


def test_llavahound_read_columns_override_default_read_set(tmp_path):
    _write_llavahound_fixture(tmp_path)

    dataset = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
        read_columns=["id"],
        transform=lambda row: {
            "query": {"text": row["id"], "media": []},
            "positive": {"text": "custom", "media": []},
            "negative": {"text": "", "media": []},
        },
    )

    assert dataset._read_columns == ["id"]
    assert dataset[0]["query"]["text"] == "sample-a"


def test_llavahound_getitems_deduplicates_video_frame_lookup_per_batch(tmp_path):
    _write_llavahound_fixture(tmp_path)
    dataset = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
        num_frames=2,
    )
    calls: list[str] = []
    original_get_video_frames = dataset._get_video_frames

    def counting_get_video_frames(video_id):
        calls.append(video_id)
        return original_get_video_frames(video_id)

    dataset._get_video_frames = counting_get_video_frames

    samples = dataset.__getitems__([0, 2])

    assert calls == ["video-a"]
    assert [sample["metadata"]["id"] for sample in samples] == ["sample-a", "sample-c"]
    assert samples[0]["query"]["media"][0]["metadata"] == samples[1]["query"]["media"][0]["metadata"]
    assert samples[0]["query"]["media"][0]["content"] is not samples[1]["query"]["media"][0]["content"]


def test_llavahound_missing_video_frames_raise_clear_error(tmp_path):
    _write_llavahound_fixture(tmp_path)
    dataset = LlavaHoundTrainDataset(
        path=str(tmp_path),
        subset="train/sft/video_caption_300k",
    )

    rows = dataset.take_rows([0])
    rows[0]["video"] = "missing-video"
    resolved = dataset._resolve_records(rows)

    try:
        dataset._transform_record(resolved[0])
    except ValueError as err:
        assert "missing required video frames" in str(err)
    else:
        raise AssertionError("Expected missing video frames to fail at runtime")


def test_llavahound_conversion_copies_raw_tables_and_discovers_subsets(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_llavahound_fixture(source)

    assert discover_instruction_subsets(source) == ("train/sft/video_caption_300k",)
    summary = convert_llavahound_root(
        source,
        output,
        instruction_subsets=("train/sft/video_caption_300k",),
        batch_size=2,
    )

    assert summary.frames.rows == 4
    assert summary.instructions[0].rows == 3
    frame_ds = lance.dataset(summary.frames.output_path)
    instruction_ds = lance.dataset(summary.instructions[0].output_path)
    assert frame_ds.schema.names == ["video_id", "frame_idx", "image"]
    assert instruction_ds.schema.names == ["id", "video", "conversations"]
    assert frame_ds.count_rows() == 4
    assert instruction_ds.count_rows() == 3
    assert any("video_id" in index.field_names for index in frame_ds.describe_indices())


def test_llavahound_conversion_rejects_same_input_output_table(tmp_path):
    _write_llavahound_fixture(tmp_path)

    try:
        convert_llavahound_root(
            tmp_path,
            tmp_path,
            instruction_subsets=("train/sft/video_caption_300k",),
        )
    except ValueError as err:
        assert "must differ" in str(err)
    else:
        raise AssertionError("Expected same input/output LLaVA-Hound conversion to fail")
