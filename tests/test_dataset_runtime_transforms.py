from __future__ import annotations

import pickle
from io import BytesIO
from types import SimpleNamespace

import lance
import pandas as pd
import pyarrow as pa
import torch
from PIL import Image
from scripts.convert.train.convert_mmeb_train_to_lance import (
    VARIANT_MAP,
    convert_metadata,
    convert_mmeb_train_root,
)

from vlm2emb.auto import AutoDataset
from vlm2emb.data.collators.training_collator import TrainingCollator
from vlm2emb.data.datasets import CombinedDataset
from vlm2emb.data.datasets.base import (
    LanceDataset,
    VideoEvalLanceDataset,
)
from vlm2emb.data.datasets.const import (
    STANDARD_IMAGE_TOKEN,
    STANDARD_VIDEO_TOKEN,
    validate_runtime_sample,
)
from vlm2emb.data.schema import MediaInput, MultiModalInput, TrainSample
from vlm2emb.data.datasets.llavahound import (
    SUPPORTED_TASK_ARCHETYPES as LLAVAHOUND_ARCHETYPES,
)
from vlm2emb.data.datasets.llavahound import (
    VRET_QUERY_PROMPT,
    VRET_TARGET_PROMPT,
    LlavaHoundTrainDataset,
    build_llavahound_train_default_transform,
    transform_llavahound_train_sample,
)
from vlm2emb.data.datasets.gqa_train import GqaTrainDataset
from vlm2emb.data.datasets.mmeb_train import (
    SUPPORTED_TASK_ARCHETYPES as MMEB_ARCHETYPES,
)
from vlm2emb.data.datasets.mmlongbench_doc_train import (
    build_mmlongbench_doc_train_default_transform,
)
from vlm2emb.data.datasets.vidore import (
    DATASET_ARCHETYPE as VIDORE_ARCHETYPE,
)
from vlm2emb.data.datasets.vidore import (
    build_vidore_train_default_transform,
    transform_vidore_answer,
    transform_vidore_train_sample,
)
from vlm2emb.data.datasets.vidore_rag import (
    DATASET_ARCHETYPE as VIDORE_RAG_ARCHETYPE,
)
from vlm2emb.data.datasets.vidore_rag import (
    QUERY_INSTRUCTION as VIDORE_RAG_QUERY_INSTRUCTION,
)
from vlm2emb.data.datasets.vidore_rag import (
    TARGET_INSTRUCTION as VIDORE_RAG_TARGET_INSTRUCTION,
)
from vlm2emb.data.datasets.vidore_rag import (
    build_vidore_rag_train_default_transform,
    transform_vidore_rag_train_sample,
)
from vlm2emb.data.datasets.visrag import (
    DATASET_ARCHETYPE as VISRAG_ARCHETYPE,
)
from vlm2emb.data.datasets.visrag import (
    QUERY_SOURCE_PROMPTS,
    TARGET_SOURCE_PROMPTS,
    build_visrag_train_default_transform,
    transform_visrag_train_sample,
)
from vlm2emb.data.datasets.videochat2_it_train import (
    build_videochat2_it_query_text,
    transform_videochat2_it_train_sample,
    VideoChat2ItTransformConfig,
)
from vlm2emb.data.processors.qwen2_vl import Qwen2VLProcessorWrapper


class FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace()
        self.image_processor = SimpleNamespace(min_pixels=3136, max_pixels=1003520)
        self.video_processor = SimpleNamespace(
            min_pixels=3136,
            max_pixels=602112,
            size={"shortest_edge": 3136, "longest_edge": 602112},
        )

    def __call__(self, text, images=None, videos=None, return_tensors="pt", **kwargs):
        assert return_tensors == "pt"
        result = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }
        if images is not None:
            result["pixel_values"] = torch.ones(1, 3, 2, 2)
            result["image_grid_thw"] = torch.tensor([[1, 2, 2]], dtype=torch.long)
        if videos is not None:
            result["pixel_values_videos"] = torch.ones(1, 3, 2, 2)
            result["video_grid_thw"] = torch.tensor([[2, 2, 2]], dtype=torch.long)
        return result


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_archive_alignment_loaders_are_registered():
    assert AutoDataset.is_registered("vidore_train")
    assert AutoDataset.is_registered("vidore_rag_train")
    assert AutoDataset.is_registered("visrag_train")


def test_convert_mmeb_train_metadata_preserves_raw_rows_without_manifest(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    dataset_name = "ImageNet_1K"
    dataset_input_dir = input_dir / dataset_name
    dataset_input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "qry": ["find the cat"],
            "qry_image_path": ["images/query.jpg"],
            "pos_text": ["tabby cat"],
            "pos_image_path": ["images/positive.jpg"],
            "neg_text": [""],
            "neg_image_path": [""],
        }
    ).to_parquet(dataset_input_dir / VARIANT_MAP["train"])

    split_summaries, image_references = convert_metadata(input_dir, output_dir, dataset_name)

    rows = (
        lance.dataset(str(output_dir / "data" / dataset_name / "train.lance"))
        .to_table()
        .to_pylist()
    )
    assert len(split_summaries) == 1
    assert split_summaries[0].name == "train"
    assert len(image_references) == 2
    assert rows[0]["qry"] == "find the cat"
    assert rows[0]["qry_image_path"] == "images/query.jpg"
    assert rows[0]["pos_image_path"] == "images/positive.jpg"
    assert set(rows[0]) == {
        "qry",
        "qry_image_path",
        "pos_text",
        "pos_image_path",
        "neg_text",
        "neg_image_path",
    }
    assert "relation_id" not in rows[0]
    assert "sample_id" not in rows[0]
    assert "source_split" not in rows[0]
    assert not (output_dir / "data" / dataset_name / "manifest.json").exists()


def test_convert_mmeb_train_metadata_drops_known_empty_positive_rows_narrowly(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    dataset_name = "OK-VQA"
    dataset_input_dir = input_dir / dataset_name
    dataset_input_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "qry": ["valid query", "bad query"],
            "qry_image_path": ["images/query.jpg", "images/bad-query.jpg"],
            "pos_text": ["valid answer", ""],
            "pos_image_path": ["", ""],
            "neg_text": ["negative", "negative"],
            "neg_image_path": ["", ""],
        }
    ).to_parquet(dataset_input_dir / VARIANT_MAP["train"])

    split_summaries, image_references = convert_metadata(input_dir, output_dir, dataset_name)

    rows = (
        lance.dataset(str(output_dir / "data" / dataset_name / "train.lance"))
        .to_table()
        .to_pylist()
    )
    assert split_summaries[0].rows == 1
    assert split_summaries[0].dropped_rows == 1
    assert rows == [{
        "qry": "valid query",
        "qry_image_path": "images/query.jpg",
        "pos_text": "valid answer",
        "pos_image_path": "",
        "neg_text": "negative",
        "neg_image_path": "",
    }]
    assert [reference.path for reference in image_references] == ["images/query.jpg"]


def test_convert_mmeb_train_metadata_rejects_empty_positive_outside_known_subsets(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    dataset_name = "ImageNet_1K"
    dataset_input_dir = input_dir / dataset_name
    dataset_input_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "qry": ["bad query"],
            "qry_image_path": ["images/bad-query.jpg"],
            "pos_text": [""],
            "pos_image_path": [""],
            "neg_text": [""],
            "neg_image_path": [""],
        }
    ).to_parquet(dataset_input_dir / VARIANT_MAP["train"])

    try:
        convert_metadata(input_dir, output_dir, dataset_name)
    except ValueError as err:
        message = str(err)
        assert "outside the narrow cleanup allowlist" in message
        assert f"subset={dataset_name}" in message
        assert "row_indices=[0]" in message
    else:
        raise AssertionError("Expected unknown MMEB empty-positive row to fail conversion")


def test_convert_mmeb_train_root_writes_raw_tables_and_path_image_side_table(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    dataset_name = "ImageNet_1K"
    dataset_input_dir = input_dir / dataset_name
    dataset_input_dir.mkdir(parents=True)

    image_dir = input_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "query.jpg").write_bytes(_make_png_bytes())
    (image_dir / "positive.jpg").write_bytes(_make_png_bytes())

    pd.DataFrame(
        {
            "qry": ["find the cat"],
            "qry_image_path": ["images/query.jpg"],
            "pos_text": ["tabby cat"],
            "pos_image_path": ["images/positive.jpg"],
            "neg_text": [""],
            "neg_image_path": [""],
        }
    ).to_parquet(dataset_input_dir / VARIANT_MAP["train"])
    pd.DataFrame(
        {
            "qry": ["find the dog"],
            "qry_image_path": ["images/query.jpg"],
            "pos_text": ["brown dog"],
            "pos_image_path": ["images/positive.jpg"],
            "neg_text": [""],
            "neg_image_path": [""],
        }
    ).to_parquet(dataset_input_dir / VARIANT_MAP["original"])

    summaries = convert_mmeb_train_root(input_dir=input_dir, output_dir=output_dir)

    train_rows = (
        lance.dataset(str(output_dir / "data" / dataset_name / "train.lance"))
        .to_table()
        .to_pylist()
    )
    original_rows = (
        lance.dataset(str(output_dir / "data" / dataset_name / "original.lance"))
        .to_table()
        .to_pylist()
    )
    image_rows = (
        lance.dataset(str(output_dir / "data" / "images" / f"{dataset_name}.lance"))
        .to_table()
        .to_pylist()
    )

    assert len(summaries) == 1
    assert summaries[0].subset == dataset_name
    assert {split.name for split in summaries[0].splits} == {"train", "original"}
    assert summaries[0].image_rows == 2
    assert train_rows[0]["qry"] == "find the cat"
    assert original_rows[0]["qry"] == "find the dog"
    assert sorted(row["path"] for row in image_rows) == ["images/positive.jpg", "images/query.jpg"]
    assert all(isinstance(row["image"], bytes) for row in image_rows)
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "split_inventory.json").exists()


def test_convert_mmeb_train_root_reports_missing_image_with_context(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    dataset_name = "ImageNet_1K"
    dataset_input_dir = input_dir / dataset_name
    dataset_input_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "qry": ["find the cat"],
            "qry_image_path": ["images/missing.jpg"],
            "pos_text": ["tabby cat"],
            "pos_image_path": ["images/positive.jpg"],
            "neg_text": [""],
            "neg_image_path": [""],
        }
    ).to_parquet(dataset_input_dir / VARIANT_MAP["train"])

    try:
        convert_mmeb_train_root(input_dir=input_dir, output_dir=output_dir)
    except FileNotFoundError as err:
        message = str(err)
        assert f"dataset={dataset_name}" in message
        assert "image_path=images/missing.jpg" in message
        assert "image asset is missing" in message
    else:
        raise AssertionError(
            "Expected MMEB conversion to report the missing image asset with context"
        )


def test_dataset_public_registry_entries_remain_available():
    import vlm2emb.data.datasets  # noqa: F401

    assert AutoDataset.is_registered("mmeb_train")
    assert AutoDataset.is_registered("vidore_train")
    assert AutoDataset.is_registered("vidore_rag_train")
    assert AutoDataset.is_registered("visrag_train")
    assert AutoDataset.is_registered("llavahound_train")


def test_dataset_package_recommends_runtime_schema_and_removes_legacy_schema_aliases():
    import vlm2emb.data.datasets as datasets_module
    import vlm2emb.data.datasets.const as const_module

    legacy_names = {"EncodingSample", "FieldData", "TrainExample", "TrainingSample"}

    assert legacy_names.isdisjoint(datasets_module.__all__)
    for name in legacy_names:
        assert not hasattr(datasets_module, name)
        assert not hasattr(const_module, name)

    assert TrainSample is not None
    assert MultiModalInput is not None
    assert MediaInput is not None


def test_validate_runtime_sample_rejects_media_without_visual_token():
    sample = {"text": "describe the image", "images": [object()]}
    try:
        validate_runtime_sample(sample, sample_name="encoding")
    except ValueError as err:
        assert "contains no visual token" in str(err)
    else:
        raise AssertionError("Expected ValueError for media sample without visual token")


def test_validate_runtime_sample_rejects_visual_token_without_media():
    sample = {"text": f"{STANDARD_IMAGE_TOKEN}\ndescribe the image", "images": []}
    try:
        validate_runtime_sample(sample, sample_name="encoding")
    except ValueError as err:
        assert "but no media" in str(err)
    else:
        raise AssertionError("Expected ValueError for visual token without media")


def test_validate_runtime_sample_rejects_legacy_visual_token_in_final_output():
    sample = {"text": "<video>\ndescribe the clip", "images": [object()]}
    try:
        validate_runtime_sample(sample, sample_name="encoding")
    except ValueError as err:
        assert "contains no visual token" in str(err)
    else:
        raise AssertionError("Expected ValueError for legacy visual token in final output")


def test_validate_runtime_sample_accepts_sample_level_media_metadata():
    sample = {
        "text": f"{STANDARD_VIDEO_TOKEN}\ndescribe the clip",
        "images": [object(), object()],
        "media_metadata": {"fps": 3.0, "total_num_frames": 8, "sampled_indices": [0, 4]},
    }

    validated = validate_runtime_sample(sample, sample_name="encoding")
    assert validated["media_metadata"]["fps"] == 3.0


def test_videochat2_it_transform_builds_video_instruction_sample():
    transformed = transform_videochat2_it_train_sample(
        {
            "id": "ssv2:0:0",
            "subset": "ssv2",
            "source_family": "Something-Something-V2",
            "task_view": "video_classification",
            "video": "99418.webm",
            "instruction": "Select the best category.",
            "question": "Options:\n(A) Pushing\n(B) Pulling",
            "answer": "Answer: (A) Pushing",
            "frame_unit_source": "ssv2",
            "frame_unit_key": "99418.webm",
            "source_relative_path": "video/classification/ssv2/train.json",
            "frames": [_make_png_bytes(), _make_png_bytes()],
            "frame_unit_type": "full_video",
            "source_key": "99418",
            "source_path": "99418.webm",
            "source_total_num_frames": 20,
            "clip_total_num_frames": 20,
            "fps": 12.0,
            "duration": 1.6,
            "clip_start": 0.0,
            "clip_end": 0.0,
            "sampled_indices": [0, 10],
            "sampled_timestamps": [0.0, 0.83],
            "decode_backend": "test",
        },
        dataset_name="VideoChat2-IT",
        split="official_train_without_mmeb_v2_eval",
        num_frames=1,
        config=VideoChat2ItTransformConfig(),
    )

    assert transformed["query"]["text"] == (
        f"{STANDARD_VIDEO_TOKEN}\nSelect the best category.\nOptions:\n(A) Pushing\n(B) Pulling\n"
    )
    assert len(transformed["query"]["media"]) == 1
    assert len(transformed["query"]["media"][0]["content"]) == 1
    assert transformed["positive"] == {"text": "Answer: (A) Pushing", "media": []}
    assert transformed["metadata"]["subset"] == "ssv2"


def test_videochat2_it_query_text_uses_default_body_when_empty():
    text = build_videochat2_it_query_text(instruction="", question="", trailing_newline="strip")
    assert text == f"{STANDARD_VIDEO_TOKEN}\nDescribe the provided video."


def test_validate_runtime_sample_rejects_non_dict_media_metadata():
    sample = {
        "text": f"{STANDARD_VIDEO_TOKEN}\ndescribe the clip",
        "images": [object()],
        "media_metadata": ["bad"],
    }
    try:
        validate_runtime_sample(sample, sample_name="encoding")
    except TypeError as err:
        assert "media_metadata must be dict" in str(err)
    else:
        raise AssertionError("Expected TypeError for non-dict media_metadata")


def test_visrag_transform_preserves_source_prompt_behavior():
    transformed = transform_visrag_train_sample(
        {
            "query": "What does the chart show?",
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "source": "ChartQA",
        },
        dataset_name="visrag_train",
    )

    assert transformed["query"]["text"] == f"{QUERY_SOURCE_PROMPTS['ChartQA']}What does the chart show?\n"
    assert transformed["positive"]["text"] == f"{STANDARD_IMAGE_TOKEN}\n{TARGET_SOURCE_PROMPTS['ChartQA']}\n"
    assert len(transformed["positive"]["media"]) == 1
    assert transformed["negative"] == {"text": "", "media": []}


def test_visrag_transform_allows_source_prompt_overrides():
    transform = build_visrag_train_default_transform(
        dataset_name="visrag_train",
        transform_kwargs={
            "query": {
                "instructions": {"by_source": {"ChartQA": "Custom chart prompt: "}},
                "instruction_body_separator": "none",
            },
            "positive": {
                "instructions": {"by_source": {"ChartQA": "Custom chart target."}},
                "visual_token_placement": "own_line",
            },
        },
    )
    transformed = transform(
        {
            "query": "What does the chart show?",
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "source": "ChartQA",
        }
    )

    assert transformed["query"]["text"] == "Custom chart prompt: What does the chart show?\n"
    assert transformed["positive"]["text"] == "<|image_pad|>\nCustom chart target.\n"

    try:
        build_visrag_train_default_transform(
            dataset_name="visrag_train",
            transform_kwargs={
                "query": {
                    "instructions": {"by_source": {"": "x"}},
                },
            },
        )
    except ValueError as err:
        assert "invalid source key" in str(err)
    else:
        raise AssertionError("Expected ValueError for invalid VisRAG prompt source")


def test_vidore_rag_transform_matches_archive_alignment_contract():
    transformed = transform_vidore_rag_train_sample(
        {
            "query": "Which page shows the answer?",
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "source": "docvqa",
        },
        dataset_name="vidore_rag_train",
    )
    assert transformed["query"]["text"] == f"{VIDORE_RAG_QUERY_INSTRUCTION} Which page shows the answer?\n"
    assert transformed["positive"]["text"] == f"{STANDARD_IMAGE_TOKEN}\n{VIDORE_RAG_TARGET_INSTRUCTION}\n"
    assert transformed["query"]["media"] == []
    assert len(transformed["positive"]["media"]) == 1


def test_vidore_rag_transform_allows_instruction_override_without_changing_defaults():
    transform = build_vidore_rag_train_default_transform(
        dataset_name="vidore_rag_train",
        transform_kwargs={
            "query": {
                "instruction": "Retrieve the relevant page:",
                "instruction_body_separator": "space",
            },
            "positive": {
                "instruction": "Read the selected document page.",
                "visual_token_placement": "own_line",
            },
        },
    )
    transformed = transform(
        {
            "query": "Which page shows the answer?",
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "source": "docvqa",
        }
    )

    assert transformed["query"]["text"] == "Retrieve the relevant page: Which page shows the answer?\n"
    assert transformed["positive"]["text"] == "<|image_pad|>\nRead the selected document page.\n"


def test_llavahound_video_retrieval_target_prompt_appends_video_token():
    transformed = transform_llavahound_train_sample(
        {
            "conversations": [
                {"from": "human", "value": "<video>\nDescribe this video."},
                {"from": "gpt", "value": "A person is cooking"},
            ],
            "_resolved_video": {
                "images": [object(), object()],
                "media_metadata": {"total_num_frames": 2, "sampled_indices": [0, 1]},
            },
        },
        dataset_name="llavahound/train/sft/video_caption_300k-video",
        config=build_llavahound_train_default_transform(
            dataset_name="unused",
            data_mode="video_retrieval",
        ).keywords["config"],
    )

    assert transformed["query"]["text"] == f"{VRET_QUERY_PROMPT}A person is cooking\n"
    assert transformed["positive"]["text"] == f"{VRET_TARGET_PROMPT.strip()} {STANDARD_VIDEO_TOKEN}\n"


def test_llavahound_transform_allows_config_overrides_for_caption_and_video_modes():
    record = {
        "conversations": [
            {"from": "human", "value": "<video>\nWhat is happening?"},
            {"from": "gpt", "value": "A person is cooking"},
        ],
        "_resolved_video": {
            "images": [object(), object()],
            "media_metadata": {"total_num_frames": 2, "sampled_indices": [0, 1]},
        },
    }
    caption_transform = build_llavahound_train_default_transform(
        dataset_name="llavahound/train/sft/video_240k_caption_15k",
        data_mode="caption_retrieval",
        transform_kwargs={
            "query": {
                "instruction": "Answer a question based on the content of a video.",
                "instruction_body_separator": "newline",
            },
        },
    )
    caption_sample = caption_transform(record)
    assert caption_sample["query"]["text"] == (
        "<|video_pad|>\nAnswer a question based on the content of a video.\nWhat is happening?\n"
    )
    assert caption_sample["positive"]["text"] == "A person is cooking\n"

    video_transform = build_llavahound_train_default_transform(
        dataset_name="llavahound/train/sft/video_caption_300k-video",
        data_mode="video_retrieval",
        transform_kwargs={
            "query": {
                "instruction": "Retrieve this video:",
                "instruction_body_separator": "space",
            },
            "positive": {
                "instruction": "Inspect the video:",
                "visual_token_placement": "inline_end",
            },
        },
    )
    video_sample = video_transform(record)
    assert video_sample["query"]["text"] == "Retrieve this video: A person is cooking\n"
    assert video_sample["positive"]["text"] == "Inspect the video: <|video_pad|>\n"

    default_caption_sample = build_llavahound_train_default_transform(
        dataset_name="llavahound/train/sft/video_240k_caption_15k",
        data_mode="caption_retrieval",
    )(record)
    assert default_caption_sample["query"]["text"] == "<|video_pad|>\nWhat is happening?\n"

    try:
        LlavaHoundTrainDataset.from_config(
            {
                "type": "llavahound_train",
                "path": "/tmp/llavahound",
                "subset": "train/sft/video_caption_300k",
                "transform": {"invalid": "x"},
            }
        )
    except ValueError as err:
        assert "Unsupported LLaVA-Hound transform sides" in str(err)
    else:
        raise AssertionError("Expected ValueError for unknown LLaVA-Hound transform key")

    try:
        LlavaHoundTrainDataset(
            path="/tmp/llavahound",
            subset="train/sft/video_caption_300k",
            transform={"invalid": "x"},
        )
    except TypeError as err:
        assert "transform must be a callable SampleTransform or None" in str(err)
    else:
        raise AssertionError("Expected TypeError for non-callable LLaVA-Hound transform")


def test_llavahound_training_sample_keeps_video_media_metadata_on_correct_field():
    caption_sample = transform_llavahound_train_sample(
        {
            "conversations": [
                {"from": "human", "value": "<video>\nWhat is happening?"},
                {"from": "gpt", "value": "A person is cooking"},
            ],
            "_resolved_video": {
                "images": [object(), object()],
                "media_metadata": {"total_num_frames": 16, "sampled_indices": [0, 4]},
            },
        },
        dataset_name="llavahound/train/sft/video_240k_caption_15k",
        config=build_llavahound_train_default_transform(
            dataset_name="unused",
            data_mode="caption_retrieval",
        ).keywords["config"],
    )
    assert caption_sample["query"]["media"][0]["metadata"] == {"total_num_frames": 16, "sampled_indices": [0, 4]}
    assert caption_sample["positive"]["media"] == []

    video_sample = transform_llavahound_train_sample(
        {
            "conversations": [
                {"from": "human", "value": "<video>\nDescribe"},
                {"from": "gpt", "value": "A person is cooking"},
            ],
            "_resolved_video": {
                "images": [object(), object()],
                "media_metadata": {"total_num_frames": 10, "sampled_indices": [1, 8]},
            },
        },
        dataset_name="llavahound/train/sft/video_caption_300k-video",
        config=build_llavahound_train_default_transform(
            dataset_name="unused",
            data_mode="video_retrieval",
        ).keywords["config"],
    )
    assert video_sample["positive"]["media"][0]["metadata"] == {"total_num_frames": 10, "sampled_indices": [1, 8]}
    assert video_sample["query"]["media"] == []


def test_missing_visual_token_now_raises_until_dataset_transform_restores_visual_path():
    wrapper = Qwen2VLProcessorWrapper(processor=FakeProcessor())

    try:
        wrapper(["describe the page"], [[object()]])
    except ValueError as err:
        assert "no visual token" in str(err)
    else:
        raise AssertionError("Expected ValueError for media sample without visual token")

    sample = transform_vidore_train_sample(
        {
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "query": "describe the page",
            "answer": "answer",
            "options": None,
        },
        dataset_name="vidore_train",
    )
    with_token = wrapper(
        [sample["query"]["text"]],
        [[sample["query"]["media"][0]["content"]]],
    )[0]

    assert "Options:" not in sample["query"]["text"]
    assert with_token["pixel_values"] is not None


def test_vidore_full_transform_matches_archive_fixoptions_v2_behavior():
    sample = transform_vidore_train_sample(
        {
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "query": "Which option is correct?",
            "answer": "D",
            "options": "['A. first', 'B. second', 'C. third', 'D. fourth', '-']",
            "source": "arxiv_qa",
        },
        dataset_name="vidore_train",
    )

    assert sample["query"]["text"] == "<|image_pad|>\nWhich option is correct? Options: ['A. first', 'B. second', 'C. third', 'D. fourth', '-']"
    assert sample["positive"]["text"] == "fourth"
    assert len(sample["query"]["media"]) == 1
    assert sample["metadata"]["source"] == "arxiv_qa"
    assert transform_vidore_answer("B", "['A. first', 'B. second']") == "second"


def test_visrag_transform_keeps_custom_source_with_fallback_prompts():
    sample = transform_visrag_train_sample(
        {
            "image": {"bytes": _make_png_bytes(), "path": "page.png"},
            "query": "find the section",
            "source": "CustomSource",
        },
        dataset_name="visrag_train",
        fallback_query_instruction="Locate the relevant content: ",
        fallback_positive_instruction="A scanned page.",
    )

    assert sample["query"]["text"] == "Locate the relevant content: find the section\n"
    assert sample["positive"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nA scanned page.\n"
    assert sample["metadata"]["source"] == "CustomSource"


def test_sample_columns_use_user_selected_columns_as_override():
    from vlm2emb.data.datasets.base import EvalLanceDataset

    eval_dataset = EvalLanceDataset("/tmp/eval.lance", columns=["metadata"])
    train_dataset = LlavaHoundTrainDataset(path="/tmp/llavahound", subset="train/sft/video_caption_300k", columns=["id"])

    assert eval_dataset._columns == ["metadata"]
    assert eval_dataset._sample_columns == ["metadata"]
    assert train_dataset._columns == ["id"]
    assert train_dataset._sample_columns == ["id"]


def test_raw_lance_dataset_keeps_exact_columns():
    dataset = LanceDataset("/tmp/raw.lance", columns=["id"])
    assert dataset._columns == ["id"]


def test_video_eval_dataset_updates_sampled_indices_in_media_metadata(tmp_path):
    image_bytes = _make_png_bytes()
    lance.write_dataset(
        pa.table(
            {
                "id": ["q0"],
                "text": ["<|video_pad|> describe clip"],
                "images": [[image_bytes, image_bytes, image_bytes, image_bytes]],
                "media_metadata": [{"fps": 4.0, "total_num_frames": 4}],
            }
        ),
        tmp_path / "video_eval.lance",
    )

    dataset = VideoEvalLanceDataset(str(tmp_path / "video_eval.lance"), num_frames=2)
    sample = dataset[0]

    assert len(sample["images"]) == 2
    assert sample["media_metadata"] == {
        "fps": 4.0,
        "total_num_frames": 4,
        "sampled_indices": [0, 3],
    }


def test_eval_datasets_are_picklable_for_spawn_workers():
    from vlm2emb.data.datasets.base import EvalLanceDataset

    eval_dataset = EvalLanceDataset("/tmp/eval.lance")
    video_dataset = VideoEvalLanceDataset("/tmp/video_eval.lance", num_frames=2)

    pickle.dumps(eval_dataset)
    pickle.dumps(video_dataset)


def test_training_collator_batches_nested_metadata_and_top_level_passthrough():
    class DummyWrapper:
        def __call__(self, texts, images, media_metadata=None):
            assert media_metadata == [None, None]
            return [
                {"text": text, "image_count": len(media)}
                for text, media in zip(texts, images, strict=True)
            ]

    collator = TrainingCollator(wrapper=DummyWrapper())
    batch = [
        {
            "query": {"text": "q1", "images": []},
            "positive": {"text": "p1", "images": []},
            "negative": {"text": "", "images": []},
            "dataset_name": "demo-a",
            "metadata": {"ntp_side": "query", "group": "a"},
        },
        {
            "query": {"text": "q2", "images": []},
            "positive": {"text": "p2", "images": []},
            "negative": {"text": "", "images": []},
            "dataset_name": "demo-b",
            "metadata": {"ntp_side": "positive", "group": "b"},
        },
    ]

    outputs = collator(batch)

    assert outputs["dataset_name"] == ["demo-a", "demo-b"]
    assert outputs["metadata"]["group"] == ["a", "b"]
    assert outputs["metadata"]["ntp_side"] == ["query", "positive"]


def test_training_collator_prefers_sample_media_metadata_over_legacy_relation_metadata():
    captured = []

    class DummyWrapper:
        def __call__(self, texts, images, media_metadata=None):
            captured.append(media_metadata)
            return [
                {"text": text, "image_count": len(media)}
                for text, media in zip(texts, images, strict=True)
            ]

    collator = TrainingCollator(wrapper=DummyWrapper())
    batch = [
        {
            "query": {
                "text": "<|video_pad|>\nquestion\n",
                "images": [Image.new("RGB", (2, 2))],
                "media_metadata": {"fps": 5.0, "total_num_frames": 20},
            },
            "positive": {"text": "label\n", "images": []},
            "negative": {"text": "", "images": []},
            "metadata": {"query_video_metadata": {"fps": 3.0, "total_num_frames": 12}},
        }
    ]

    collator(batch)

    assert captured[0] == [{"fps": 5.0, "total_num_frames": 20}]


def test_training_collator_forwards_query_video_metadata_to_wrapper():
    captured = []

    class DummyWrapper:
        def __call__(self, texts, images, media_metadata=None):
            captured.append(media_metadata)
            return [
                {"text": text, "image_count": len(media)}
                for text, media in zip(texts, images, strict=True)
            ]

    collator = TrainingCollator(wrapper=DummyWrapper())
    batch = [
        {
            "query": {"text": "<|video_pad|>\nquestion\n", "images": [Image.new("RGB", (2, 2))]},
            "positive": {"text": "label\n", "images": []},
            "negative": {"text": "", "images": []},
            "metadata": {"query_video_metadata": {"fps": 3.0, "total_num_frames": 12}},
        }
    ]

    collator(batch)

    assert captured[0] == [{"fps": 3.0, "total_num_frames": 12}]


def test_training_collator_rejects_mixed_media_and_legacy_images():
    class DummyWrapper:
        def __call__(self, texts, images, media_metadata=None):
            return [{"text": text} for text in texts]

    collator = TrainingCollator(wrapper=DummyWrapper())
    image = Image.new("RGB", (2, 2))
    batch = [
        {
            "query": {"text": "<|image_pad|>\nquestion\n", "media": [], "images": [image]},
            "positive": {"text": "label\n", "images": []},
            "negative": {"text": "", "images": []},
        }
    ]

    try:
        collator(batch)
    except ValueError as err:
        assert "contains both MultiModalInput.media and legacy images" in str(err)
    else:
        raise AssertionError("Expected mixed media/images training side to fail fast")


def test_training_collator_forwards_train_sample_media_payload_to_schema_wrapper():
    captured = []

    class DummyWrapper:
        def process_multimodal_batch(self, payloads):
            captured.append(payloads)
            return [
                {"text": payload["text"], "media_count": len(payload.get("media", []))}
                for payload in payloads
            ]

    collator = TrainingCollator(wrapper=DummyWrapper())
    frames = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
    batch = [
        {
            "query": {
                "text": "<|video_pad|>\nquestion\n",
                "media": [
                    {
                        "kind": "video",
                        "content": frames,
                        "metadata": {"fps": 5.0, "total_num_frames": 8},
                    }
                ],
            },
            "positive": {"text": "label\n", "media": []},
            "negative": {"text": "", "media": []},
        }
    ]

    outputs = collator(batch)

    assert outputs["query"][0]["media_count"] == 1
    assert captured[0][0]["media"][0]["kind"] == "video"
    assert captured[0][0]["media"][0]["content"] is frames
    assert captured[0][0]["media"][0]["metadata"] == {"fps": 5.0, "total_num_frames": 8}


def test_training_collator_forwards_mixed_schema_media_slots_to_schema_wrapper():
    captured = []

    class DummyWrapper:
        def process_multimodal_batch(self, payloads):
            captured.append(payloads)
            return [{"text": payload["text"]} for payload in payloads]

    collator = TrainingCollator(wrapper=DummyWrapper())
    image = Image.new("RGB", (2, 2))
    batch = [
        {
            "query": {
                "text": "<|image_pad|><|video_pad|>\nquestion\n",
                "media": [
                    {"kind": "image", "content": image},
                    {
                        "kind": "video",
                        "content": [image],
                        "metadata": {"fps": 5.0},
                    },
                ],
            },
            "positive": {"text": "label\n", "media": []},
            "negative": {"text": "", "media": []},
        }
    ]

    collator(batch)

    assert captured[0][0]["media"][0]["kind"] == "image"
    assert captured[0][0]["media"][1]["kind"] == "video"


def test_lance_dataset_loader_rejects_unknown_keyword_arguments():
    try:
        AutoDataset.build(
            "vidore_train",
            path="/tmp/vidore",
            unexpected_option=True,
        )
    except TypeError as err:
        assert "unexpected_option" in str(err)
    else:
        raise AssertionError("Expected unknown Lance dataset loader parameter to fail fast")


def test_combined_dataset_loader_rejects_unknown_keyword_arguments():
    try:
        AutoDataset.build(
            "combined",
            datasets={},
            unexpected_option=True,
        )
    except TypeError as err:
        assert "unexpected_option" in str(err)
    else:
        raise AssertionError("Expected unknown combined dataset loader parameter to fail fast")


def test_train_dataset_falsy_callable_transform_replaces_default():
    class FalsyTransform:
        def __bool__(self):
            return False

        def __call__(self, record):
            return {
                "query": {"text": "custom query", "media": []},
                "positive": {"text": "custom positive", "media": []},
                "negative": {"text": "", "media": []},
                "metadata": dict(record),
            }

    transform = FalsyTransform()
    dataset = GqaTrainDataset(path="/tmp/gqa", transform=transform, lazy=True)

    assert dataset._transform is transform


def test_train_dataset_rejects_non_callable_transform():
    try:
        GqaTrainDataset(path="/tmp/gqa", transform={"query": {}}, lazy=True)
    except TypeError as err:
        assert "transform must be a callable SampleTransform or None" in str(err)
    else:
        raise AssertionError("Expected TypeError for non-callable training transform")


def test_default_train_transforms_are_picklable():
    pickle.dumps(build_vidore_train_default_transform(dataset_name="vidore_train"))
    pickle.dumps(build_vidore_rag_train_default_transform(dataset_name="vidore_rag_train"))
    pickle.dumps(build_visrag_train_default_transform(dataset_name="visrag_train"))
    pickle.dumps(
        build_llavahound_train_default_transform(
            dataset_name="llavahound/train/sft/video_caption_300k",
            data_mode="caption_retrieval",
        )
    )
    pickle.dumps(
        build_llavahound_train_default_transform(
            dataset_name="llavahound/train/sft/video_caption_300k-video",
            data_mode="video_retrieval",
        )
    )
    pickle.dumps(
        build_mmlongbench_doc_train_default_transform(
            dataset_name="MMLongBench-doc",
            split="official_train_without_mmeb_v2_eval",
        )
    )


def test_current_dataset_modules_expose_archetype_ownership_explicitly():
    assert MMEB_ARCHETYPES == ("classification", "question_answer", "retrieval", "grounding")
    assert VIDORE_ARCHETYPE == "document_retrieval"
    assert VIDORE_RAG_ARCHETYPE == "document_retrieval"
    assert VISRAG_ARCHETYPE == "document_retrieval"
    assert LLAVAHOUND_ARCHETYPES == ("retrieval",)


def test_combined_dataset_getitems_delegates_to_child_batch_fetch_and_preserves_order():
    class DummyDataset:
        def __init__(self, prefix: str, size: int) -> None:
            self.prefix = prefix
            self.size = size
            self.batch_calls: list[list[int]] = []
            self.item_calls: list[int] = []

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int) -> str:
            self.item_calls.append(idx)
            return f"{self.prefix}-item-{idx}"

        def __getitems__(self, indices: list[int]) -> list[str]:
            self.batch_calls.append(list(indices))
            return [f"{self.prefix}-batch-{idx}" for idx in indices]

    left = DummyDataset("left", 3)
    right = DummyDataset("right", 4)
    combined = CombinedDataset(datasets=[left, right], names=["left", "right"])

    samples = combined.__getitems__([4, 1, 5, 0])

    assert samples == [
        "right-batch-1",
        "left-batch-1",
        "right-batch-2",
        "left-batch-0",
    ]
    assert left.batch_calls == [[1, 0]]
    assert right.batch_calls == [[1, 2]]
    assert left.item_calls == []
    assert right.item_calls == []
