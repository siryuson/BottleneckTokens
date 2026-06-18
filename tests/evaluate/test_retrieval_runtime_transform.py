from __future__ import annotations

import io
import json
import pickle

import lance
import pyarrow as pa
import pytest
from PIL import Image

from vlm2emb.data.collators.eval_collator import EvalCollator
from vlm2emb.data.conversion.mmeb_v2.datasets import get_all_specs
from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN
from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import BENCHMARK_INDEX_ROWS
from vlm2emb.data.datasets.mmeb_v2.dispatch import (
    get_parser_module,
)
from vlm2emb.data.datasets.mmeb_v2.parsers import image_cls as image_cls_parser
from vlm2emb.data.datasets.mmeb_v2.parsers import msrvtt as msrvtt_parser
from vlm2emb.data.datasets.transforms import transform_eval_sample_with_media_token
from vlm2emb.data.processors.qwen2_5_vl import Qwen2_5VLProcessorWrapper
from vlm2emb.data.processors.qwen2_vl import Qwen2VLProcessorWrapper
from vlm2emb.data.processors.qwen3_vl import Qwen3VLProcessorWrapper
from vlm2emb.evaluation.benchmarks.mmeb import MMEB_RUNTIME_REGISTRY, MMEBBenchmark


def _make_jpeg_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _build_dataset(
    dataset_path,
    *,
    dataset_name: str,
    transform_kwargs: dict[str, object] | None = None,
):
    return build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs=transform_kwargs or {},
    )


def _get_default_transform_kwargs(parser_module, dataset_name: str) -> dict[str, object]:
    return dict(parser_module.get_default_transform_kwargs(dataset_name))


def _assert_dataset_contract(
    dataset_name: str,
    expected_query: tuple[str, ...],
    expected_candidate: tuple[str, ...],
) -> None:
    parser_module = get_parser_module(dataset_name)
    assert tuple(parser_module.get_query_read_columns(dataset_name)) == expected_query
    assert tuple(parser_module.get_candidate_read_columns(dataset_name)) == expected_candidate


def test_eval_transform_helper_injects_image_token_for_media_samples():
    transformed = transform_eval_sample_with_media_token(
        {"id": "q0", "text": "describe the image", "images": [object()]},
        token=STANDARD_IMAGE_TOKEN,
    )

    assert transformed["text"].startswith(STANDARD_IMAGE_TOKEN)


def test_mmeb_runtime_registry_keeps_subset_coverage_in_sync_with_conversion_registry():
    expected = {spec.name for spec in get_all_specs()}
    actual = set(MMEB_RUNTIME_REGISTRY)

    assert actual == expected


def test_mmeb_runtime_transforms_are_picklable_for_spawn_workers():
    for dataset_name, _, _, _ in BENCHMARK_INDEX_ROWS:
        parser_module = get_parser_module(dataset_name)
        query_transform = parser_module.build_query_transform(
            dataset_name=dataset_name,
            transform_kwargs={"runtime_mode": "canonical"},
        )
        candidate_transform = parser_module.build_candidate_transform(
            dataset_name=dataset_name,
            transform_kwargs={"runtime_mode": "canonical"},
        )
        pickle.dumps(query_transform)
        pickle.dumps(candidate_transform)


def test_mmeb_benchmark_routes_registered_runtime_type_without_family_prefix_hooks(monkeypatch):
    import vlm2emb.evaluation.benchmarks.mmeb as mmeb_mod

    captured: list[dict[str, object]] = []

    def fake_build_dataset(
        *,
        dataset_name,
        artifact_path,
        transform_kwargs=None,
    ):
        captured.append(
            {
                "path": artifact_path,
                "dataset_name": dataset_name,
                "transform_kwargs": transform_kwargs,
            }
        )
        return object()

    def fake_evaluator(**kwargs):
        return kwargs

    monkeypatch.setattr(mmeb_mod, "build_mmeb_eval_dataset", fake_build_dataset)
    monkeypatch.setattr(mmeb_mod, "RetrievalEvaluator", fake_evaluator)

    benchmark = MMEBBenchmark(data_path="/tmp/mmeb", num_frames=16)
    benchmark._dataset_meta = {
        "VisRAG_ChartQA": {
            "task_type": "visdoc_visrag",
            "path": "/tmp/mmeb/visdoc-tasks/VisRAG_ChartQA",
            "modality": "visdoc",
        }
    }
    monkeypatch.setattr(benchmark, "_resolve_datasets", lambda: None)

    evaluators = benchmark._create_evaluators()

    assert len(evaluators) == 1
    assert evaluators[0]["metrics"] == mmeb_mod.DEFAULT_RETRIEVAL_METRICS
    assert "k_values" not in evaluators[0]
    assert captured[0]["path"] == "/tmp/mmeb/visdoc-tasks/VisRAG_ChartQA"
    assert captured[0]["dataset_name"] == "VisRAG_ChartQA"
    assert captured[0]["transform_kwargs"] == {"num_frames": 16, "runtime_mode": "canonical"}


def test_mmeb_benchmark_routes_image_wave_with_explicit_runtime_type(monkeypatch):
    import vlm2emb.evaluation.benchmarks.mmeb as mmeb_mod

    captured: list[dict[str, object]] = []

    def fake_build_dataset(
        *,
        dataset_name,
        artifact_path,
        transform_kwargs=None,
    ):
        captured.append(
            {
                "path": artifact_path,
                "dataset_name": dataset_name,
                "transform_kwargs": transform_kwargs,
            }
        )
        return object()

    def fake_evaluator(**kwargs):
        return kwargs

    monkeypatch.setattr(mmeb_mod, "build_mmeb_eval_dataset", fake_build_dataset)
    monkeypatch.setattr(mmeb_mod, "RetrievalEvaluator", fake_evaluator)

    benchmark = MMEBBenchmark(data_path="/tmp/mmeb", num_frames=12)
    benchmark._dataset_meta = {
        "ImageNet-1K": {
            "task_type": "image_classification",
            "path": "/tmp/mmeb/image-tasks/ImageNet-1K",
            "modality": "image",
        }
    }
    monkeypatch.setattr(benchmark, "_resolve_datasets", lambda: None)

    evaluators = benchmark._create_evaluators()

    assert len(evaluators) == 1
    assert captured[0]["path"] == "/tmp/mmeb/image-tasks/ImageNet-1K"
    assert captured[0]["dataset_name"] == "ImageNet-1K"
    assert captured[0]["transform_kwargs"] == {"num_frames": 12, "runtime_mode": "canonical"}


def test_mmeb_benchmark_rejects_custom_metrics_missing_aggregation_metric(monkeypatch):
    benchmark = MMEBBenchmark(data_path="/tmp/mmeb", metrics=["hit@1"])
    benchmark._dataset_meta = {
        "VisRAG_ChartQA": {
            "task_type": "visdoc_visrag",
            "path": "/tmp/mmeb/visdoc-tasks/VisRAG_ChartQA",
            "modality": "visdoc",
        }
    }
    monkeypatch.setattr(benchmark, "_resolve_datasets", lambda: None)

    with pytest.raises(ValueError, match="ndcg@5"):
        benchmark._create_evaluators()


def test_mmeb_benchmark_aggregate_rejects_missing_metric():
    benchmark = MMEBBenchmark(data_path="/tmp/mmeb")
    benchmark._dataset_meta = {
        "VisRAG_ChartQA": {
            "task_type": "visdoc_visrag",
            "path": "/tmp/mmeb/visdoc-tasks/VisRAG_ChartQA",
            "modality": "visdoc",
        }
    }

    with pytest.raises(ValueError, match="ndcg@5"):
        benchmark.aggregate({"VisRAG_ChartQA": {"hit@1": 1.0}})


def test_mmeb_benchmark_aggregates_on_non_main_process_with_consistent_scores():
    class _Runtime:
        is_main_process = False

    class _Evaluator:
        name = "VOC2007"

        def __call__(self, model, processor_wrapper, accelerator=None, **kwargs):
            return {"hit@1": 0.5}

    benchmark = MMEBBenchmark(data_path="/tmp/mmeb")
    benchmark._dataset_meta = {
        "VOC2007": {
            "task_type": "image_classification",
            "path": "/tmp/mmeb/image-tasks/VOC2007",
            "modality": "image",
        }
    }
    benchmark._evaluators = [_Evaluator()]

    results = benchmark(
        model=object(),
        processor_wrapper=object(),
        accelerator=_Runtime(),
    )

    assert results == {
        "VOC2007": {"hit@1": 0.5},
        "_summary": {"image_classification": 0.5, "image": 0.5, "overall": 0.5},
    }


def test_mmeb_benchmark_respects_explicit_runtime_mode_override(monkeypatch):
    import vlm2emb.evaluation.benchmarks.mmeb as mmeb_mod

    captured: list[dict[str, object]] = []

    def fake_build_dataset(
        *,
        dataset_name,
        artifact_path,
        transform_kwargs=None,
    ):
        captured.append(
            {
                "path": artifact_path,
                "dataset_name": dataset_name,
                "transform_kwargs": transform_kwargs,
            }
        )
        return object()

    def fake_evaluator(**kwargs):
        return kwargs

    monkeypatch.setattr(mmeb_mod, "build_mmeb_eval_dataset", fake_build_dataset)
    monkeypatch.setattr(mmeb_mod, "RetrievalEvaluator", fake_evaluator)

    benchmark = MMEBBenchmark(
        data_path="/tmp/mmeb",
        num_frames=6,
        runtime_mode="origin",
    )
    benchmark._dataset_meta = {
        "MSR-VTT": {
            "task_type": "video_retrieval",
            "path": "/tmp/mmeb/video-tasks/MSR-VTT",
            "modality": "video",
        }
    }
    monkeypatch.setattr(benchmark, "_resolve_datasets", lambda: None)

    evaluators = benchmark._create_evaluators()

    assert len(evaluators) == 1
    assert captured[0]["dataset_name"] == "MSR-VTT"
    assert captured[0]["transform_kwargs"] == {"num_frames": 6, "runtime_mode": "origin"}


def test_mmeb_benchmark_routes_visdoc_wave_with_explicit_runtime_type(monkeypatch):
    import vlm2emb.evaluation.benchmarks.mmeb as mmeb_mod

    captured: list[dict[str, object]] = []

    def fake_build_dataset(
        *,
        dataset_name,
        artifact_path,
        transform_kwargs=None,
    ):
        captured.append(
            {
                "path": artifact_path,
                "dataset_name": dataset_name,
                "transform_kwargs": transform_kwargs,
            }
        )
        return object()

    def fake_evaluator(**kwargs):
        return kwargs

    monkeypatch.setattr(mmeb_mod, "build_mmeb_eval_dataset", fake_build_dataset)
    monkeypatch.setattr(mmeb_mod, "RetrievalEvaluator", fake_evaluator)

    benchmark = MMEBBenchmark(data_path="/tmp/mmeb", num_frames=4)
    benchmark._dataset_meta = {
        "ViDoSeek-page": {
            "task_type": "visdoc_ood",
            "path": "/tmp/mmeb/visdoc-tasks/ViDoSeek-page",
            "modality": "visdoc",
        }
    }
    monkeypatch.setattr(benchmark, "_resolve_datasets", lambda: None)

    evaluators = benchmark._create_evaluators()

    assert len(evaluators) == 1
    assert captured[0]["dataset_name"] == "ViDoSeek-page"
    assert captured[0]["transform_kwargs"] == {"num_frames": 4, "runtime_mode": "canonical"}


def test_retrieval_eval_dataset_ignores_metadata_json_for_runtime_routing(tmp_path):
    dataset_path = tmp_path / "VisDial"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_inst": [None],
            "qry_text": ["Question with preserved blank line.\n\n"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "tgt_inst": [None],
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n"],
            "tgt_img_path": ["VisDial/image_0.jpg"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MSCOCO_t2i", "task_type": "image_retrieval"}, file_handle)

    dataset = build_mmeb_eval_dataset(
        dataset_name="VisDial",
        artifact_path=str(dataset_path),
        transform_kwargs={},
    )

    assert dataset.name == "VisDial"


def test_mmeb_benchmark_rejects_unknown_allowlist_subset(tmp_path):
    benchmark = MMEBBenchmark(data_path=str(tmp_path), datasets=["NotARealSubset"])

    with pytest.raises(ValueError, match="Unknown MMEB subsets requested"):
        benchmark._resolve_datasets()


def test_mmeb_benchmark_fails_fast_when_expected_subset_directory_is_missing(tmp_path, monkeypatch):
    import vlm2emb.evaluation.benchmarks.mmeb as mmeb_mod

    monkeypatch.setattr(
        mmeb_mod,
        "MMEB_RUNTIME_REGISTRY",
        {
            "VisDial": "image-tasks/VisDial",
            "MSR-VTT": "video-tasks/MSR-VTT",
        },
    )

    visdial_dir = tmp_path / "image-tasks" / "VisDial"
    visdial_dir.mkdir(parents=True)
    for artifact_name in ("queries.lance", "candidates.lance", "qrels.lance"):
        (visdial_dir / artifact_name).mkdir()

    benchmark = MMEBBenchmark(data_path=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Missing MMEB subset directories"):
        benchmark._resolve_datasets()


def test_mmeb_benchmark_fails_fast_when_subset_artifacts_are_incomplete(tmp_path, monkeypatch):
    import vlm2emb.evaluation.benchmarks.mmeb as mmeb_mod

    monkeypatch.setattr(
        mmeb_mod,
        "MMEB_RUNTIME_REGISTRY",
        {"VisDial": "image-tasks/VisDial"},
    )

    visdial_dir = tmp_path / "image-tasks" / "VisDial"
    visdial_dir.mkdir(parents=True)
    (visdial_dir / "queries.lance").mkdir()

    benchmark = MMEBBenchmark(data_path=str(tmp_path), datasets=["VisDial"])

    with pytest.raises(FileNotFoundError, match="Incomplete MMEB subset artifacts"):
        benchmark._resolve_datasets()


def test_retrieval_eval_dataset_can_route_without_metadata_when_runtime_hints_are_provided(tmp_path):
    dataset_path = tmp_path / "ImageNet-1K"
    dataset_path.mkdir()

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["<|image_1|>\nRepresent the given image for classification\n"],
            "qry_inst": [None],
            "qry_img_path": ["ImageNet-1K/image_0.jpg"],
            "image": [image_buffer.getvalue()],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["tabby cat"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ImageNet-1K",
    )

    assert dataset.name == "ImageNet-1K"
    assert dataset.metadata["benchmark"] == "MMEB-V2"


def test_retrieval_eval_dataset_builds_image_items_with_new_schema(tmp_path):
    dataset_path = tmp_path / "ImageNet-1K"
    dataset_path.mkdir()

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")

    lance.write_dataset(
        pa.table({
            "id": [0],
            "sample_id": ["sample_query"],
            "qry_text": ["<|image_1|>\nRepresent the given image for classification\n"],
            "qry_inst": [None],
            "qry_img_path": ["ImageNet-1K/image_0.jpg"],
            "image": [image_buffer.getvalue()],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["tabby cat"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ImageNet-1K", "task_type": "image_classification"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ImageNet-1K",
    )
    query = dataset.queries[0]
    candidate = dataset.candidates[0]

    assert query["id"] == "0"
    assert query["query"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image for classification\n"
    assert len(query["query"]["media"]) == 1
    assert candidate["id"] == "0"
    assert candidate["candidate"]["text"] == "tabby cat"
    assert candidate["candidate"]["media"] == []
    assert query == {
        "id": "0",
        "query": {
            "text": "<|image_pad|>\nRepresent the given image for classification\n",
            "media": [
                {
                    "kind": "image",
                    "content": query["query"]["media"][0]["content"],
                }
            ],
        },
    }


def test_retrieval_eval_dataset_transforms_image_cls_query_without_legacy_text_fields(tmp_path):
    dataset_path = tmp_path / "ImageNet-1K"
    dataset_path.mkdir()

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["<|image_1|>\nRepresent the given image for classification\n"],
            "qry_inst": [None],
            "qry_img_path": ["ImageNet-1K/image_0.jpg"],
            "image": [image_buffer.getvalue()],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["tench"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ImageNet-1K", "task_type": "image_classification"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ImageNet-1K",
    )

    sample = dataset.queries[0]

    assert sample["id"] == "0"
    assert sample["query"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image for classification\n"
    assert "render_metadata_json" not in sample.get("metadata", {})
    assert len(sample["query"]["media"]) == 1


def test_image_cls_parser_is_self_contained_and_exposes_expected_defaults():
    query_columns = image_cls_parser.QUERY_READ_COLUMNS_BY_DATASET["ImageNet-1K"]
    candidate_columns = image_cls_parser.CANDIDATE_READ_COLUMNS_BY_DATASET["ImageNet-1K"]
    default_transform_kwargs = image_cls_parser.DEFAULT_TRANSFORM_KWARGS_BY_DATASET["ImageNet-1K"]
    parser_module = get_parser_module("ImageNet-1K")

    assert query_columns == ("id", "qry_text", "image")
    assert candidate_columns == ("id", "text")
    assert tuple(parser_module.get_query_read_columns("ImageNet-1K")) == query_columns
    assert tuple(parser_module.get_candidate_read_columns("ImageNet-1K")) == candidate_columns
    parser_defaults = _get_default_transform_kwargs(parser_module, "ImageNet-1K")
    for key, value in default_transform_kwargs.items():
        assert parser_defaults[key] == value
    assert "canonical_strip_query_trailing_newline" not in parser_defaults

    query_transform = image_cls_parser.build_query_transform(
        dataset_name="ImageNet-1K",
    )
    candidate_transform = image_cls_parser.build_candidate_transform(
        dataset_name="ImageNet-1K",
    )

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")
    image = Image.open(io.BytesIO(image_buffer.getvalue())).convert("RGB")

    query_sample = query_transform(
        {
            "id": 1,
            "qry_text": "<|image_1|>\nRepresent the given image for classification\n",
            "images": [image],
        }
    )
    candidate_sample = candidate_transform({"id": 1, "text": "tabby cat"})

    assert query_sample["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image for classification\n"
    )
    assert len(query_sample["query"]["media"]) == 1
    assert candidate_sample["candidate"]["text"] == "tabby cat"
    assert candidate_sample["candidate"]["media"] == []


def test_retrieval_eval_dataset_vizwiz_canonical_normalizes_query_whitespace_only(tmp_path):
    dataset_path = tmp_path / "VizWiz"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["<|image_1|>\nRepresent the given image with the following question: What is the title of this book? \n"],
            "qry_img_path": ["VizWiz/image_0.jpg"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["book title"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "VizWiz", "task_type": "image_question_answer"}, file_handle)

    origin_dataset = _build_dataset(
        dataset_path,
        dataset_name="VizWiz",
        transform_kwargs={"runtime_mode": "origin"},
    )
    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="VizWiz",
        transform_kwargs={"runtime_mode": "canonical"},
    )

    assert origin_dataset.queries[0]["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\n"
        "Represent the given image with the following question: What is the title of this book? \n"
    )
    assert canonical_dataset.queries[0]["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\n"
        "Represent the given image with the following question: What is the title of this book?\n"
    )
    assert origin_dataset.candidates[0]["candidate"]["text"] == "book title"
    assert canonical_dataset.candidates[0]["candidate"]["text"] == "book title"


def test_mmeb_runtime_registry_exposes_parameterized_subset_defaults():
    ok_vqa_defaults = _get_default_transform_kwargs(get_parser_module("OK-VQA"), "OK-VQA")
    vizwiz_defaults = _get_default_transform_kwargs(get_parser_module("VizWiz"), "VizWiz")
    visdial_defaults = _get_default_transform_kwargs(get_parser_module("VisDial"), "VisDial")
    visrag_defaults = _get_default_transform_kwargs(get_parser_module("VisRAG_ChartQA"), "VisRAG_ChartQA")
    mscoco_i2t_defaults = _get_default_transform_kwargs(get_parser_module("MSCOCO_i2t"), "MSCOCO_i2t")
    visualnews_i2t_defaults = _get_default_transform_kwargs(get_parser_module("VisualNews_i2t"), "VisualNews_i2t")

    assert ok_vqa_defaults["query_instruction_leading_newline"] is True
    assert vizwiz_defaults["canonical_query_whitespace_normalization"] is True
    assert visdial_defaults["canonical_trim_single_extra_blank_line"] is True
    assert visrag_defaults["candidate_instruction"] == (
        "Understand the content of the provided document image."
    )
    assert mscoco_i2t_defaults == {"runtime_mode": "canonical"}
    assert visualnews_i2t_defaults == {"runtime_mode": "canonical"}


def test_typical_subsets_use_subset_specific_exact_columns():
    _assert_dataset_contract("VisDial", ("id", "qry_inst", "qry_text"), ("id", "tgt_inst", "tgt_text", "tgt_img_path", "image"))
    _assert_dataset_contract("MSCOCO_t2i", ("id", "qry_text"), ("id", "tgt_text", "image"))
    _assert_dataset_contract("ImageNet-1K", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("ImageNet-A", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("ImageNet-R", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("N24News", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("HatefulMemes", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("VOC2007", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("SUN397", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("Place365", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("ObjectNet", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("Country211", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("WebQA", ("id", "qry_text"), ("id", "tgt_inst", "tgt_text", "image"))
    _assert_dataset_contract("MSCOCO_i2t", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("VisualNews_i2t", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("VisualNews_t2i", ("id", "qry_text"), ("id", "tgt_text", "image"))
    _assert_dataset_contract("Wiki-SS-NQ", ("id", "qry_text"), ("id", "tgt_text", "image"))
    _assert_dataset_contract("OK-VQA", ("id", "qry_inst", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("A-OKVQA", ("id", "qry_inst", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("DocVQA", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("VizWiz", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("InfographicsVQA", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("ChartQA", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("Visual7W", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("TextVQA", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("GQA", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("ScienceQA", ("id", "qry_text", "image"), ("id", "text"))
    _assert_dataset_contract("MSR-VTT", ("id", "caption"), ("id", "video_id", "video"))
    _assert_dataset_contract("DiDeMo", ("id", "caption"), ("id", "video", "video_path", "source"))


def test_msrvtt_parser_build_transforms_are_self_contained():
    query_transform = msrvtt_parser.build_query_transform(
        dataset_name="MSR-VTT",
    )
    candidate_transform = msrvtt_parser.build_candidate_transform(
        dataset_name="MSR-VTT",
    )

    query_item = query_transform(
        {
            "id": 0,
            "caption": "a dog runs\n\n",
        }
    )
    candidate_item = candidate_transform(
        {
            "id": 0,
            "video_id": "video_0",
            "video": [_make_jpeg_bytes()],
        }
    )

    assert query_item == {
        "id": "0",
        "query": {
            "text": "Find a video that contains the following visual content: a dog runs\n",
            "media": [],
        },
    }
    assert candidate_item["candidate"]["text"] == (
        "<|video_pad|>\nUnderstand the content of the provided video.\n"
    )
    assert candidate_item["metadata"] == {"video_id": "video_0"}

    def assert_columns(dataset_name: str, expected_query: tuple[str, ...], expected_candidate: tuple[str, ...]) -> None:
        parser_module = get_parser_module(dataset_name)
        assert tuple(parser_module.get_query_read_columns(dataset_name)) == expected_query
        assert tuple(parser_module.get_candidate_read_columns(dataset_name)) == expected_candidate

    assert_columns("ViDoSeek-page", ("id", "query", "corpus_range"), ("id", "image", "path"))
    assert_columns("RefCOCO", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("MSCOCO", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("Visual7W-Pointing", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("K700", ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"), ("id", "text"))
    assert_columns("UCF101", ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"), ("id", "text"))
    assert_columns("HMDB51", ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"), ("id", "text"))
    assert_columns("Breakfast", ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"), ("id", "text"))
    assert_columns("SmthSmthV2", ("id", "pos_text", "video", "video_id"), ("id", "text"))
    assert_columns("RefCOCO-Matching", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("ViDoSeek-doc", ("id", "query", "corpus_range"), ("id", "image"))
    assert_columns("ViDoRe_arxivqa", ("id", "query"), ("id", "image"))
    assert_columns("ViDoRe_infovqa", ("id", "query"), ("id", "image"))
    assert_columns("ViDoRe_tabfquad", ("id", "query"), ("id", "image", "path"))
    assert_columns("ViDoRe_tatdqa", ("id", "query"), ("id", "image", "path"))
    assert_columns("ViDoRe_shiftproject", ("id", "query"), ("id", "image"))
    assert_columns("ViDoRe_syntheticDocQA_energy", ("id", "query"), ("id", "image"))
    assert_columns("ViDoRe_docvqa", ("id", "query"), ("id", "image"))
    assert_columns("YouCook2", ("id", "sentence"), ("id", "video", "source_id", "segment", "video_path", "youtube_id"))
    assert_columns("VATEX", ("id", "enCap"), ("id", "video", "videoID"))
    assert_columns("CIRR", ("id", "qry_inst", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("NIGHTS", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("OVEN", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("FashionIQ", ("id", "qry_text", "image"), ("id", "tgt_text", "image"))
    assert_columns("EDIS", ("id", "qry_text"), ("id", "tgt_text", "image"))
    assert_columns("QVHighlight", ("id", "query", "video"), ("id", "video"))
    assert_columns("Charades-STA", ("id", "query", "video"), ("id", "video"))
    assert_columns("MomentSeeker", ("id", "query", "image", "video"), ("id", "video"))
    assert_columns("Video-MME", ("id", "question", "options", "video"), ("id", "text"))
    assert_columns("NExTQA", ("id", "question", "a0", "a1", "a2", "a3", "a4", "video"), ("id", "text"))
    assert_columns("EgoSchema", ("id", "question", "option", "video"), ("id", "text"))
    assert_columns("MVBench", ("id", "question", "candidates", "video"), ("id", "text"))
    assert_columns("ActivityNetQA", ("id", "question", "video"), ("id", "text"))
    assert_columns("VisRAG_ChartQA", ("id", "query"), ("id", "image", "path"))
    assert_columns("MMLongBench-page", ("id", "query", "corpus_range"), ("id", "image", "path"))


def test_retrieval_eval_dataset_image_qa_defaults_can_override_instruction_layout(tmp_path):
    dataset_path = tmp_path / "OK-VQA"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["Question one?"],
            "qry_inst": ["<|image_1|> Represent the given image with the following question:"],
            "qry_img_path": ["OK-VQA/image_0.jpg"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["answer-a"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "OK-VQA", "task_type": "image_question_answer"}, file_handle)

    default_dataset = _build_dataset(
        dataset_path,
        dataset_name="OK-VQA",
    )
    overridden_dataset = _build_dataset(
        dataset_path,
        dataset_name="OK-VQA",
        transform_kwargs={"query_instruction_leading_newline": False},
    )

    assert default_dataset.queries[0]["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image with the following question: Question one?\n"
    )
    assert overridden_dataset.queries[0]["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN} Represent the given image with the following question: Question one?\n"
    )


def test_retrieval_eval_dataset_transforms_image_qa_query_without_legacy_text_fields(tmp_path):
    dataset_path = tmp_path / "OK-VQA"
    dataset_path.mkdir()

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["<|image_1|>\nRepresent the given image with the following question: Question one?\n"],
            "qry_inst": [None],
            "qry_img_path": ["OK-VQA/image_0.jpg"],
            "image": [image_buffer.getvalue()],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["answer-a"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "OK-VQA", "task_type": "image_question_answer"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="OK-VQA",
    )

    sample = dataset.queries[0]

    assert sample["id"] == "0"
    assert sample["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image with the following question: Question one?\n"
    )
    assert "render_metadata_json" not in sample.get("metadata", {})
    assert len(sample["query"]["media"]) == 1


def test_retrieval_eval_dataset_visdial_modes_produce_different_query_text(tmp_path):
    dataset_path = tmp_path / "VisDial"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_inst": [None],
            "qry_text": ["Question with preserved blank line.\n\n"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "tgt_inst": [None],
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n"],
            "tgt_img_path": ["VisDial/image_0.jpg"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "VisDial", "task_type": "image_retrieval"}, file_handle)

    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="VisDial",
        transform_kwargs={"runtime_mode": "canonical"},
    )
    parser_dataset = _build_dataset(
        dataset_path,
        dataset_name="VisDial",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.queries[0]["query"]["text"] == "Question with preserved blank line.\n"
    assert parser_dataset.queries[0]["query"]["text"] == "Question with preserved blank line.\n\n"


def test_retrieval_eval_dataset_visdial_canonical_uses_newline_after_dialog_instruction(tmp_path):
    dataset_path = tmp_path / "VisDial"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_inst": [None],
            "qry_text": [
                "Represent the given dialogue about an image, which is used for image retrieval: "
                "Q:is the photo in color\nA:yes\n"
            ],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "tgt_inst": [None],
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n"],
            "tgt_img_path": ["VisDial/image_0.jpg"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "VisDial", "task_type": "image_retrieval"}, file_handle)

    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="VisDial",
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = _build_dataset(
        dataset_path,
        dataset_name="VisDial",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.queries[0]["query"]["text"] == (
        "Represent the given dialogue about an image, which is used for image retrieval:\n"
        "Q:is the photo in color\nA:yes\n"
    )
    assert origin_dataset.queries[0]["query"]["text"] == (
        "Represent the given dialogue about an image, which is used for image retrieval: "
        "Q:is the photo in color\nA:yes\n"
    )


def test_retrieval_eval_dataset_transforms_image_t2i_candidate_without_legacy_text_fields(tmp_path):
    dataset_path = tmp_path / "MSCOCO_t2i"
    dataset_path.mkdir()

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["Find me the matching image.\n"],
            "qry_img_path": [""],
            "image": [None],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n"],
            "tgt_inst": [None],
            "tgt_img_path": ["MSCOCO_t2i/image_0.jpg"],
            "image": [image_buffer.getvalue()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MSCOCO_t2i", "task_type": "image_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="MSCOCO_t2i",
    )

    query = dataset.queries[0]
    candidate = dataset.candidates[0]

    assert query["query"]["text"] == "Find me the matching image.\n"
    assert query["query"]["media"] == []
    assert candidate["candidate"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image.\n"
    assert len(candidate["candidate"]["media"]) == 1


def test_retrieval_eval_dataset_image_t2i_text_only_query_alignment_applies_to_origin_and_canonical(tmp_path):
    dataset_path = tmp_path / "WebQA"
    dataset_path.mkdir()

    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_buffer, format="JPEG")

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_text": ["<|image_1|>\nFind a Wikipedia image that answers this question.\n"],
            "qry_img_path": [""],
            "image": [None],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "tgt_text": ["<|image_1|>\nRepresent the given Wikipedia image.\n"],
            "tgt_inst": [None],
            "tgt_img_path": ["WebQA/image_0.jpg"],
            "image": [image_buffer.getvalue()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "WebQA", "task_type": "image_retrieval"}, file_handle)

    origin_dataset = build_mmeb_eval_dataset(
        dataset_name="WebQA",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )
    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name="WebQA",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )

    origin_query = origin_dataset.queries[0]
    origin_candidate = origin_dataset.candidates[0]
    canonical_query = canonical_dataset.queries[0]
    canonical_candidate = canonical_dataset.candidates[0]

    aligned_query_text = "Find a Wikipedia image that answers this question.\n"
    assert origin_query["query"]["text"] == aligned_query_text
    assert origin_query["query"]["media"] == []
    assert canonical_query["query"]["text"] == aligned_query_text
    assert canonical_query["query"]["media"] == []
    assert origin_query.get("_validation") is None
    assert canonical_query.get("_validation") is None
    assert origin_query.get("metadata", {}).get("surface_repairs") == [
        "visual_token_alignment_strip_without_media"
    ]
    assert canonical_query.get("metadata", {}).get("surface_repairs") == [
        "visual_token_alignment_strip_without_media"
    ]
    assert origin_candidate["candidate"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nRepresent the given Wikipedia image.\n"
    assert len(origin_candidate["candidate"]["media"]) == 1
    assert canonical_candidate["candidate"]["text"] == origin_candidate["candidate"]["text"]
    assert len(canonical_candidate["candidate"]["media"]) == 1


def test_retrieval_eval_dataset_visdial_canonical_candidate_uses_block_layout(tmp_path):
    dataset_path = tmp_path / "VisDial"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_inst": [None],
            "qry_text": ["Question one.\n"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "tgt_inst": [None],
            "tgt_text": ["<|image_1|> Represent the given image.\n"],
            "tgt_img_path": ["VisDial/image_0.jpg"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "VisDial", "task_type": "image_retrieval"}, file_handle)

    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="VisDial",
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = _build_dataset(
        dataset_path,
        dataset_name="VisDial",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.candidates[0]["candidate"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image.\n"
    )
    assert origin_dataset.candidates[0]["candidate"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN} Represent the given image.\n"
    )


def test_retrieval_eval_dataset_transforms_video_classification_query_with_new_schema(tmp_path):
    dataset_path = tmp_path / "K700"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "video_id": ["video_0"],
            "video_path": ["video_0.mp4"],
            "qry_instruction": ["Recognize the category of the video content."],
            "qry_text": [None],
            "pos_text": ["running"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 0))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0, 1],
            "text": ["running", "jumping"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "K700", "task_type": "video_classification"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="K700",
    )

    query = dataset.queries[0]
    candidate = dataset.candidates[0]

    assert query["query"]["text"] == "<|video_pad|>\nRecognize the category of the video content.\n"
    assert query["query"]["media"][0]["kind"] == "video"
    assert len(query["query"]["media"][0]["content"]) == 2
    assert candidate["candidate"]["text"] == "running"
    assert candidate["candidate"]["media"] == []


def test_retrieval_eval_dataset_transforms_video_qa_query_with_new_schema(tmp_path):
    dataset_path = tmp_path / "NExTQA"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "question": ["What is the person doing?"],
            "answer": [1],
            "a0": ["running"],
            "a1": ["jumping"],
            "a2": ["sitting"],
            "a3": ["sleeping"],
            "a4": ["reading"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 0, 255))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0, 1, 2, 3, 4],
            "text": ["running", "jumping", "sitting", "sleeping", "reading"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0, 1, 2, 3, 4]],
            "candidate_scores": [[0.0, 1.0, 0.0, 0.0, 0.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "NExTQA", "task_type": "video_question_answer"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="NExTQA",
    )

    query = dataset.queries[0]

    assert query["query"]["text"] == (
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the "
        "provided candidates. Return only the exact text of your chosen answer. Question:  "
        "What is the person doing?\n"
        "Options:\n"
        "(A) running\n"
        "(B) jumping\n"
        "(C) sitting\n"
        "(D) sleeping\n"
        "(E) reading\n"
    )
    assert query["query"]["media"][0]["kind"] == "video"


def test_retrieval_eval_dataset_transforms_videomme_origin_with_archive_shape(tmp_path):
    dataset_path = tmp_path / "Video-MME"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "question": ["Which decoration has the largest number?"],
            "options": [["A. Apples.", "B. Candles.", "C. Berries.", "D. The same number."]],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 0, 255))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0, 1, 2, 3],
            "text": ["A. Apples.", "B. Candles.", "C. Berries.", "D. The same number."],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0, 1, 2, 3]],
            "candidate_scores": [[0.0, 0.0, 1.0, 0.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "Video-MME", "task_type": "video_question_answer"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="Video-MME",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert dataset.queries[0]["query"]["text"] == (
        "<|video_pad|> Given a video and a question, select the most accurate answer from the "
        "provided candidates. Return only the exact text of your chosen answer. Question:  "
        "Which decoration has the largest number?\n"
        "A. Apples.\nB. Candles.\nC. Berries.\nD. The same number."
    )
    assert dataset.candidates[0]["candidate"]["text"] == "A. Apples."


def test_retrieval_eval_dataset_transforms_egoschema_origin_with_archive_shape(tmp_path):
    dataset_path = tmp_path / "EgoSchema"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "question": ["What is c doing?"],
            "option": [["A. C is cooking.", "B. C is doing laundry.", "C. C is cleaning."]],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 0, 255))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0, 1, 2],
            "text": ["A. C is cooking.", "B. C is doing laundry.", "C. C is cleaning."],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0, 1, 2]],
            "candidate_scores": [[1.0, 0.0, 0.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "EgoSchema", "task_type": "video_question_answer"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="EgoSchema",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert dataset.queries[0]["query"]["text"] == (
        "<|video_pad|> Given a video and a question, select the most accurate answer from the "
        "provided candidates. Return only the exact text of your chosen answer. Question:  "
        "What is c doing?A. C is cooking. B. C is doing laundry. C. C is cleaning."
    )
    assert dataset.candidates[0]["candidate"]["text"] == "A. C is cooking."


def test_retrieval_eval_dataset_transforms_mvbench_origin_with_prefixed_candidate_artifact(tmp_path):
    dataset_path = tmp_path / "MVBench"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "question": ["Why did Castle dress like a fairy?"],
            "candidates": [[
                "To get her to trust him",
                "He secretly loved fairies",
                "He lost a bet with Emily",
            ]],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 0, 255))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0, 1, 2],
            "text": [
                "(A) To get her to trust him",
                "(B) He secretly loved fairies",
                "(C) He lost a bet with Emily",
            ],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MVBench", "task_type": "video_question_answer"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="MVBench",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert dataset.queries[0]["query"]["text"] == (
        "<|video_pad|> Given a video and a question, select the most accurate answer from the "
        "provided candidates. Return only the exact text of your chosen answer. Question:  "
        "Why did Castle dress like a fairy?\n"
        "Options:\n"
        "(A) To get her to trust him\n"
        "(B) He secretly loved fairies\n"
        "(C) He lost a bet with Emily"
    )
    assert dataset.candidates[0]["candidate"]["text"] == "(A) To get her to trust him"
    assert dataset.qrels.to_table().to_pylist() == [
        {
            "query_id": 0,
            "mode": "sparse",
            "candidate_ids": [0],
            "candidate_scores": [1.0],
        }
    ]


def test_retrieval_eval_dataset_transforms_activitynetqa_origin_with_archive_shape(tmp_path):
    dataset_path = tmp_path / "ActivityNetQA"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "question": ["does the girl have long hair"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 0, 255))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0, 1],
            "text": ["yes", "no"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0, 1]],
            "candidate_scores": [[1.0, 0.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ActivityNetQA", "task_type": "video_question_answer"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ActivityNetQA",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert dataset.queries[0]["query"]["text"] == (
        "<|video_pad|> Given a video and a question, select the most accurate answer from the "
        "provided candidates. Return only the exact text of your chosen answer. Question:  "
        "does the girl have long hair? (A) yes; (B) no."
    )
    assert dataset.candidates[0]["candidate"]["text"] == "yes"


def test_retrieval_eval_dataset_video_retrieval_modes_produce_different_query_text(tmp_path):
    dataset_path = tmp_path / "MSR-VTT"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "caption": ["a dog runs\n\n"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video_id": ["video_0"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MSR-VTT", "task_type": "video_retrieval"}, file_handle)

    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="MSR-VTT",
        transform_kwargs={"runtime_mode": "canonical"},
    )
    parser_dataset = _build_dataset(
        dataset_path,
        dataset_name="MSR-VTT",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.queries[0]["query"]["text"] == (
        "Find a video that contains the following visual content: a dog runs\n"
    )
    assert parser_dataset.queries[0]["query"]["text"] == (
        "Find a video that contains the following visual content: a dog runs\n\n"
    )


def test_retrieval_eval_dataset_didemo_origin_matches_archive_prompt(tmp_path):
    dataset_path = tmp_path / "DiDeMo"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "caption": ["a dog runs"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
            "video_path": ["didemo/video_0.mp4"],
            "source": ["DiDeMo"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "DiDeMo", "task_type": "video_retrieval"}, file_handle)

    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="DiDeMo",
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = _build_dataset(
        dataset_path,
        dataset_name="DiDeMo",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.queries[0]["query"]["text"] == (
        "Find a video that includes the following described scenes: a dog runs\n"
    )
    assert origin_dataset.queries[0]["query"]["text"] == (
        "Find a video that includes the following described scenes: a dog runs"
    )


def test_retrieval_eval_dataset_ucf101_origin_uses_archive_instruction_shape(tmp_path):
    dataset_path = tmp_path / "UCF101"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "qry_instruction": ["What activities or sports are being performed by the person in the video?"],
            "qry_text": [None],
            "video_id": ["video_0"],
            "video_path": ["video_0.mp4"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["Knitting"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "UCF101", "task_type": "video_classification"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="UCF101",
        transform_kwargs={"runtime_mode": "origin"},
    )

    query = dataset.queries[0]
    assert query["query"]["text"] == (
        "<|video_pad|> What activities or sports are being performed by the person in the video?"
    )
    assert query["query"]["media"][0]["kind"] == "video"


def test_retrieval_eval_dataset_visdoc_origin_matches_archive_style(tmp_path):
    dataset_path = tmp_path / "ViDoSeek-page"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": ["Apply for Nordic Swan Ecolabel license?"],
            "corpus_range": ["page"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [3],
            "path": ["doc_3.png"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[3]],
            "candidate_scores": [[3.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ViDoSeek-page", "task_type": "visdoc_ood"}, file_handle)

    canonical_dataset = _build_dataset(
        dataset_path,
        dataset_name="ViDoSeek-page",
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = _build_dataset(
        dataset_path,
        dataset_name="ViDoSeek-page",
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.queries[0]["query"]["text"] == (
        "Find a document image that matches the given query: Apply for Nordic Swan Ecolabel license?\n"
    )
    assert origin_dataset.queries[0]["query"]["text"] == (
        "Find a document image that matches the given query: Apply for Nordic Swan Ecolabel license?"
    )
    assert canonical_dataset.candidates[0]["candidate"]["text"] == (
        "<|image_pad|>\nUnderstand the content of the provided document image.\n"
    )
    assert origin_dataset.candidates[0]["candidate"]["text"] == (
        "<|image_pad|> Understand the content of the provided document image."
    )


def test_video_retrieval_loader_fails_fast_when_candidate_media_is_missing(tmp_path):
    dataset_path = tmp_path / "MSR-VTT"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "caption": ["a dog runs"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video_id": ["video_0"],
            "video": [[]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MSR-VTT", "task_type": "video_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="MSR-VTT",
    )

    with pytest.raises(ValueError, match="Missing required media for MMEB eval transform"):
        _ = dataset.candidates[0]


def test_retrieval_eval_dataset_transforms_video_retrieval_candidate_with_new_schema(tmp_path):
    dataset_path = tmp_path / "MSR-VTT"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "caption": ["a dog runs"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video_id": ["video_0"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MSR-VTT", "task_type": "video_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="MSR-VTT",
    )

    query = dataset.queries[0]
    candidate = dataset.candidates[0]

    assert query["query"]["text"] == "Find a video that contains the following visual content: a dog runs\n"
    assert query["query"]["media"] == []
    assert candidate["candidate"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert candidate["candidate"]["media"][0]["kind"] == "video"
    assert candidate == {
        "id": "0",
        "candidate": {
            "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
            "media": [
                {
                    "kind": "video",
                    "content": candidate["candidate"]["media"][0]["content"],
                    "metadata": candidate["candidate"]["media"][0]["metadata"],
                }
            ],
        },
        "metadata": {
            "video_id": "video_0",
        },
    }


def test_retrieval_eval_dataset_transforms_vatex_query_from_encap_field(tmp_path):
    dataset_path = tmp_path / "VATEX"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "videoID": ["video_0"],
            "enCap": [["a person is cooking"]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "videoID": ["video_0"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "VATEX", "task_type": "video_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="VATEX",
    )

    assert dataset.queries[0]["query"]["text"] == (
        "Select a video that fits the description provided: a person is cooking\n"
    )


def test_retrieval_eval_dataset_transforms_youcook2_query_from_sentence_field(tmp_path):
    dataset_path = tmp_path / "YouCook2"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "sentence": ["slice the vegetables"],
            "video_path": ["clip_0.mp4"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "source_id": ["clip_0"],
            "segment": ["segment_0"],
            "video_path": ["clip_0.mp4"],
            "youtube_id": ["yt_0"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "YouCook2", "task_type": "video_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="YouCook2",
    )

    assert dataset.queries[0]["query"]["text"] == (
        "Find a video that demonstrates the following action while making a recipe: slice the vegetables\n"
    )


def test_retrieval_eval_dataset_transforms_video_moment_query_with_new_schema(tmp_path):
    dataset_path = tmp_path / "QVHighlight"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": ["Find the matching clip."],
            "clips_dir_path": ["/tmp/source/sample_query"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((255, 255, 0))]],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((255, 0, 255))]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "QVHighlight", "task_type": "video_moment_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="QVHighlight",
    )

    query = dataset.queries[0]
    candidate = dataset.candidates[0]

    assert query["query"]["text"] == (
        "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: "
        "Find the matching clip.\n"
    )
    assert query["query"]["media"][0]["kind"] == "video"
    assert candidate["candidate"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"
    assert candidate["candidate"]["media"][0]["kind"] == "video"


def test_retrieval_eval_dataset_transforms_momentseeker_image_query_with_image_media(tmp_path):
    dataset_path = tmp_path / "MomentSeeker"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": ["Find the matching clip for this image."],
            "image": [_make_jpeg_bytes()],
            "video": [None],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((255, 0, 255))]],
            "output_path": ["videos/pos_0.mp4"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MomentSeeker", "task_type": "video_moment_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="MomentSeeker",
    )

    query = dataset.queries[0]
    assert query["query"]["text"] == (
        "<|image_pad|>\nSelect the video clip that aligns with the given text and image: "
        "Find the matching clip for this image.\n"
    )
    assert query["query"]["media"][0]["kind"] == "image"


def test_retrieval_eval_dataset_transforms_momentseeker_text_only_query_without_media(tmp_path):
    dataset_path = tmp_path / "MomentSeeker"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": ["Find the matching clip for this text only query."],
            "image": [None],
            "video": [None],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((255, 0, 255))]],
            "output_path": ["videos/pos_0.mp4"],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["exhaustive"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MomentSeeker", "task_type": "video_moment_retrieval"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="MomentSeeker",
    )

    query = dataset.queries[0]
    assert query["query"]["text"] == (
        "Find the clip that corresponds to the given text: "
        "Find the matching clip for this text only query.\n"
    )
    assert query["query"]["media"] == []


def test_retrieval_eval_dataset_visdoc_defaults_can_override_instruction_text(tmp_path):
    dataset_path = tmp_path / "ViDoSeek-page"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": ["Apply for Nordic Swan Ecolabel license?"],
            "corpus_range": ["page"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [3],
            "path": ["doc_3.png"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[3]],
            "candidate_scores": [[3.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ViDoSeek-page", "task_type": "visdoc_ood"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ViDoSeek-page",
        transform_kwargs={"query_instruction": "Locate the matching page:"},
    )

    assert dataset.queries[0]["query"]["text"] == (
        "Locate the matching page: Apply for Nordic Swan Ecolabel license?\n"
    )


def test_retrieval_eval_dataset_transforms_visdoc_candidate_with_new_schema(tmp_path):
    dataset_path = tmp_path / "ViDoSeek-page"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": ["Apply for Nordic Swan Ecolabel license?"],
            "corpus_range": ["page"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [3],
            "path": ["doc_3.png"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[3]],
            "candidate_scores": [[3.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ViDoSeek-page", "task_type": "visdoc_ood"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ViDoSeek-page",
    )

    query = dataset.queries[0]
    candidate = dataset.candidates[0]

    assert query["query"]["text"] == (
        "Find a document image that matches the given query: Apply for Nordic Swan Ecolabel license?\n"
    )
    assert query["query"]["media"] == []
    assert candidate["candidate"]["text"] == (
        "<|image_pad|>\nUnderstand the content of the provided document image.\n"
    )
    assert candidate["candidate"]["media"][0]["kind"] == "image"
    assert query == {
        "id": "0",
        "query": {
            "text": "Find a document image that matches the given query: Apply for Nordic Swan Ecolabel license?\n",
            "media": [],
        },
        "metadata": {
            "corpus_range": "page",
        },
    }


def test_eval_collator_forwards_retrieval_schema_batch_to_wrapper(monkeypatch):
    captured: dict[str, object] = {}

    class DummyWrapper:
        def process_retrieval_batch(self, batch):
            captured["batch"] = batch
            return [{"input_ids": object()} for _ in batch]

    def fake_batch_processor_outputs(samples, wrapper):
        captured["samples"] = samples
        captured["wrapper"] = wrapper
        return {"batched": True}

    monkeypatch.setattr(
        "vlm2emb.data.collators.eval_collator.batch_processor_outputs",
        fake_batch_processor_outputs,
    )

    collator = EvalCollator(processor_wrapper=DummyWrapper())
    result = collator([
        {
            "id": "q0",
            "query": {
                "text": "<|image_pad|> classify cat\n",
                "media": [{"kind": "image", "content": object()}],
            },
        },
        {
            "id": "q1",
            "query": {
                "text": "plain text",
                "media": [],
            },
        },
    ])

    assert len(captured["batch"]) == 2
    assert result == {"inputs": {"batched": True}}


def test_eval_collator_rejects_wrapper_without_retrieval_batch_support():
    collator = EvalCollator(processor_wrapper=object())
    with pytest.raises(TypeError, match="does not support retrieval schema batches"):
        collator([{"id": "q0", "query": {"text": "plain text", "media": []}}])


@pytest.mark.parametrize(
    ("wrapper_cls", "wrapper_name"),
    [
        (Qwen2VLProcessorWrapper, "Qwen2VLProcessorWrapper"),
        (Qwen2_5VLProcessorWrapper, "Qwen2_5VLProcessorWrapper"),
        (Qwen3VLProcessorWrapper, "Qwen3VLProcessorWrapper"),
    ],
)
def test_processor_wrappers_raise_for_raw_video_retrieval_media(wrapper_cls, wrapper_name):
    wrapper = object.__new__(wrapper_cls)
    with pytest.raises(NotImplementedError, match="Raw-video retrieval media is not supported yet"):
        wrapper.process_retrieval_batch(
            [
                {
                    "id": "c0",
                    "candidate": {
                        "text": "<|video_pad|>\nUnderstand the content of the provided video.\n",
                        "media": [{"kind": "video", "content": object()}],
                    },
                }
            ]
        )


@pytest.mark.parametrize(
    "media",
    [
        [
            {"kind": "image", "content": Image.new("RGB", (2, 2))},
            {"kind": "video", "content": [Image.new("RGB", (2, 2))]},
        ],
        [
            {"kind": "video", "content": [Image.new("RGB", (2, 2))]},
            {"kind": "image", "content": Image.new("RGB", (2, 2))},
        ],
    ],
)
@pytest.mark.parametrize(
    ("wrapper_cls", "wrapper_name"),
    [
        (Qwen2VLProcessorWrapper, "Qwen2VLProcessorWrapper"),
        (Qwen2_5VLProcessorWrapper, "Qwen2_5VLProcessorWrapper"),
        (Qwen3VLProcessorWrapper, "Qwen3VLProcessorWrapper"),
    ],
)
def test_processor_wrappers_reject_mixed_retrieval_media_in_both_orders(
    wrapper_cls,
    wrapper_name,
    media,
):
    wrapper = object.__new__(wrapper_cls)
    with pytest.raises(ValueError, match="Mixed image/video media is not supported"):
        wrapper.process_retrieval_batch(
            [
                {
                    "id": "c0",
                    "candidate": {
                        "text": "<|image_pad|><|video_pad|>\nUnderstand the content.\n",
                        "media": media,
                    },
                }
            ]
        )


def test_video_retrieval_loader_fails_fast_when_required_caption_column_is_missing(tmp_path):
    dataset_path = tmp_path / "MSR-VTT"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "text": ["legacy prompt only"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [0],
            "video_id": ["video_0"],
            "video": [[_make_jpeg_bytes(), _make_jpeg_bytes((0, 255, 255))]],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[0]],
            "candidate_scores": [[1.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "MSR-VTT", "task_type": "video_retrieval"}, file_handle)

    with pytest.raises(ValueError, match="Missing required eval columns"):
        _build_dataset(
            dataset_path,
            dataset_name="MSR-VTT",
        )


def test_visdoc_loader_fails_fast_when_required_query_text_is_missing(tmp_path):
    dataset_path = tmp_path / "ViDoSeek-page"
    dataset_path.mkdir()

    lance.write_dataset(
        pa.table({
            "id": [0],
            "query": [""],
            "corpus_range": ["page"],
        }),
        str(dataset_path / "queries.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "id": [3],
            "path": ["doc_3.png"],
            "image": [_make_jpeg_bytes()],
        }),
        str(dataset_path / "candidates.lance"),
        mode="create",
    )
    lance.write_dataset(
        pa.table({
            "query_id": [0],
            "mode": ["sparse"],
            "candidate_ids": [[3]],
            "candidate_scores": [[3.0]],
        }),
        str(dataset_path / "qrels.lance"),
        mode="create",
    )
    with (dataset_path / "metadata.json").open("w") as file_handle:
        json.dump({"name": "ViDoSeek-page", "task_type": "visdoc_ood"}, file_handle)

    dataset = _build_dataset(
        dataset_path,
        dataset_name="ViDoSeek-page",
    )

    with pytest.raises(ValueError, match="Missing required raw field for visdoc eval transform"):
        _ = dataset.queries[0]
