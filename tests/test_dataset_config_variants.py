import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

import vlm2emb.data.datasets  # noqa: F401
from vlm2emb.auto import AutoDataset
from vlm2emb.config import load_config
from vlm2emb.data.datasets.combine import load_combined_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEBQA_TRANSFORM = {
    "query": {
        "visual_token_alignment": "strip_without_media",
        "trailing_newline": "ensure_single",
    },
    "positive": {
        "visual_token_placement": "own_line",
        "trailing_newline": "ensure_single",
    },
    "negative": {"empty": "empty_multimodal_input"},
}
EXPANDED_VIDEO_TYPES = {
    "ssv2_train",
    "hmdb51_train",
    "ucf101_train",
    "msrvtt_train",
    "msvd_train",
    "didemo_train",
    "youcook2_train",
    "activitynet_captions_train",
    "breakfast_train",
    "charades_sta_train",
    "qvhighlight_train",
    "nextqa_train",
    "llavahound_train",
    "kinetics700_train",
}
EXPANDED_MMEB_VISDOC_SUBSETS = {"DocVQA", "InfographicsVQA", "ChartQA"}


def _assert_no_inherit(node: object) -> None:
    if isinstance(node, DictConfig):
        assert "_inherit_" not in node
        for value in node.values():
            _assert_no_inherit(value)
        return

    if isinstance(node, dict):
        assert "_inherit_" not in node
        for value in node.values():
            _assert_no_inherit(value)
        return

    if isinstance(node, list):
        for item in node:
            _assert_no_inherit(item)


def _load_repo_config(relative_path: str):
    return load_config(REPO_ROOT / relative_path)


def _plain_config(node):
    return OmegaConf.to_container(node, resolve=True)


def _expanded_config_sample_counts() -> dict[str, int]:
    text = (REPO_ROOT / "configs/datasets/expanded_train_datasets_v1.yaml").read_text()
    samples: dict[str, int] = {}
    current_name: str | None = None
    last_sample: int | None = None

    for line in text.splitlines():
        name_match = re.match(r"([A-Za-z0-9_.\-]+):\s*$", line)
        if name_match and not line.startswith(" "):
            current_name = name_match.group(1)
            last_sample = None
            continue

        sample_match = re.match(r"\s*# samples: ([0-9,]+)", line)
        if sample_match:
            last_sample = int(sample_match.group(1).replace(",", ""))
            continue

        if current_name and last_sample is not None and re.match(r"\s*weight:", line):
            samples[current_name] = last_sample
            last_sample = None

    return samples


def _expanded_modality(config_name: str, dataset_config) -> str:
    dataset_type = dataset_config.type
    if dataset_type in {"vidore_train", "vidore_rag_train", "visrag_train", "mmlongbench_doc_train"}:
        return "visdoc"
    if (
        dataset_type == "mmeb_train"
        and config_name in EXPANDED_MMEB_VISDOC_SUBSETS
    ):
        return "visdoc"
    if dataset_type in EXPANDED_VIDEO_TYPES:
        return "video"
    return "image"


def _assert_dataset_fields_match(
    base_dataset,
    variant_dataset,
) -> None:
    assert variant_dataset.type == base_dataset.type
    assert variant_dataset.path == base_dataset.path
    assert variant_dataset.subset == base_dataset.subset
    if "transform" in base_dataset:
        assert _plain_config(variant_dataset.transform) == _plain_config(base_dataset.transform)


def _without_metadata(dataset_config):
    payload = dict(_plain_config(dataset_config))
    payload.pop("metadata", None)
    return payload


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        return index


def test_btoks_variant_only_adds_ntp_side_delta() -> None:
    base_config = _load_repo_config("configs/datasets/mmeb_train.yaml")
    btoks_config = _load_repo_config("configs/datasets/btoks_mmeb_train.yaml")

    _assert_no_inherit(btoks_config)

    _assert_dataset_fields_match(base_config.ImageNet_1K, btoks_config.ImageNet_1K)
    assert "metadata" not in base_config.ImageNet_1K
    assert btoks_config.ImageNet_1K.weight == base_config.ImageNet_1K.weight
    assert btoks_config.ImageNet_1K.metadata.ntp_side == "positive"

    _assert_dataset_fields_match(base_config.VisDial, btoks_config.VisDial)
    assert btoks_config.VisDial.weight == base_config.VisDial.weight
    assert btoks_config.VisDial.metadata.ntp_side == "query"

    _assert_dataset_fields_match(base_config.WebQA, btoks_config.WebQA)
    assert btoks_config.WebQA.weight == base_config.WebQA.weight
    assert btoks_config.WebQA.metadata.ntp_side == "both"
    assert btoks_config.CIRR.metadata.ntp_side == "none"
    assert btoks_config.NIGHTS.metadata.ntp_side == "none"
    assert btoks_config.MSCOCO.metadata.ntp_side == "query"


def test_optimized_variant_only_overrides_weight() -> None:
    base_config = _load_repo_config("configs/datasets/mmeb_train.yaml")
    optimized_config = _load_repo_config("configs/datasets/mmeb_train_optimized.yaml")

    _assert_no_inherit(optimized_config)

    _assert_dataset_fields_match(base_config.ImageNet_1K, optimized_config.ImageNet_1K)
    assert "metadata" not in optimized_config.ImageNet_1K
    assert optimized_config.ImageNet_1K.weight == 3
    assert optimized_config.ImageNet_1K.weight != base_config.ImageNet_1K.weight

    _assert_dataset_fields_match(base_config.VisDial, optimized_config.VisDial)
    assert optimized_config.VisDial.weight == 3

    _assert_dataset_fields_match(base_config.WebQA, optimized_config.WebQA)
    assert optimized_config.WebQA.weight == 1


def test_btoks_optimized_variant_preserves_btoks_metadata_and_optimized_weights() -> None:
    base_config = _load_repo_config("configs/datasets/mmeb_train.yaml")
    btoks_config = _load_repo_config("configs/datasets/btoks_mmeb_train.yaml")
    btoks_optimized_config = _load_repo_config(
        "configs/datasets/btoks_mmeb_train_optimized.yaml"
    )

    _assert_no_inherit(btoks_optimized_config)

    _assert_dataset_fields_match(
        base_config.ImageNet_1K,
        btoks_optimized_config.ImageNet_1K,
    )
    assert btoks_optimized_config.ImageNet_1K.weight == 3
    assert btoks_optimized_config.ImageNet_1K.metadata.ntp_side == (
        btoks_config.ImageNet_1K.metadata.ntp_side
    )

    _assert_dataset_fields_match(base_config.VisDial, btoks_optimized_config.VisDial)
    assert btoks_optimized_config.VisDial.weight == 3
    assert btoks_optimized_config.VisDial.metadata.ntp_side == (
        btoks_config.VisDial.metadata.ntp_side
    )

    _assert_dataset_fields_match(base_config.WebQA, btoks_optimized_config.WebQA)
    assert btoks_optimized_config.WebQA.weight == 1
    assert btoks_optimized_config.WebQA.metadata.ntp_side == (
        btoks_config.WebQA.metadata.ntp_side
    )


def test_presets_resolve_without_residual_inherit_keys() -> None:
    vlm2vec_preset = _load_repo_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
    expanded_preset = _load_repo_config(
        "configs/presets/vlm2vec_qwen2vl_2b_expanded_train_v1.yaml"
    )
    btoks_preset = _load_repo_config("configs/presets/btoks_qwen2vl_2b_v2.yaml")
    btoks_expanded_preset = _load_repo_config(
        "configs/presets/btoks_qwen2vl_2b_v1_expanded_train_v1.yaml"
    )
    btoks_qwen3_v2_expanded_preset = _load_repo_config(
        "configs/presets/btoks_qwen3vl_2b_v1_expanded_train_v2.yaml"
    )

    _assert_no_inherit(vlm2vec_preset)
    _assert_no_inherit(expanded_preset)
    _assert_no_inherit(btoks_preset)
    _assert_no_inherit(btoks_expanded_preset)
    _assert_no_inherit(btoks_qwen3_v2_expanded_preset)

    vlm2vec_datasets = vlm2vec_preset.train.dataset.datasets
    expanded_datasets = expanded_preset.train.dataset.datasets
    btoks_datasets = btoks_preset.train.dataset.datasets

    assert vlm2vec_datasets.ImageNet_1K.weight == 3
    assert "metadata" not in vlm2vec_datasets.ImageNet_1K
    assert vlm2vec_datasets.VisDial.type == "mmeb_train"
    assert "render" not in vlm2vec_datasets.VisDial
    assert "render" not in vlm2vec_datasets.WebQA

    assert btoks_datasets.ImageNet_1K.weight == 1
    assert btoks_datasets.ImageNet_1K.metadata.ntp_side == "positive"
    assert btoks_datasets.VisDial.metadata.ntp_side == "query"
    assert btoks_datasets.VisDial.type == "mmeb_train"
    assert "render" not in btoks_datasets.VisDial
    assert btoks_datasets.WebQA.metadata.ntp_side == "both"
    assert btoks_datasets.CIRR.metadata.ntp_side == "none"
    assert btoks_datasets.NIGHTS.metadata.ntp_side == "none"
    assert btoks_datasets.MSCOCO.metadata.ntp_side == "query"
    assert "render" not in btoks_datasets.WebQA

    assert (
        expanded_preset.train.args.run_name
        == "vlm2vec_qwen2vl_2b_expanded_train_v1"
    )
    assert len(expanded_datasets) == 62
    assert "MMLongBench-doc" not in expanded_datasets
    assert expanded_datasets.GQA.weight == 7.3

    btoks_expanded_datasets = btoks_expanded_preset.train.dataset.datasets
    assert btoks_expanded_preset.train.trainer.type == "btoks_trainer"
    assert btoks_expanded_preset.model.modules.pooling.type == "BToksPooling"
    assert btoks_expanded_preset.processor.wrap_visual_tokens_with_boundaries is True
    assert (
        btoks_expanded_preset.train.args.run_name
        == "btoks_qwen2vl_2b_v1_expanded_train_v1"
    )
    assert len(btoks_expanded_datasets) == 62
    assert btoks_expanded_datasets.GQA.metadata.ntp_side == "positive"
    assert btoks_expanded_datasets["MSR-VTT"].metadata.ntp_side == "query"
    assert btoks_expanded_datasets["MSR-VTT"].weight == 0.3
    assert btoks_expanded_datasets.MSVD.transform.query.caption_selection == "random"
    assert btoks_expanded_datasets.FashionIQ_train.metadata.ntp_side == "none"

    btoks_qwen3_v2_expanded_datasets = btoks_qwen3_v2_expanded_preset.train.dataset.datasets
    assert btoks_qwen3_v2_expanded_preset.train.trainer.type == "btoks_trainer"
    assert btoks_qwen3_v2_expanded_preset.model.modules.pooling.type == "BToksPooling"
    assert btoks_qwen3_v2_expanded_preset.processor.type == "qwen3_vl"
    assert (
        btoks_qwen3_v2_expanded_preset.train.args.run_name
        == "btoks_qwen3vl_2b_v1_expanded_train_v2"
    )
    assert len(btoks_qwen3_v2_expanded_datasets) == 68
    assert btoks_qwen3_v2_expanded_datasets.VizWiz.metadata.ntp_side == "positive"
    assert btoks_qwen3_v2_expanded_datasets.RefCOCO.metadata.ntp_side == "none"
    assert btoks_qwen3_v2_expanded_datasets.VATEX.metadata.ntp_side == "query"
    assert btoks_qwen3_v2_expanded_datasets.K700.metadata.ntp_side == "positive"
    assert btoks_qwen3_v2_expanded_datasets["VideoChat2-IT"].metadata.ntp_side == "positive"


def test_mmeb_train_default_entrypoint_uses_combined_and_legacy_experiment_keeps_batch_interleave() -> None:
    vlm2vec_preset = _load_repo_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
    legacy_experiment = _load_repo_config("configs/experiments/vlm2vec.yaml")

    _assert_no_inherit(vlm2vec_preset)
    _assert_no_inherit(legacy_experiment)

    assert AutoDataset.is_registered("combined") is True
    assert AutoDataset.is_registered("batch_interleave") is False
    assert vlm2vec_preset.train.dataset.type == "combined"
    assert legacy_experiment.train.dataset.type == "batch_interleave"


def test_mmeb_train_configs_use_new_registry_without_render_overrides() -> None:
    base_config = _load_repo_config("configs/datasets/mmeb_train.yaml")

    mmeb_entries = {}
    for dataset_name, dataset_config in base_config.items():
        if dataset_name.startswith("_"):
            continue
        if dataset_config.type != "mmeb_train":
            continue
        mmeb_entries[dataset_name] = dataset_config

    assert len(mmeb_entries) == 20
    assert all("render" not in dataset_config for dataset_config in mmeb_entries.values())
    assert _plain_config(mmeb_entries["WebQA"].transform) == EXPECTED_WEBQA_TRANSFORM
    assert mmeb_entries["VisDial"].transform.query.trailing_newline == "normalize_blank_tail"
    assert (
        mmeb_entries["ChartQA"].transform.positive.whitespace_normalization
        == "strip_trailing_horizontal"
    )


def test_new_video_training_entries_build_from_all_train_config() -> None:
    config = _load_repo_config("configs/datasets/expanded_train_datasets_v1.yaml")

    expected = {
        "NEXTQA": ("nextqa_train", "NExTQA"),
        "YouCook2": ("youcook2_train", "YouCook2"),
        "ActivityNet-Captions": ("activitynet_captions_train", "ActivityNet-Captions"),
    }
    for config_name, (dataset_type, dataset_name) in expected.items():
        dataset_config = config[config_name]
        assert dataset_config.type == dataset_type
        dataset = AutoDataset.from_config(dataset_config)
        assert dataset.dataset_name == dataset_name
        assert dataset.split == "official_train_without_mmeb_v2_eval"


def test_btoks_expanded_variant_only_adds_ntp_side_metadata() -> None:
    base_config = _load_repo_config("configs/datasets/expanded_train_datasets_v1.yaml")
    btoks_config = _load_repo_config("configs/datasets/btoks_expanded_train_datasets_v1.yaml")

    _assert_no_inherit(btoks_config)
    assert len(btoks_config) == len(base_config) == 62

    for dataset_name, base_dataset in base_config.items():
        btoks_dataset = btoks_config[dataset_name]
        assert _without_metadata(btoks_dataset) == _plain_config(base_dataset)
        assert btoks_dataset.metadata.ntp_side in {"query", "positive", "both", "none"}

    assert btoks_config.ImageNet_1K.metadata.ntp_side == "positive"
    assert btoks_config.VisDial.metadata.ntp_side == "query"
    assert btoks_config.WebQA.metadata.ntp_side == "both"
    assert btoks_config.CIRR.metadata.ntp_side == "none"
    assert btoks_config.colpali_train_set.metadata.ntp_side == "positive"
    assert btoks_config.colpali_train_set_arxiv_qa.metadata.ntp_side == "query"
    assert btoks_config.colpali_train_set_docvqa.metadata.ntp_side == "query"
    assert btoks_config["MSR-VTT"].metadata.ntp_side == "query"
    assert btoks_config.NEXTQA.metadata.ntp_side == "positive"
    assert btoks_config.OVEN_it2t_train.metadata.ntp_side == "positive"
    assert btoks_config.EDIS_train.metadata.ntp_side == "query"
    assert btoks_config.visual7W_pointing_train.metadata.ntp_side == "none"


def test_expanded_v2_config_is_trainable_v1_plus_enabled_v2_entries() -> None:
    v1_config = _load_repo_config("configs/datasets/expanded_train_datasets_v1.yaml")
    config = _load_repo_config("configs/datasets/expanded_train_datasets_v2.yaml")
    btoks_config = _load_repo_config("configs/datasets/btoks_expanded_train_datasets_v2.yaml")

    enabled_v2_names = {
        "VizWiz",
        "RefCOCO",
        "RefCOCO-Matching",
        "VATEX",
        "K700",
        "VideoChat2-IT",
    }
    excluded_v2_candidates = {
        "ViDoSeek-page",
        "ViDoSeek-doc",
        "MMLongBench-page",
    }
    assert set(v1_config.keys()).issubset(set(config.keys()))
    assert set(config.keys()) == set(v1_config.keys()) | enabled_v2_names
    assert set(btoks_config.keys()) == set(config.keys())
    assert excluded_v2_candidates.isdisjoint(config.keys())
    assert "NEXTQA" in config
    assert "YouCook2" in config
    assert "ActivityNet-Captions" in config

    for dataset_name, dataset_config in config.items():
        assert "type" in dataset_config
        assert "path" in dataset_config

    assert config.RefCOCO.type == "refcoco_train"
    assert config.VATEX.type == "vatex_train"
    assert config["VideoChat2-IT"].type == "videochat2_it_train"
    assert "metadata" not in config.VizWiz
    assert "metadata" not in config.RefCOCO
    assert "metadata" not in config.VATEX
    assert "metadata" not in config["VideoChat2-IT"]

    for dataset_name in enabled_v2_names:
        btoks_dataset = btoks_config[dataset_name]
        expected_payload = _plain_config(config[dataset_name])
        expected_payload.setdefault("metadata", {})
        expected_payload["metadata"]["ntp_side"] = btoks_dataset.metadata.ntp_side
        assert _plain_config(btoks_dataset) == expected_payload

    assert btoks_config["MSR-VTT"].metadata.ntp_side == "query"


def test_expanded_v3_config_lists_stage1_candidates_without_enabling_them() -> None:
    v2_config = _load_repo_config("configs/datasets/expanded_train_datasets_v2.yaml")
    config = _load_repo_config("configs/datasets/expanded_train_datasets_v3.yaml")

    assert set(v2_config.keys()).issubset(set(config.keys()))
    assert "_stage1_candidates" in config
    assert config._stage1_candidates.image_text_pre_alignment.priority[0].dataset == "CC12M"
    assert config._stage1_candidates.video_pre_alignment_and_retrieval.priority[0].dataset == "WebVid-10M"
    video_candidates = config._stage1_candidates.video_pre_alignment_and_retrieval.priority
    videochat2_it = next(
        entry
        for entry in video_candidates
        if entry.dataset == "OpenGVLab/VideoChat2-IT-clean"
    )
    assert videochat2_it.status == "partial_enabled_v2_remaining_candidate"
    assert videochat2_it.total_valid_samples == 877_489
    assert videochat2_it.split_policy == "separate_subsets"
    videochat2_subsets = {subset.name: subset for subset in videochat2_it.subsets}
    assert {name: subset.valid_samples for name, subset in videochat2_subsets.items()} == {
        "textvr": 39_648,
        "youcook2": 8_700,
        "k710": 38_977,
        "ssv2": 40_000,
        "videochat2": 9_584,
        "videochatgpt": 13_303,
        "next_qa": 34_132,
        "clevrer_qa": 40_000,
        "clevrer_mc": 40_000,
        "ego_qa": 7_797,
        "tgif_frame_qa": 39_149,
        "tgif_transition_qa": 52_696,
        "webvid": 399_740,
        "videochat": 6_889,
        "videochat1": 4_300,
        "webvid_qa": 99_954,
    }
    assert videochat2_subsets["ego_qa"].decision == "keep_candidate_without_ego4d_full"
    assert videochat2_subsets["next_qa"].decision == "enabled_media_backed_rows"
    assert config._stage1_candidates.eval_only_or_diagnostic.priority[0].dataset == "CIRCO"
    assert "LAION-400M" not in config
    assert "WebVid-10M" not in config
    assert "MagicLens" not in config

    trainable_entries = {
        name: dataset_config
        for name, dataset_config in config.items()
        if not str(name).startswith("_")
    }
    assert set(trainable_entries.keys()) == set(v2_config.keys())
    for dataset_config in trainable_entries.values():
        assert "type" in dataset_config
        assert "path" in dataset_config


def test_combined_dataset_ignores_metadata_keys(monkeypatch) -> None:
    seen_configs = []

    def _fake_from_config(config):
        seen_configs.append(dict(config))
        return _TinyDataset()

    monkeypatch.setattr(AutoDataset, "from_config", _fake_from_config)

    dataset = load_combined_dataset(
        OmegaConf.create(
            {
                "_stage1_candidates": {"priority": [{"dataset": "CC12M"}]},
                "Tiny": {"type": "tiny_train", "path": "/tmp/tiny", "weight": 2.0},
            }
        )
    )

    assert dataset.names == ["Tiny"]
    assert dataset.weights == [2.0]
    assert seen_configs == [{"type": "tiny_train", "path": "/tmp/tiny"}]


def test_expanded_train_weights_keep_all_exhausted_epoch_bounded() -> None:
    config = _load_repo_config("configs/datasets/expanded_train_datasets_v1.yaml")
    samples = _expanded_config_sample_counts()
    weights = {name: float(config[name].weight) for name in samples}
    total_weight = sum(weights.values())

    assert min(weights.values()) >= 0.1
    expected_epoch_rows = max(
        sample_count / (weights[name] / total_weight)
        for name, sample_count in samples.items()
    )
    modality_weights = {"image": 0.0, "video": 0.0, "visdoc": 0.0}
    for name, weight in weights.items():
        modality_weights[_expanded_modality(name, config[name])] += weight

    assert len(weights) == 62
    assert sum(samples.values()) == 6_466_803
    assert expected_epoch_rows < 15_000_000
    assert 45 <= modality_weights["image"] <= 47
    assert 16 <= modality_weights["video"] <= 18
    assert 31 <= modality_weights["visdoc"] <= 32
    assert config.GQA.weight == 7.3
    assert "K700" not in config
