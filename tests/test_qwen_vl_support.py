from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from vlm2emb.data.processors import (
    PROCESSOR_CLASS_MAP,
    Qwen2_5VLProcessorWrapper,
    Qwen2VLProcessorWrapper,
    Qwen3VLProcessorWrapper,
    batch_processor_outputs,
)
from vlm2emb.modules.backbone import Qwen2_5VLBackbone, Qwen2VLBackbone, Qwen3VLBackbone
from vlm2emb.training.train import _validate_eval_config


class FakeTokenizer:
    def __init__(self, padding_side: str = "right"):
        self.padding_side = padding_side
        self.deprecation_warnings = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        mapping = {
            "<|image_pad|>": 151655,
            "<|video_pad|>": 151656,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
        }
        return mapping.get(token, -1)

    def pad(self, batch, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        max_len = max(len(ids) for ids in batch["input_ids"])
        input_ids = []
        attention_mask = []
        for ids in batch["input_ids"]:
            pad_len = max_len - len(ids)
            if self.padding_side == "left":
                input_ids.append([0] * pad_len + ids)
                attention_mask.append([0] * pad_len + [1] * len(ids))
            else:
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class FakeQwen3Processor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(min_pixels=3136, max_pixels=1003520)
        self.video_processor = SimpleNamespace(size={"shortest_edge": 3136, "longest_edge": 602112})
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.calls = []

    def __call__(self, text, images=None, videos=None, return_mm_token_type_ids=False, **kwargs):
        self.calls.append({
            "text": text,
            "images": images,
            "videos": videos,
            "return_mm_token_type_ids": return_mm_token_type_ids,
            "kwargs": kwargs,
        })
        result = {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }
        if images is not None:
            result["pixel_values"] = torch.ones(1, 3, 2, 2)
            result["image_grid_thw"] = torch.tensor([[1, 2, 2]], dtype=torch.long)
        if videos is not None:
            result["pixel_values_videos"] = torch.ones(1, 3, 2, 2)
            result["video_grid_thw"] = torch.tensor([[2, 2, 2]], dtype=torch.long)
        if return_mm_token_type_ids:
            result["mm_token_type_ids"] = torch.tensor([[1, 0]], dtype=torch.long)
        return result


class FakeQwen25Processor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(min_pixels=3136, max_pixels=1003520)
        self.video_processor = SimpleNamespace(
            min_pixels=3136,
            max_pixels=602112,
            temporal_patch_size=2,
        )
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.calls = []

    def __call__(self, text, images=None, videos=None, return_mm_token_type_ids=False, **kwargs):
        self.calls.append({
            "text": text,
            "images": images,
            "videos": videos,
            "return_mm_token_type_ids": return_mm_token_type_ids,
            "kwargs": kwargs,
        })
        result = {
            "input_ids": torch.tensor([[21, 22, 23]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        if images is not None:
            result["pixel_values"] = torch.ones(1, 3, 2, 2)
            result["image_grid_thw"] = torch.tensor([[1, 2, 2]], dtype=torch.long)
        if videos is not None:
            result["pixel_values_videos"] = torch.ones(1, 3, 2, 2)
            result["video_grid_thw"] = torch.tensor([[2, 2, 2]], dtype=torch.long)
            result["second_per_grid_ts"] = torch.tensor([0.5], dtype=torch.float)
        if return_mm_token_type_ids:
            result["mm_token_type_ids"] = torch.tensor([[1, 0, 0]], dtype=torch.long)
        return result


class FakeQwen2Processor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(min_pixels=3136, max_pixels=1003520)
        self.video_processor = SimpleNamespace(
            min_pixels=3136,
            max_pixels=602112,
            size={"shortest_edge": 3136, "longest_edge": 602112},
        )
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.calls = []

    def __call__(self, text, images=None, videos=None, **kwargs):
        self.calls.append({
            "text": text,
            "images": images,
            "videos": videos,
            "kwargs": kwargs,
        })
        result = {
            "input_ids": torch.tensor([[31, 32, 33]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        if images is not None:
            result["pixel_values"] = torch.ones(1, 3, 2, 2)
            result["image_grid_thw"] = torch.tensor([[1, 2, 2]], dtype=torch.long)
        if videos is not None:
            result["pixel_values_videos"] = torch.ones(1, 3, 2, 2)
            result["video_grid_thw"] = torch.tensor([[2, 2, 2]], dtype=torch.long)
        return result


class FakeQwen2InnerModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden = torch.randn(2, 3, 6)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,),
            past_key_values="inner-cache",
        )


class FakeQwen2TopModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.model = FakeQwen2InnerModel()
        self.config = SimpleNamespace(
            hidden_size=6,
            text_config=SimpleNamespace(hidden_size=6),
            vision_config=SimpleNamespace(hidden_size=6),
        )

    def forward(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.125), past_key_values="top-cache")


class FakeQwen3InnerModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            last_hidden_state=torch.randn(2, 3, 4),
            past_key_values="inner-cache",
        )


class FakeQwen3TopModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.model = FakeQwen3InnerModel()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=4),
            vision_config=SimpleNamespace(hidden_size=4),
        )

    def forward(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.25), past_key_values="top-cache")


class FakeQwen25InnerModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden = torch.randn(2, 3, 8)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,),
            past_key_values="inner-cache",
        )


class FakeQwen25TopModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.model = FakeQwen25InnerModel()
        self.config = SimpleNamespace(
            hidden_size=8,
            text_config=SimpleNamespace(hidden_size=8),
        )

    def forward(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.5), past_key_values="top-cache")


def test_qwen3_wrapper_preserves_mm_token_type_ids():
    wrapper = Qwen3VLProcessorWrapper(processor=FakeQwen3Processor())
    samples = wrapper(["<|image_pad|> test"], [[object()]])
    assert torch.equal(samples[0]["mm_token_type_ids"], torch.tensor([[1, 0]]))


def test_qwen2_wrapper_canonicalizes_visual_segment_before_processor_call():
    processor = FakeQwen2Processor()
    wrapper = Qwen2VLProcessorWrapper(processor=processor)
    wrapper(["<|image_pad|> describe the page"], [[object()]])
    assert processor.calls[-1]["text"] == ["<|vision_start|><|image_pad|><|vision_end|> describe the page"]


def test_qwen2_wrapper_can_disable_visual_boundary_wrapping():
    processor = FakeQwen2Processor()
    wrapper = Qwen2VLProcessorWrapper(
        processor=processor,
        wrap_visual_tokens_with_boundaries=False,
    )
    wrapper(["<|image_pad|> describe the page"], [[object()]])
    assert processor.calls[-1]["text"] == ["<|image_pad|> describe the page"]


def test_qwen25_wrapper_canonicalizes_visual_segment_before_processor_call():
    processor = FakeQwen25Processor()
    wrapper = Qwen2_5VLProcessorWrapper(processor=processor)
    wrapper(["<|video_pad|> describe the clip"], [[object(), object()]])
    assert processor.calls[-1]["text"] == ["<|vision_start|><|video_pad|><|vision_end|> describe the clip"]


def test_qwen25_wrapper_can_disable_visual_boundary_wrapping():
    processor = FakeQwen25Processor()
    wrapper = Qwen2_5VLProcessorWrapper(
        processor=processor,
        wrap_visual_tokens_with_boundaries=False,
    )
    wrapper(["<|video_pad|> describe the clip"], [[object(), object()]])
    assert processor.calls[-1]["text"] == ["<|video_pad|> describe the clip"]


def test_qwen3_wrapper_canonicalizes_visual_segment_before_processor_call():
    processor = FakeQwen3Processor()
    wrapper = Qwen3VLProcessorWrapper(processor=processor)
    wrapper(["<|image_pad|> test"], [[object()]])
    assert processor.calls[-1]["text"] == ["<|vision_start|><|image_pad|><|vision_end|> test"]


def test_qwen3_wrapper_can_disable_visual_boundary_wrapping():
    processor = FakeQwen3Processor()
    wrapper = Qwen3VLProcessorWrapper(
        processor=processor,
        wrap_visual_tokens_with_boundaries=False,
    )
    wrapper(["<|image_pad|> test"], [[object()]])
    assert processor.calls[-1]["text"] == ["<|image_pad|> test"]


def test_qwen3_wrapper_passes_archive_compatible_video_metadata():
    processor = FakeQwen3Processor()
    wrapper = Qwen3VLProcessorWrapper(processor=processor)
    frames = [object(), object(), object()]
    wrapper(["<|video_pad|> test"], [frames])
    assert processor.calls[-1]["kwargs"]["video_metadata"] == [
        {"fps": 1.0, "total_num_frames": 3, "frames_indices": [0, 1, 2]}
    ]


def test_qwen3_wrapper_prefers_dataset_video_metadata_when_provided():
    processor = FakeQwen3Processor()
    wrapper = Qwen3VLProcessorWrapper(processor=processor)
    frames = [object(), object()]
    wrapper(
        ["<|video_pad|> test"],
        [frames],
        media_metadata=[{"fps": 7.5, "total_num_frames": 24, "sampled_indices": [0, 4]}],
    )
    assert processor.calls[-1]["kwargs"]["video_metadata"] == [
        {"fps": 7.5, "total_num_frames": 24, "frames_indices": [0, 4]}
    ]


def test_qwen3_wrapper_repeats_single_video_frame_for_temporal_factor():
    processor = FakeQwen3Processor()
    wrapper = Qwen3VLProcessorWrapper(processor=processor)
    frame = object()
    wrapper(
        ["<|video_pad|> test"],
        [[frame]],
        media_metadata=[{"fps": 7.5, "total_num_frames": 24, "sampled_indices": [4]}],
    )

    assert processor.calls[-1]["videos"] == [[frame, frame]]
    assert processor.calls[-1]["kwargs"]["video_metadata"] == [
        {"fps": 7.5, "total_num_frames": 24, "frames_indices": [4, 4]}
    ]


def test_qwen2_backbone_uses_last_hidden_state_without_forcing_hidden_states():
    model = FakeQwen2TopModel()
    backbone = Qwen2VLBackbone(model=model)
    output = backbone(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
    )
    assert output["last_hidden_state"].shape == (2, 3, 6)
    assert output["hidden_states"] is None
    assert model.model.calls[-1]["output_hidden_states"] is False
    assert model.model.calls[-1]["return_dict"] is True


def test_qwen2_backbone_returns_hidden_states_when_explicitly_requested():
    model = FakeQwen2TopModel()
    backbone = Qwen2VLBackbone(model=model)
    output = backbone(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        output_hidden_states=True,
    )
    assert output["hidden_states"] is not None
    assert model.model.calls[-1]["output_hidden_states"] is True


def test_qwen2_backbone_rejects_missing_absolute_local_checkpoint(tmp_path):
    missing_checkpoint = tmp_path / "missing-checkpoint"
    try:
        Qwen2VLBackbone.from_pretrained(str(missing_checkpoint))
    except FileNotFoundError as err:
        assert "Local checkpoint path does not exist" in str(err)
        assert str(missing_checkpoint) in str(err)
    else:
        raise AssertionError("Expected FileNotFoundError for missing local checkpoint")


def test_qwen2_wrapper_raises_when_media_has_no_visual_token():
    wrapper = Qwen2VLProcessorWrapper(processor=FakeQwen2Processor())
    try:
        wrapper(["describe the page"], [[object()]])
    except ValueError as err:
        assert "no visual token" in str(err)
    else:
        raise AssertionError("Expected ValueError for media sample without visual token")


def test_qwen2_wrapper_raises_when_visual_token_has_no_media():
    wrapper = Qwen2VLProcessorWrapper(processor=FakeQwen2Processor())
    try:
        wrapper(["<|image_pad|> describe the page"], [[]])
    except ValueError as err:
        assert "no media provided" in str(err)
    else:
        raise AssertionError("Expected ValueError for visual token without media")


def test_qwen2_wrapper_raises_when_both_visual_tokens_present():
    wrapper = Qwen2VLProcessorWrapper(processor=FakeQwen2Processor())
    try:
        wrapper(["<|video_pad|>\n<|image_pad|> mixed"], [[object()]])
    except ValueError as err:
        assert "Ambiguous Qwen visual sample" in str(err)
    else:
        raise AssertionError("Expected ValueError for ambiguous visual sample")


def test_qwen2_wrapper_rejects_missing_absolute_local_processor(tmp_path):
    missing_processor = tmp_path / "missing-processor"
    try:
        Qwen2VLProcessorWrapper.from_config({"processor_name": str(missing_processor)})
    except FileNotFoundError as err:
        assert "Local processor path does not exist" in str(err)
        assert str(missing_processor) in str(err)
    else:
        raise AssertionError("Expected FileNotFoundError for missing local processor")


def test_batch_processor_outputs_pads_mm_token_type_ids_and_cats_second_per_grid_ts():
    wrapper = SimpleNamespace(tokenizer=FakeTokenizer())
    samples = [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[1, 0, 0]], dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
            "second_per_grid_ts": None,
        },
        {
            "input_ids": torch.tensor([[4, 5]], dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[1, 0]], dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
            "second_per_grid_ts": torch.tensor([0.5], dtype=torch.float),
        },
    ]

    batch = batch_processor_outputs(samples, wrapper)
    assert torch.equal(
        batch["mm_token_type_ids"],
        torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.long),
    )
    assert torch.equal(batch["second_per_grid_ts"], torch.tensor([0.5], dtype=torch.float))


def test_batch_processor_outputs_pads_generation_loss_mask():
    wrapper = SimpleNamespace(tokenizer=FakeTokenizer())
    samples = [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "generation_loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
        },
        {
            "input_ids": torch.tensor([[4, 5]], dtype=torch.long),
            "generation_loss_mask": torch.tensor([[0, 1]], dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
        },
    ]
    batch = batch_processor_outputs(samples, wrapper)
    assert torch.equal(
        batch["generation_loss_mask"],
        torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.long),
    )


def test_batch_processor_outputs_keeps_qwen2_compatibility_when_new_fields_absent():
    wrapper = SimpleNamespace(tokenizer=FakeTokenizer())
    samples = [
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
        }
    ]

    batch = batch_processor_outputs(samples, wrapper)
    assert batch["mm_token_type_ids"] is None
    assert batch["second_per_grid_ts"] is None


def test_qwen3_backbone_uses_inner_model_for_embeddings():
    backbone = Qwen3VLBackbone(model=FakeQwen3TopModel())
    output = backbone(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
    )
    assert output["last_hidden_state"].shape == (2, 3, 4)
    assert output["hidden_states"] is None
    assert backbone.model.model.calls
    assert backbone.model.model.calls[-1]["return_dict"] is True


def test_qwen3_backbone_raises_when_requesting_hidden_states():
    backbone = Qwen3VLBackbone(model=FakeQwen3TopModel())
    try:
        backbone(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            output_hidden_states=True,
        )
    except NotImplementedError as err:
        assert "does not support output_hidden_states" in str(err)
    else:
        raise AssertionError("Expected NotImplementedError for Qwen3 hidden states request")


def test_qwen25_wrapper_keeps_second_per_grid_ts_and_mm_token_type_ids():
    wrapper = Qwen2_5VLProcessorWrapper(processor=FakeQwen25Processor())
    samples = wrapper(["<|video_pad|> test"], [[object(), object()]])
    assert torch.equal(samples[0]["second_per_grid_ts"], torch.tensor([0.5], dtype=torch.float))
    assert torch.equal(samples[0]["mm_token_type_ids"], torch.tensor([[1, 0, 0]], dtype=torch.long))


def test_qwen25_wrapper_prefers_dataset_video_fps_when_provided():
    processor = FakeQwen25Processor()
    wrapper = Qwen2_5VLProcessorWrapper(processor=processor, fps=2.0)
    wrapper(
        ["<|video_pad|> test"],
        [[object(), object()]],
        media_metadata=[{"fps": 6.0}],
    )
    assert processor.calls[-1]["kwargs"]["videos_kwargs"]["fps"] == 6.0


def test_qwen25_wrapper_raises_when_media_has_no_visual_token():
    wrapper = Qwen2_5VLProcessorWrapper(processor=FakeQwen25Processor())
    try:
        wrapper(["describe the clip"], [[object(), object()]])
    except ValueError as err:
        assert "no visual token" in str(err)
    else:
        raise AssertionError("Expected ValueError for media sample without visual token")


def test_qwen3_wrapper_raises_when_both_visual_tokens_present():
    wrapper = Qwen3VLProcessorWrapper(processor=FakeQwen3Processor())
    try:
        wrapper(["<|video_pad|>\n<|image_pad|> mixed"], [[object()]])
    except ValueError as err:
        assert "Ambiguous Qwen visual sample" in str(err)
    else:
        raise AssertionError("Expected ValueError for ambiguous visual sample")


def test_qwen25_backbone_forwards_second_per_grid_ts():
    model = FakeQwen25TopModel()
    backbone = Qwen2_5VLBackbone(model=model)
    seconds = torch.tensor([0.5], dtype=torch.float)
    output = backbone(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        second_per_grid_ts=seconds,
        output_hidden_states=True,
    )
    assert output["last_hidden_state"].shape == (2, 3, 8)
    assert torch.equal(output["second_per_grid_ts"], seconds)
    assert model.model.calls[-1]["second_per_grid_ts"] is seconds


def test_qwen25_config_files_reference_3b_model_and_wrapper():
    model_cfg = OmegaConf.load("configs/models/vlm2vec_qwen25vl_3b.yaml")
    preset_cfg = OmegaConf.load("configs/presets/vlm2vec_qwen25vl_3b.yaml")

    assert model_cfg.modules.backbone.type == "Qwen2_5VLBackbone"
    assert model_cfg.modules.backbone.model_name_or_path == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert preset_cfg.processor.type == "qwen2_5_vl"
    assert preset_cfg.processor.processor_name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert preset_cfg.eval.type == "mmeb"
    assert preset_cfg.eval.data_path == "./data/MMEB-V2"
    assert preset_cfg.eval.batch_size == 8
    assert preset_cfg.eval.num_workers == 0


def test_btoks_qwen3_dataset_v1_preset_references_qwen3_and_btoks_v1():
    model_cfg = OmegaConf.load("configs/models/btoks_qwen3vl_2b_v1.yaml")
    preset_cfg = OmegaConf.load("configs/presets/btoks_qwen3vl_2b_v1_expanded_train_v1.yaml")

    assert model_cfg.modules.injector.type == "BToksTokenInjector"
    assert model_cfg.modules.backbone.type == "Qwen3VLBackbone"
    assert model_cfg.modules.backbone.model_name_or_path.endswith("Qwen3-VL-2B-Instruct")
    assert model_cfg.modules.pooling.type == "BToksPooling"
    assert preset_cfg.processor.type == "qwen3_vl"
    assert preset_cfg.processor.processor_name.endswith("Qwen3-VL-2B-Instruct")
    assert preset_cfg.train.trainer.type == "btoks_trainer"
    assert preset_cfg.train.dataset.datasets._inherit_ == "../datasets/btoks_expanded_train_datasets_v1.yaml"
    assert preset_cfg.train.args.run_name == "btoks_qwen3vl_2b_v1_expanded_train_v1"


def test_btoks_qwen3_dataset_v2_preset_references_qwen3_and_btoks_v2():
    model_cfg = OmegaConf.load("configs/models/btoks_qwen3vl_2b_v1.yaml")
    preset_cfg = OmegaConf.load("configs/presets/btoks_qwen3vl_2b_v1_expanded_train_v2.yaml")

    assert model_cfg.modules.injector.type == "BToksTokenInjector"
    assert model_cfg.modules.backbone.type == "Qwen3VLBackbone"
    assert model_cfg.modules.backbone.model_name_or_path.endswith("Qwen3-VL-2B-Instruct")
    assert model_cfg.modules.pooling.type == "BToksPooling"
    assert preset_cfg.processor.type == "qwen3_vl"
    assert preset_cfg.processor.processor_name.endswith("Qwen3-VL-2B-Instruct")
    assert preset_cfg.train.trainer.type == "btoks_trainer"
    assert preset_cfg.train.dataset.datasets._inherit_ == "../datasets/btoks_expanded_train_datasets_v2.yaml"
    assert preset_cfg.train.args.run_name == "btoks_qwen3vl_2b_v1_expanded_train_v2"


def test_qwen25_registry_entry_exists():
    assert PROCESSOR_CLASS_MAP["Qwen2_5_VLProcessor"] == "qwen2_5_vl"


def test_validate_eval_config_requires_type():
    try:
        _validate_eval_config({"batch_size": 16})
    except ValueError as err:
        assert "missing required 'type'" in str(err)
    else:
        raise AssertionError("Expected ValueError for eval config without type")
