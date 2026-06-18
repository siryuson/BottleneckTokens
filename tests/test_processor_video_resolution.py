"""Tests for separate image/video resolution in processor wrappers."""

import numpy as np
from PIL import Image

from vlm2emb.data.processors.qwen2_vl import Qwen2VLProcessorWrapper
from vlm2emb.data.processors.qwen3_vl import Qwen3VLProcessorWrapper


def _make_test_image(size=(500, 500)):
    """Create a test PIL Image."""
    return Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))


class TestQwen2VLVideoResolution:
    """Test Qwen2VLProcessorWrapper with separate image/video resolution."""

    def test_different_image_and_video_max_pixels(self):
        """image and video should have different max_pixels."""
        wrapper = Qwen2VLProcessorWrapper.from_config({
            "type": "qwen2_vl",
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 3136,
            "video_max_pixels": 50000,
        })
        img_proc = wrapper.processor.image_processor
        vid_proc = wrapper.processor.video_processor
        assert img_proc.max_pixels == 200704
        assert vid_proc.max_pixels == 50000

    def test_different_image_and_video_min_pixels(self):
        """image and video should have different min_pixels."""
        wrapper = Qwen2VLProcessorWrapper.from_config({
            "type": "qwen2_vl",
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 784,
            "video_max_pixels": 200704,
        })
        img_proc = wrapper.processor.image_processor
        vid_proc = wrapper.processor.video_processor
        assert img_proc.min_pixels == 3136
        assert vid_proc.min_pixels == 784

    def test_same_values_no_override(self):
        """When image and video have same values, no videos_kwargs needed."""
        wrapper = Qwen2VLProcessorWrapper.from_config({
            "type": "qwen2_vl",
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 3136,
            "video_max_pixels": 200704,  # explicitly same as image
        })
        assert wrapper._video_size is None

    def test_backward_compat_old_params(self):
        """Old min_pixels/max_pixels should still work, applied to both."""
        wrapper = Qwen2VLProcessorWrapper.from_config({
            "type": "qwen2_vl",
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "min_pixels": 3136,
            "max_pixels": 200704,
        })
        img_proc = wrapper.processor.image_processor
        vid_proc = wrapper.processor.video_processor
        assert img_proc.max_pixels == 200704
        # video defaults to 28*28*768=602112, not image_max_pixels
        assert vid_proc.max_pixels == 602112

    def test_video_resolution_actually_applied(self):
        """Video path should produce smaller grid_thw than image path."""
        wrapper = Qwen2VLProcessorWrapper.from_config({
            "type": "qwen2_vl",
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 3136,
            "video_max_pixels": 3136,
        })
        img = _make_test_image()
        image_results = wrapper(
            texts=["<|image_pad|> test"],
            images=[[img]],
        )
        video_results = wrapper(
            texts=["<|video_pad|> test"],
            images=[[img, img]],
        )
        image_grid = image_results[0]["image_grid_thw"]
        video_grid = video_results[0]["video_grid_thw"]
        image_spatial = image_grid[0, 1] * image_grid[0, 2]
        video_spatial = video_grid[0, 1] * video_grid[0, 2]
        assert video_spatial < image_spatial

    def test_save_load_preserves_video_resolution(self, tmp_path):
        """Save and reload should preserve separate image/video resolution."""
        wrapper = Qwen2VLProcessorWrapper.from_config({
            "type": "qwen2_vl",
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 784,
            "video_max_pixels": 50000,
        })
        save_dir = str(tmp_path / "proc")
        wrapper.save_pretrained(save_dir)

        from transformers import AutoProcessor
        reloaded = AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)
        assert reloaded.image_processor.min_pixels == 3136
        assert reloaded.image_processor.max_pixels == 200704
        assert reloaded.video_processor.min_pixels == 784
        assert reloaded.video_processor.max_pixels == 50000

class TestQwen3VLVideoResolution:
    """Test Qwen3VLProcessorWrapper with separate image/video resolution."""

    def test_different_image_and_video_max_pixels(self):
        wrapper = Qwen3VLProcessorWrapper.from_config({
            "type": "qwen3_vl",
            "processor_name": "Qwen/Qwen3-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 3136,
            "video_max_pixels": 50000,
        })
        assert wrapper.processor.image_processor.max_pixels == 200704
        # Qwen3VL video_processor uses size dict
        assert wrapper.processor.video_processor.size["longest_edge"] == 50000

    def test_backward_compat_old_params(self):
        wrapper = Qwen3VLProcessorWrapper.from_config({
            "type": "qwen3_vl",
            "processor_name": "Qwen/Qwen3-VL-2B-Instruct",
            "min_pixels": 3136,
            "max_pixels": 200704,
        })
        assert wrapper.processor.image_processor.max_pixels == 200704
        # video defaults to 28*28*768=602112, not image_max_pixels
        assert wrapper.processor.video_processor.size["longest_edge"] == 602112

    def test_video_resolution_actually_applied(self):
        wrapper = Qwen3VLProcessorWrapper.from_config({
            "type": "qwen3_vl",
            "processor_name": "Qwen/Qwen3-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 3136,
            "video_max_pixels": 3136,
        })
        img = _make_test_image()
        image_results = wrapper(
            texts=["<|image_pad|> test"],
            images=[[img]],
        )
        video_results = wrapper(
            texts=["<|video_pad|> test"],
            images=[[img, img]],
        )
        image_spatial = image_results[0]["image_grid_thw"][0, 1] * image_results[0]["image_grid_thw"][0, 2]
        video_spatial = video_results[0]["video_grid_thw"][0, 1] * video_results[0]["video_grid_thw"][0, 2]
        assert video_spatial < image_spatial

    def test_save_load_preserves_video_resolution(self, tmp_path):
        wrapper = Qwen3VLProcessorWrapper.from_config({
            "type": "qwen3_vl",
            "processor_name": "Qwen/Qwen3-VL-2B-Instruct",
            "image_min_pixels": 3136,
            "image_max_pixels": 200704,
            "video_min_pixels": 784,
            "video_max_pixels": 50000,
        })
        save_dir = str(tmp_path / "proc")
        wrapper.save_pretrained(save_dir)

        from transformers import AutoProcessor
        reloaded = AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)
        assert reloaded.image_processor.min_pixels == 3136
        assert reloaded.image_processor.max_pixels == 200704
        # Qwen3VL video_processor uses size dict, not min_pixels/max_pixels
        assert reloaded.video_processor.size["shortest_edge"] == 784
        assert reloaded.video_processor.size["longest_edge"] == 50000
