"""Public MMEB-V2 evaluation conversion exports.

This package exposes the minimal conversion surface used by the dedicated CLI
and local callers:

- dataset list queries such as dataset names, modality, and pipeline
- concrete image, video, and visdoc conversion functions
- grouped helpers that iterate over all datasets within one pipeline family

The package intentionally stays conversion-only. Runtime loading, benchmark
execution, and parser compatibility logic live outside this namespace.
"""

from .datasets import get_dataset_names, get_modality, get_pipeline
from .image import (
    convert_all_image_cls_datasets,
    convert_all_image_datasets,
    convert_all_image_i2i_vg_datasets,
    convert_all_image_i2t_datasets,
    convert_all_image_qa_datasets,
    convert_all_image_t2i_datasets,
    convert_image_cls_dataset,
    convert_image_i2i_vg_dataset,
    convert_image_i2t_dataset,
    convert_image_qa_dataset,
    convert_image_t2i_dataset,
    convert_imagenet_1k,
)
from .video import (
    convert_all_video_cls_datasets,
    convert_all_video_momentseeker_datasets,
    convert_all_video_mret_datasets,
    convert_all_video_qa_datasets,
    convert_all_video_ret_datasets,
    convert_video_cls_dataset,
    convert_video_momentseeker_dataset,
    convert_video_mret_dataset,
    convert_video_qa_dataset,
    convert_video_ret_dataset,
)
from .visdoc import convert_all_visdoc_datasets, convert_visdoc_dataset

__all__ = [
    "convert_all_image_cls_datasets",
    "convert_all_image_datasets",
    "convert_all_image_i2i_vg_datasets",
    "convert_all_image_i2t_datasets",
    "convert_all_image_qa_datasets",
    "convert_all_image_t2i_datasets",
    "convert_image_cls_dataset",
    "convert_image_i2i_vg_dataset",
    "convert_image_i2t_dataset",
    "convert_image_qa_dataset",
    "convert_image_t2i_dataset",
    "convert_imagenet_1k",
    "convert_all_video_cls_datasets",
    "convert_all_video_momentseeker_datasets",
    "convert_all_video_mret_datasets",
    "convert_all_video_qa_datasets",
    "convert_all_video_ret_datasets",
    "convert_all_visdoc_datasets",
    "convert_video_cls_dataset",
    "convert_video_momentseeker_dataset",
    "convert_video_mret_dataset",
    "convert_video_qa_dataset",
    "convert_video_ret_dataset",
    "convert_visdoc_dataset",
    "get_dataset_names",
    "get_modality",
    "get_pipeline",
]
