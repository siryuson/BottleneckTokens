"""Shared low-level utilities for MMEB-V2 conversion outputs.

The helpers in this module stay intentionally small:

- create output directories
- resolve one dataset-local metadata task type written on disk
- write minimal ``metadata.json`` records for converted artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

_TASK_TYPE_BY_PIPELINE: dict[str, str] = {
    "image_cls": "image_classification",
    "image_qa": "image_question_answer",
    "image_i2t": "image_retrieval",
    "image_t2i": "image_retrieval",
    "video_cls": "video_classification",
    "video_qa": "video_question_answer",
    "video_text_retrieval": "video_retrieval",
    "video_moment_retrieval": "video_moment_retrieval",
    "video_momentseeker": "video_moment_retrieval",
    "visdoc_beir": "",
}

_IMAGE_I2I_VG_RETRIEVAL_DATASETS = {
    "CIRR",
    "NIGHTS",
    "OVEN",
    "FashionIQ",
}

_IMAGE_I2I_VG_GROUNDING_DATASETS = {
    "Visual7W-Pointing",
    "MSCOCO",
    "RefCOCO",
    "RefCOCO-Matching",
}


def ensure_output_root(path: Path) -> Path:
    """Create and return one normalized output root."""

    path.mkdir(parents=True, exist_ok=True)
    return path


_VISDOC_VIDORE_V1_DATASETS = {
    "ViDoRe_arxivqa",
    "ViDoRe_docvqa",
    "ViDoRe_infovqa",
    "ViDoRe_tabfquad",
    "ViDoRe_tatdqa",
    "ViDoRe_shiftproject",
    "ViDoRe_syntheticDocQA_artificial_intelligence",
    "ViDoRe_syntheticDocQA_energy",
    "ViDoRe_syntheticDocQA_government_reports",
    "ViDoRe_syntheticDocQA_healthcare_industry",
}

_VISDOC_VIDORE_V2_DATASETS = {
    "ViDoRe_esg_reports_human_labeled_v2",
    "ViDoRe_biomedical_lectures_v2_multilingual",
    "ViDoRe_economics_reports_v2_multilingual",
    "ViDoRe_esg_reports_v2_multilingual",
}

_VISDOC_VISRAG_DATASETS = {
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
    "VisRAG_InfoVQA",
    "VisRAG_PlotQA",
}

_VISDOC_OOD_DATASETS = {
    "ViDoSeek-page",
    "ViDoSeek-doc",
    "MMLongBench-page",
    "MMLongBench-doc",
}


def get_metadata_task_type(dataset_name: str, pipeline: str) -> str:
    """Resolve the minimal metadata task type from one dataset pipeline."""

    if pipeline == "image_i2i_vg":
        if dataset_name in _IMAGE_I2I_VG_RETRIEVAL_DATASETS:
            return "image_retrieval"
        if dataset_name in _IMAGE_I2I_VG_GROUNDING_DATASETS:
            return "image_visual_grounding"
        raise ValueError(
            f"Unsupported MMEB-V2 image_i2i_vg dataset for metadata: {dataset_name!r}"
        )

    if pipeline == "visdoc_beir":
        if dataset_name in _VISDOC_VIDORE_V1_DATASETS:
            return "visdoc_vidore_v1"
        if dataset_name in _VISDOC_VIDORE_V2_DATASETS:
            return "visdoc_vidore_v2"
        if dataset_name in _VISDOC_VISRAG_DATASETS:
            return "visdoc_visrag"
        if dataset_name in _VISDOC_OOD_DATASETS:
            return "visdoc_ood"
        raise ValueError(f"Unsupported MMEB-V2 visdoc dataset for metadata: {dataset_name!r}")

    try:
        return _TASK_TYPE_BY_PIPELINE[pipeline]
    except KeyError as exc:
        raise ValueError(f"Unsupported MMEB-V2 pipeline for metadata: {pipeline!r}") from exc


def write_minimal_metadata(
    output_path: Path,
    *,
    dataset_name: str,
    task_type: str,
    modality: str,
) -> None:
    """Write one minimal metadata.json for one dataset artifact."""

    metadata = {
        "name": dataset_name,
        "task_type": task_type,
        "modality": modality,
        "benchmark": "MMEB-V2",
    }
    with (output_path / "metadata.json").open("w") as file_handle:
        json.dump(metadata, file_handle, indent=2, ensure_ascii=False)
