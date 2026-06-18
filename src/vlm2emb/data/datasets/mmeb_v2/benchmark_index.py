"""Benchmark-only MMEB-V2 subset index.

The row order is intentional. ``MMEBBenchmark`` iterates subsets in the exact
order declared here when no explicit allowlist is provided. Keep this list
aligned with the historical canonical MMEB full-eval order so new runs remain
directly comparable to existing checkpoint baselines and run logs.
"""

from __future__ import annotations


BENCHMARK_INDEX_ROWS = (
    ("VOC2007", "image-tasks/VOC2007", "image", "image_classification"),
    ("N24News", "image-tasks/N24News", "image", "image_classification"),
    ("SUN397", "image-tasks/SUN397", "image", "image_classification"),
    ("ObjectNet", "image-tasks/ObjectNet", "image", "image_classification"),
    ("Country211", "image-tasks/Country211", "image", "image_classification"),
    ("Place365", "image-tasks/Place365", "image", "image_classification"),
    ("ImageNet-1K", "image-tasks/ImageNet-1K", "image", "image_classification"),
    ("HatefulMemes", "image-tasks/HatefulMemes", "image", "image_classification"),
    ("ImageNet-A", "image-tasks/ImageNet-A", "image", "image_classification"),
    ("ImageNet-R", "image-tasks/ImageNet-R", "image", "image_classification"),
    ("OK-VQA", "image-tasks/OK-VQA", "image", "image_question_answer"),
    ("A-OKVQA", "image-tasks/A-OKVQA", "image", "image_question_answer"),
    ("DocVQA", "image-tasks/DocVQA", "image", "image_question_answer"),
    ("InfographicsVQA", "image-tasks/InfographicsVQA", "image", "image_question_answer"),
    ("ChartQA", "image-tasks/ChartQA", "image", "image_question_answer"),
    ("Visual7W", "image-tasks/Visual7W", "image", "image_question_answer"),
    ("ScienceQA", "image-tasks/ScienceQA", "image", "image_question_answer"),
    ("GQA", "image-tasks/GQA", "image", "image_question_answer"),
    ("TextVQA", "image-tasks/TextVQA", "image", "image_question_answer"),
    ("VizWiz", "image-tasks/VizWiz", "image", "image_question_answer"),
    ("VisDial", "image-tasks/VisDial", "image", "image_retrieval"),
    ("CIRR", "image-tasks/CIRR", "image", "image_retrieval"),
    ("VisualNews_t2i", "image-tasks/VisualNews_t2i", "image", "image_retrieval"),
    ("VisualNews_i2t", "image-tasks/VisualNews_i2t", "image", "image_retrieval"),
    ("MSCOCO_t2i", "image-tasks/MSCOCO_t2i", "image", "image_retrieval"),
    ("MSCOCO_i2t", "image-tasks/MSCOCO_i2t", "image", "image_retrieval"),
    ("NIGHTS", "image-tasks/NIGHTS", "image", "image_retrieval"),
    ("WebQA", "image-tasks/WebQA", "image", "image_retrieval"),
    ("FashionIQ", "image-tasks/FashionIQ", "image", "image_retrieval"),
    ("Wiki-SS-NQ", "image-tasks/Wiki-SS-NQ", "image", "image_retrieval"),
    ("OVEN", "image-tasks/OVEN", "image", "image_retrieval"),
    ("EDIS", "image-tasks/EDIS", "image", "image_retrieval"),
    ("MSCOCO", "image-tasks/MSCOCO", "image", "image_visual_grounding"),
    ("RefCOCO", "image-tasks/RefCOCO", "image", "image_visual_grounding"),
    ("RefCOCO-Matching", "image-tasks/RefCOCO-Matching", "image", "image_visual_grounding"),
    ("Visual7W-Pointing", "image-tasks/Visual7W-Pointing", "image", "image_visual_grounding"),
    ("K700", "video-tasks/K700", "video", "video_classification"),
    ("UCF101", "video-tasks/UCF101", "video", "video_classification"),
    ("HMDB51", "video-tasks/HMDB51", "video", "video_classification"),
    ("SmthSmthV2", "video-tasks/SmthSmthV2", "video", "video_classification"),
    ("Breakfast", "video-tasks/Breakfast", "video", "video_classification"),
    ("Video-MME", "video-tasks/Video-MME", "video", "video_question_answer"),
    ("MVBench", "video-tasks/MVBench", "video", "video_question_answer"),
    ("NExTQA", "video-tasks/NExTQA", "video", "video_question_answer"),
    ("EgoSchema", "video-tasks/EgoSchema", "video", "video_question_answer"),
    ("ActivityNetQA", "video-tasks/ActivityNetQA", "video", "video_question_answer"),
    ("MSR-VTT", "video-tasks/MSR-VTT", "video", "video_retrieval"),
    ("MSVD", "video-tasks/MSVD", "video", "video_retrieval"),
    ("DiDeMo", "video-tasks/DiDeMo", "video", "video_retrieval"),
    ("VATEX", "video-tasks/VATEX", "video", "video_retrieval"),
    ("YouCook2", "video-tasks/YouCook2", "video", "video_retrieval"),
    ("QVHighlight", "video-tasks/QVHighlight", "video", "video_moment_retrieval"),
    ("Charades-STA", "video-tasks/Charades-STA", "video", "video_moment_retrieval"),
    ("MomentSeeker", "video-tasks/MomentSeeker", "video", "video_moment_retrieval"),
    ("ViDoRe_arxivqa", "visdoc-tasks/ViDoRe_arxivqa", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_docvqa", "visdoc-tasks/ViDoRe_docvqa", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_infovqa", "visdoc-tasks/ViDoRe_infovqa", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_tabfquad", "visdoc-tasks/ViDoRe_tabfquad", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_tatdqa", "visdoc-tasks/ViDoRe_tatdqa", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_shiftproject", "visdoc-tasks/ViDoRe_shiftproject", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_syntheticDocQA_artificial_intelligence", "visdoc-tasks/ViDoRe_syntheticDocQA_artificial_intelligence", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_syntheticDocQA_energy", "visdoc-tasks/ViDoRe_syntheticDocQA_energy", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_syntheticDocQA_government_reports", "visdoc-tasks/ViDoRe_syntheticDocQA_government_reports", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_syntheticDocQA_healthcare_industry", "visdoc-tasks/ViDoRe_syntheticDocQA_healthcare_industry", "visdoc", "visdoc_vidore_v1"),
    ("ViDoRe_esg_reports_human_labeled_v2", "visdoc-tasks/ViDoRe_esg_reports_human_labeled_v2", "visdoc", "visdoc_vidore_v2"),
    ("ViDoRe_biomedical_lectures_v2_multilingual", "visdoc-tasks/ViDoRe_biomedical_lectures_v2_multilingual", "visdoc", "visdoc_vidore_v2"),
    ("ViDoRe_economics_reports_v2_multilingual", "visdoc-tasks/ViDoRe_economics_reports_v2_multilingual", "visdoc", "visdoc_vidore_v2"),
    ("ViDoRe_esg_reports_v2_multilingual", "visdoc-tasks/ViDoRe_esg_reports_v2_multilingual", "visdoc", "visdoc_vidore_v2"),
    ("VisRAG_ArxivQA", "visdoc-tasks/VisRAG_ArxivQA", "visdoc", "visdoc_visrag"),
    ("VisRAG_ChartQA", "visdoc-tasks/VisRAG_ChartQA", "visdoc", "visdoc_visrag"),
    ("VisRAG_MP-DocVQA", "visdoc-tasks/VisRAG_MP-DocVQA", "visdoc", "visdoc_visrag"),
    ("VisRAG_SlideVQA", "visdoc-tasks/VisRAG_SlideVQA", "visdoc", "visdoc_visrag"),
    ("VisRAG_InfoVQA", "visdoc-tasks/VisRAG_InfoVQA", "visdoc", "visdoc_visrag"),
    ("VisRAG_PlotQA", "visdoc-tasks/VisRAG_PlotQA", "visdoc", "visdoc_visrag"),
    ("ViDoSeek-page", "visdoc-tasks/ViDoSeek-page", "visdoc", "visdoc_ood"),
    ("ViDoSeek-doc", "visdoc-tasks/ViDoSeek-doc", "visdoc", "visdoc_ood"),
    ("MMLongBench-page", "visdoc-tasks/MMLongBench-page", "visdoc", "visdoc_ood"),
    ("MMLongBench-doc", "visdoc-tasks/MMLongBench-doc", "visdoc", "visdoc_ood"),
)

PATH_REL_BY_DATASET = {name: path_rel for name, path_rel, _, _ in BENCHMARK_INDEX_ROWS}
MODALITY_BY_DATASET = {name: modality for name, _, modality, _ in BENCHMARK_INDEX_ROWS}
TASK_TYPE_BY_DATASET = {name: task_type for name, _, _, task_type in BENCHMARK_INDEX_ROWS}

_datasets_by_modality: dict[str, list[str]] = {"image": [], "video": [], "visdoc": []}
for dataset_name, _, modality, _ in BENCHMARK_INDEX_ROWS:
    _datasets_by_modality[modality].append(dataset_name)
DATASETS_BY_MODALITY = {
    modality: tuple(dataset_names)
    for modality, dataset_names in _datasets_by_modality.items()
}


def has_benchmark_dataset(dataset_name: str) -> bool:
    return dataset_name in PATH_REL_BY_DATASET


def get_path_rel(dataset_name: str) -> str:
    return PATH_REL_BY_DATASET[dataset_name]


def get_modality(dataset_name: str) -> str:
    return MODALITY_BY_DATASET[dataset_name]


def get_task_type(dataset_name: str) -> str:
    return TASK_TYPE_BY_DATASET[dataset_name]


__all__ = [
    "BENCHMARK_INDEX_ROWS",
    "DATASETS_BY_MODALITY",
    "PATH_REL_BY_DATASET",
    "MODALITY_BY_DATASET",
    "TASK_TYPE_BY_DATASET",
    "get_modality",
    "get_path_rel",
    "get_task_type",
    "has_benchmark_dataset",
]
