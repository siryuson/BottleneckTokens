"""Static MMEB-V2 dataset-to-parser dispatch.

This map is intentionally thin: it only answers "which parser module owns this
subset?" It does not carry benchmark defaults, prompt wording, or text-format
knobs. Those stay in the dataset-local parser modules so the runtime boundary
remains:

``benchmark -> loader -> parser module -> query/candidate transform``.
"""

from __future__ import annotations

from typing import Any

from vlm2emb.data.datasets.mmeb_v2 import parsers as parser_modules

PARSER_BY_DATASET = {
    "HatefulMemes": parser_modules.image_cls,
    "VOC2007": parser_modules.image_cls,
    "SUN397": parser_modules.image_cls,
    "Place365": parser_modules.image_cls,
    "ImageNet-A": parser_modules.image_cls,
    "ImageNet-R": parser_modules.image_cls,
    "ObjectNet": parser_modules.image_cls,
    "Country211": parser_modules.image_cls,
    "N24News": parser_modules.image_cls,
    "ImageNet-1K": parser_modules.image_cls,
    "VizWiz": parser_modules.image_qa,
    "ChartQA": parser_modules.image_qa,
    "ScienceQA": parser_modules.image_qa,
    "GQA": parser_modules.image_qa,
    "A-OKVQA": parser_modules.image_qa,
    "DocVQA": parser_modules.image_qa,
    "InfographicsVQA": parser_modules.image_qa,
    "Visual7W": parser_modules.image_qa,
    "TextVQA": parser_modules.image_qa,
    "OK-VQA": parser_modules.image_qa,
    "MSCOCO_i2t": parser_modules.image_i2t,
    "VisualNews_i2t": parser_modules.image_i2t,
    "VisualNews_t2i": parser_modules.image_t2i,
    "Wiki-SS-NQ": parser_modules.image_t2i,
    "EDIS": parser_modules.image_t2i,
    "WebQA": parser_modules.image_t2i,
    "MSCOCO_t2i": parser_modules.image_t2i,
    "VisDial": parser_modules.image_t2i,
    "NIGHTS": parser_modules.image_i2i_vg,
    "OVEN": parser_modules.image_i2i_vg,
    "FashionIQ": parser_modules.image_i2i_vg,
    "CIRR": parser_modules.image_i2i_vg,
    "Visual7W-Pointing": parser_modules.image_i2i_vg,
    "MSCOCO": parser_modules.image_i2i_vg,
    "RefCOCO": parser_modules.image_i2i_vg,
    "RefCOCO-Matching": parser_modules.image_i2i_vg,
    "K700": parser_modules.video_classification,
    "UCF101": parser_modules.video_classification,
    "HMDB51": parser_modules.video_classification,
    "Breakfast": parser_modules.video_classification,
    "SmthSmthV2": parser_modules.ssv2,
    "DiDeMo": parser_modules.didemo,
    "MSVD": parser_modules.msvd,
    "MSR-VTT": parser_modules.msrvtt,
    "YouCook2": parser_modules.youcook2,
    "VATEX": parser_modules.vatex,
    "QVHighlight": parser_modules.moment_retrieval,
    "Charades-STA": parser_modules.moment_retrieval,
    "MomentSeeker": parser_modules.momentseeker,
    "Video-MME": parser_modules.videomme,
    "NExTQA": parser_modules.nextqa,
    "EgoSchema": parser_modules.egoschema,
    "MVBench": parser_modules.mvbench,
    "ActivityNetQA": parser_modules.activitynetqa,
    "ViDoRe_syntheticDocQA_artificial_intelligence": parser_modules.vidore,
    "ViDoRe_syntheticDocQA_government_reports": parser_modules.vidore,
    "ViDoRe_syntheticDocQA_healthcare_industry": parser_modules.vidore,
    "ViDoRe_shiftproject": parser_modules.vidore,
    "ViDoRe_syntheticDocQA_energy": parser_modules.vidore,
    "ViDoRe_infovqa": parser_modules.vidore,
    "ViDoRe_tabfquad": parser_modules.vidore,
    "ViDoRe_tatdqa": parser_modules.vidore,
    "ViDoRe_arxivqa": parser_modules.vidore,
    "ViDoRe_docvqa": parser_modules.vidore,
    "ViDoRe_esg_reports_human_labeled_v2": parser_modules.vidore,
    "ViDoRe_biomedical_lectures_v2_multilingual": parser_modules.vidore,
    "ViDoRe_economics_reports_v2_multilingual": parser_modules.vidore,
    "ViDoRe_esg_reports_v2_multilingual": parser_modules.vidore,
    "MMLongBench-page": parser_modules.vidore,
    "MMLongBench-doc": parser_modules.vidore,
    "ViDoSeek-doc": parser_modules.vidore,
    "ViDoSeek-page": parser_modules.vidore,
    "VisRAG_ArxivQA": parser_modules.visrag,
    "VisRAG_ChartQA": parser_modules.visrag,
    "VisRAG_MP-DocVQA": parser_modules.visrag,
    "VisRAG_SlideVQA": parser_modules.visrag,
    "VisRAG_InfoVQA": parser_modules.visrag,
    "VisRAG_PlotQA": parser_modules.visrag,
}


def get_parser_module(dataset_name: str) -> Any:
    try:
        return PARSER_BY_DATASET[dataset_name]
    except KeyError as exc:
        raise KeyError(f"Unknown MMEB-V2 runtime dataset: {dataset_name}") from exc
