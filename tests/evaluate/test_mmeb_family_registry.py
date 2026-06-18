from vlm2emb.data.datasets.mmeb_v2.benchmark_index import BENCHMARK_INDEX_ROWS
from vlm2emb.data.datasets.mmeb_v2.dispatch import PARSER_BY_DATASET, get_parser_module
from vlm2emb.data.datasets.mmeb_v2.parsers import (
    activitynetqa,
    didemo,
    moment_retrieval,
    momentseeker,
    msrvtt,
    msvd,
    mvbench,
    nextqa,
    ssv2,
    vatex,
    videomme,
    vidore,
    visrag,
    youcook2,
)


def test_family_registry_keeps_unique_78_subset_contract() -> None:
    names = [dataset_name for dataset_name, _, _, _ in BENCHMARK_INDEX_ROWS]

    assert len(BENCHMARK_INDEX_ROWS) == 78
    assert len(names) == len(set(names))
    assert set(names) == set(PARSER_BY_DATASET)
    assert len(PARSER_BY_DATASET) == 78


def test_benchmark_index_preserves_canonical_full_eval_order() -> None:
    names = [dataset_name for dataset_name, _, _, _ in BENCHMARK_INDEX_ROWS]

    assert names == [
        "VOC2007",
        "N24News",
        "SUN397",
        "ObjectNet",
        "Country211",
        "Place365",
        "ImageNet-1K",
        "HatefulMemes",
        "ImageNet-A",
        "ImageNet-R",
        "OK-VQA",
        "A-OKVQA",
        "DocVQA",
        "InfographicsVQA",
        "ChartQA",
        "Visual7W",
        "ScienceQA",
        "GQA",
        "TextVQA",
        "VizWiz",
        "VisDial",
        "CIRR",
        "VisualNews_t2i",
        "VisualNews_i2t",
        "MSCOCO_t2i",
        "MSCOCO_i2t",
        "NIGHTS",
        "WebQA",
        "FashionIQ",
        "Wiki-SS-NQ",
        "OVEN",
        "EDIS",
        "MSCOCO",
        "RefCOCO",
        "RefCOCO-Matching",
        "Visual7W-Pointing",
        "K700",
        "UCF101",
        "HMDB51",
        "SmthSmthV2",
        "Breakfast",
        "Video-MME",
        "MVBench",
        "NExTQA",
        "EgoSchema",
        "ActivityNetQA",
        "MSR-VTT",
        "MSVD",
        "DiDeMo",
        "VATEX",
        "YouCook2",
        "QVHighlight",
        "Charades-STA",
        "MomentSeeker",
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
        "ViDoRe_esg_reports_human_labeled_v2",
        "ViDoRe_biomedical_lectures_v2_multilingual",
        "ViDoRe_economics_reports_v2_multilingual",
        "ViDoRe_esg_reports_v2_multilingual",
        "VisRAG_ArxivQA",
        "VisRAG_ChartQA",
        "VisRAG_MP-DocVQA",
        "VisRAG_SlideVQA",
        "VisRAG_InfoVQA",
        "VisRAG_PlotQA",
        "ViDoSeek-page",
        "ViDoSeek-doc",
        "MMLongBench-page",
        "MMLongBench-doc",
    ]


def test_parser_aligned_family_modules_export_expected_subsets() -> None:
    assert ssv2.DATASET_NAME == "SmthSmthV2"
    assert didemo.DATASET_NAME == "DiDeMo"
    assert msvd.DATASET_NAME == "MSVD"
    assert msrvtt.DATASET_NAME == "MSR-VTT"
    assert youcook2.DATASET_NAME == "YouCook2"
    assert vatex.DATASET_NAME == "VATEX"
    assert tuple(moment_retrieval.DATASET_NAMES) == (
        "QVHighlight",
        "Charades-STA",
    )
    assert momentseeker.DATASET_NAME == "MomentSeeker"
    assert videomme.DATASET_NAME == "Video-MME"
    assert nextqa.DATASET_NAME == "NExTQA"
    assert mvbench.DATASET_NAME == "MVBench"
    assert activitynetqa.DATASET_NAME == "ActivityNetQA"
    assert len(vidore.DATASET_NAMES) == 18
    assert len(visrag.DATASET_NAMES) == 6


def test_parser_resolution_matches_dataset_to_parser_map() -> None:
    assert get_parser_module("MSR-VTT") is msrvtt
    assert get_parser_module("SmthSmthV2") is ssv2
    assert get_parser_module("ViDoRe_shiftproject") is vidore
    assert get_parser_module("VisRAG_ChartQA") is visrag
