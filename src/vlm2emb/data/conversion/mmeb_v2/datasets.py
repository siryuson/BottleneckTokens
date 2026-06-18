"""Minimal MMEB-V2 dataset list used by conversion dispatch.

This module is the single manual dataset list for 12-01 conversion. Each spec
only records the facts needed to route one dataset into the correct conversion
pipeline and locate its required sources.
"""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_MODALITY = "image"
VIDEO_MODALITY = "video"
VISDOC_MODALITY = "visdoc"


@dataclass(frozen=True)
class SourceSpec:
    """One explicit source consumed by one pipeline."""

    role: str
    kind: str
    root: str
    ref: str

    @property
    def key(self) -> str:
        return f"{self.root}:{self.ref}"


@dataclass(frozen=True)
class DatasetSpec:
    """Minimal dataset entry used for conversion dispatch."""

    name: str
    modality: str
    pipeline: str
    sources: tuple[SourceSpec, ...] = ()


def _source(role: str, kind: str, root: str, ref: str) -> SourceSpec:
    """Construct one immutable source entry for the manual dataset list."""
    return SourceSpec(role=role, kind=kind, root=root, ref=ref)


IMAGE_MEDIA_ROOT = _source("media_root", "local_dir", "mmeb_v2", "image-tasks/MMEB")

_ALL_SPECS: tuple[DatasetSpec, ...] = (
    *(
        DatasetSpec(
            name=name,
            modality=IMAGE_MODALITY,
            pipeline="image_cls",
            sources=(
                _source("annotation", "local_dir", "mmeb_eval", name),
                IMAGE_MEDIA_ROOT,
            ),
        )
        for name in (
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
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=IMAGE_MODALITY,
            pipeline="image_qa",
            sources=(
                _source("annotation", "local_dir", "mmeb_eval", name),
                IMAGE_MEDIA_ROOT,
            ),
        )
        for name in (
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
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=IMAGE_MODALITY,
            pipeline="image_i2t",
            sources=(
                _source("annotation", "local_dir", "mmeb_eval", name),
                IMAGE_MEDIA_ROOT,
            ),
        )
        for name in (
            "MSCOCO_i2t",
            "VisualNews_i2t",
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=IMAGE_MODALITY,
            pipeline="image_t2i",
            sources=(
                _source("annotation", "local_dir", "mmeb_eval", name),
                IMAGE_MEDIA_ROOT,
            ),
        )
        for name in (
            "VisDial",
            "MSCOCO_t2i",
            "VisualNews_t2i",
            "WebQA",
            "EDIS",
            "Wiki-SS-NQ",
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=IMAGE_MODALITY,
            pipeline="image_i2i_vg",
            sources=(
                _source("annotation", "local_dir", "mmeb_eval", name),
                IMAGE_MEDIA_ROOT,
            ),
        )
        for name in (
            "CIRR",
            "NIGHTS",
            "OVEN",
            "FashionIQ",
            "MSCOCO",
            "RefCOCO",
            "RefCOCO-Matching",
            "Visual7W-Pointing",
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=VIDEO_MODALITY,
            pipeline="video_cls",
            sources=(
                _source("frames", "local_dir", "mmeb_v2", frames_ref),
                _source("annotation", "local_file", "mmeb_v2", annotation_ref),
            ),
        )
        for name, frames_ref, annotation_ref in (
            ("K700", "video-tasks/frames/K700", "video-tasks/data/k700.jsonl"),
            ("UCF101", "video-tasks/frames/UCF101", "video-tasks/data/ucf101.jsonl"),
            ("HMDB51", "video-tasks/frames/HMDB51", "video-tasks/data/hmdb51.jsonl"),
            ("Breakfast", "video-tasks/frames/Breakfast", "video-tasks/data/breakfast.jsonl"),
            ("SmthSmthV2", "video-tasks/frames/SSv2", "video-tasks/data/ssv2.jsonl"),
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=VIDEO_MODALITY,
            pipeline="video_qa",
            sources=(
                _source("frames", "local_dir", "mmeb_v2", frames_ref),
                _source("annotation", annotation_kind, annotation_root, annotation_ref),
            ),
        )
        for name, frames_ref, annotation_kind, annotation_root, annotation_ref in (
            ("Video-MME", "video-tasks/frames/video_qa/Video-MME", "hf_dataset", "hf", "siyrus/BToks-Video-MME::test"),
            ("MVBench", "video-tasks/frames/video_qa/MVBench", "hf_dataset", "hf", "siyrus/BToks-MVBench::train"),
            ("NExTQA", "video-tasks/frames/video_qa/NExTQA", "hf_dataset", "hf", "siyrus/BToks-NExTQA::MC/test"),
            ("EgoSchema", "video-tasks/frames/video_qa/egoschema", "hf_dataset", "hf", "siyrus/BToks-EgoSchema::Subset/test"),
            ("ActivityNetQA", "video-tasks/frames/video_qa/ActivityNetQA", "local_file", "mmeb_v2", "video-tasks/data/activitynetqa.jsonl"),
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=VIDEO_MODALITY,
            pipeline="video_text_retrieval",
            sources=(
                _source("frames", "local_dir", "mmeb_v2", f"video-tasks/frames/data/ziyan/video_retrieval/{name}/frames"),
                _source("annotation", "hf_dataset", "hf", annotation_ref),
            ),
        )
        for name, annotation_ref in (
            ("MSR-VTT", "siyrus/BToks-MSR-VTT::test_1k/test"),
            ("MSVD", "siyrus/BToks-MSVD::test"),
            ("DiDeMo", "siyrus/BToks-DiDeMo::test"),
            ("VATEX", "siyrus/BToks-VATEX::test"),
            ("YouCook2", "lmms-lab/YouCook2::val"),
        )
    ),
    *(
        DatasetSpec(
            name=name,
            modality=VIDEO_MODALITY,
            pipeline="video_moment_retrieval",
            sources=(
                _source("frames", "local_dir", "mmeb_v2", f"video-tasks/frames/video_mret/{name}"),
                _source("annotation", "local_file", "mmeb_v2", f"video-tasks/data/{annotation_file}"),
            ),
        )
        for name, annotation_file in (
            ("QVHighlight", "qvhighlight.jsonl"),
            ("Charades-STA", "charades_sta.jsonl"),
        )
    ),
    DatasetSpec(
        name="MomentSeeker",
        modality=VIDEO_MODALITY,
        pipeline="video_momentseeker",
        sources=(
            _source("frames", "local_dir", "mmeb_v2", "video-tasks/frames/video_mret/MomentSeeker"),
            _source("annotation", "hf_dataset", "hf", "siyrus/BToks-MomentSeeker::test"),
        ),
    ),
    *(
        DatasetSpec(
            name=name,
            modality=VISDOC_MODALITY,
            pipeline="visdoc_beir",
            sources=(
                _source("annotation", kind, root, ref),
            ),
        )
        for name, kind, root, ref in (
            ("ViDoRe_arxivqa", "local_dir", "mmeb_v2", "visdoc-tasks/data/arxivqa_test_subsampled_beir"),
            ("ViDoRe_docvqa", "local_dir", "mmeb_v2", "visdoc-tasks/data/docvqa_test_subsampled_beir"),
            ("ViDoRe_infovqa", "local_dir", "mmeb_v2", "visdoc-tasks/data/infovqa_test_subsampled_beir"),
            ("ViDoRe_tabfquad", "local_dir", "mmeb_v2", "visdoc-tasks/data/tabfquad_test_subsampled_beir"),
            ("ViDoRe_tatdqa", "local_dir", "mmeb_v2", "visdoc-tasks/data/tatdqa_test_beir"),
            ("ViDoRe_shiftproject", "local_dir", "mmeb_v2", "visdoc-tasks/data/shiftproject_test_beir"),
            ("ViDoRe_syntheticDocQA_artificial_intelligence", "local_dir", "mmeb_v2", "visdoc-tasks/data/syntheticDocQA_artificial_intelligence_test_beir"),
            ("ViDoRe_syntheticDocQA_energy", "local_dir", "mmeb_v2", "visdoc-tasks/data/syntheticDocQA_energy_test_beir"),
            ("ViDoRe_syntheticDocQA_government_reports", "local_dir", "mmeb_v2", "visdoc-tasks/data/syntheticDocQA_government_reports_test_beir"),
            ("ViDoRe_syntheticDocQA_healthcare_industry", "local_dir", "mmeb_v2", "visdoc-tasks/data/syntheticDocQA_healthcare_industry_test_beir"),
            ("ViDoRe_esg_reports_human_labeled_v2", "hf_dataset", "hf", "vidore/esg_reports_human_labeled_v2::test"),
            ("ViDoRe_biomedical_lectures_v2_multilingual", "local_dir", "mmeb_v2", "visdoc-tasks/data/biomedical_lectures_v2"),
            ("ViDoRe_economics_reports_v2_multilingual", "local_dir", "mmeb_v2", "visdoc-tasks/data/economics_reports_v2"),
            ("ViDoRe_esg_reports_v2_multilingual", "local_dir", "mmeb_v2", "visdoc-tasks/data/esg_reports_v2"),
            ("VisRAG_ArxivQA", "local_dir", "mmeb_v2", "visdoc-tasks/data/VisRAG-Ret-Test-ArxivQA"),
            ("VisRAG_ChartQA", "local_dir", "mmeb_v2", "visdoc-tasks/data/VisRAG-Ret-Test-ChartQA"),
            ("VisRAG_MP-DocVQA", "local_dir", "mmeb_v2", "visdoc-tasks/data/VisRAG-Ret-Test-MP-DocVQA"),
            ("VisRAG_SlideVQA", "local_dir", "mmeb_v2", "visdoc-tasks/data/VisRAG-Ret-Test-SlideVQA"),
            ("VisRAG_InfoVQA", "local_dir", "mmeb_v2", "visdoc-tasks/data/VisRAG-Ret-Test-InfoVQA"),
            ("VisRAG_PlotQA", "local_dir", "mmeb_v2", "visdoc-tasks/data/VisRAG-Ret-Test-PlotQA"),
            ("ViDoSeek-page", "hf_dataset", "hf", "siyrus/BToks-ViDoSeek-page-fixed::test"),
            ("ViDoSeek-doc", "local_dir", "mmeb_v2", "visdoc-tasks/data/ViDoSeek"),
            ("MMLongBench-page", "hf_dataset", "hf", "siyrus/BToks-MMLongBench-page-fixed::test"),
            ("MMLongBench-doc", "local_dir", "mmeb_v2", "visdoc-tasks/data/MMLongBench-doc"),
        )
    ),
)

_SPEC_BY_NAME = {spec.name: spec for spec in _ALL_SPECS}


def get_all_specs() -> tuple[DatasetSpec, ...]:
    """Return the full MMEB-V2 dataset spec list in declaration order."""
    return _ALL_SPECS


def get_spec(dataset_name: str) -> DatasetSpec:
    """Return one dataset spec or raise when the name is unknown."""
    try:
        return _SPEC_BY_NAME[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown MMEB-V2 dataset: {dataset_name}") from exc


def get_dataset_names() -> tuple[str, ...]:
    """Return all registered MMEB-V2 dataset names."""
    return tuple(spec.name for spec in _ALL_SPECS)


def get_modality(dataset_name: str) -> str:
    """Return the modality declared for one dataset."""
    return get_spec(dataset_name).modality


def get_pipeline(dataset_name: str) -> str:
    """Return the pipeline key declared for one dataset."""
    return get_spec(dataset_name).pipeline


def get_specs_by_modality(modality: str) -> tuple[DatasetSpec, ...]:
    """Return all dataset specs that match the given modality."""
    return tuple(spec for spec in _ALL_SPECS if spec.modality == modality)


def get_specs_by_pipeline(pipeline: str, *, modality: str | None = None) -> tuple[DatasetSpec, ...]:
    """Return all dataset specs routed through one pipeline."""
    return tuple(
        spec
        for spec in _ALL_SPECS
        if spec.pipeline == pipeline and (modality is None or spec.modality == modality)
    )
