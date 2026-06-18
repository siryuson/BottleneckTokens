"""Dataset implementations for the BToks public runtime.

This module imports the dataset classes that must register themselves with
``AutoDataset``. Prefer building datasets through ``vlm2emb.auto.AutoDataset``;
this package is not a complete public API surface for every historical helper.

Architecture: Lance-based PyTorch Datasets
============================================
Training datasets use Lance format with PyTorch Dataset interface:

    PyTorch Dataset (Lance) -> __getitem__() returns dict with PIL.Image
        -> TrainingCollator (simplified, no image loading) -> ProcessorWrapper -> Model

Key Design:
    - Images are decoded in Dataset.__getitem__() and returned as PIL.Image
    - Collator no longer handles image loading
    - Three methods: __len__(), __getitem__(idx), __getitems__(indices), __iter__()
    - Conversion owns storage correctness (source/media/qrels/id)
    - Runtime dataset owns adjustable transform logic (prompt/token/newline)
    - Runtime transforms can be injected without regenerating Lance storage

Training dataset classes (Lance-based):
    - MmebTrainDataset (mmeb_train): MMEB training subsets on the new runtime path
    - VidoreTrainDataset (vidore_train): Document retrieval training
    - VidoreRagTrainDataset (vidore_rag_train): Archive-aligned ViDoRe RAG view
    - VisragTrainDataset (visrag_train): VisRAG document retrieval
    - LlavaHoundTrainDataset (llavahound_train): Video training
    - Ssv2TrainDataset (ssv2_train): SmthSmthV2 video classification training
    - Hmdb51TrainDataset (hmdb51_train): HMDB51 video classification training
    - Ucf101TrainDataset (ucf101_train): UCF101 video classification training
    - Kinetics700TrainDataset (kinetics700_train): Kinetics-700 video classification training
    - BreakfastTrainDataset (breakfast_train): Breakfast video classification training
    - MsrvttTrainDataset (msrvtt_train): MSR-VTT text-to-video retrieval training
    - MsvdTrainDataset (msvd_train): MSVD text-to-video retrieval training
    - DidemoTrainDataset (didemo_train): DiDeMo text-to-video retrieval training
    - CharadesStaTrainDataset (charades_sta_train): Charades-STA video moment retrieval training
    - QvhighlightTrainDataset (qvhighlight_train): QVHighlights video moment retrieval training
    - Place365TrainDataset (place365_train): Place365 image classification training
    - Country211TrainDataset (country211_train): Country211 image classification training
    - ScienceqaTrainDataset (scienceqa_train): ScienceQA image-question answering training
    - GqaTrainDataset (gqa_train): GQA image-question answering training
    - TextvqaTrainDataset (textvqa_train): TextVQA image-question answering training
    - VizwizTrainDataset (vizwiz_train): VizWiz image-question answering training
    - MbeirTrainDataset (mbeir_train): M-BEIR retrieval training tasks
    - WikissnqTrainDataset (wikissnq_train): Wiki-SS-NQ text-to-image training
    - MMLongBenchDocTrainDataset (mmlongbench_doc_train): MMLongBench-Doc text-to-page training
    - NextqaTrainDataset (nextqa_train): NExTQA video-question answering training
    - Youcook2TrainDataset (youcook2_train): YouCook2 recipe action video retrieval
    - ActivitynetCaptionsTrainDataset (activitynet_captions_train): ActivityNet Captions segment retrieval
    - VatexTrainDataset (vatex_train): VATEX bilingual video retrieval
    - VideoChat2ItTrainDataset (videochat2_it_train): VideoChat2-IT split video instruction training
    - RefcocoTrainDataset (refcoco_train): RefCOCO visual grounding
    - Visual7wPointingTrainDataset (visual7w_pointing_train): Visual7W Pointing visual grounding

Evaluation dataset classes:
    - RetrievalEvalDataset: Generic retrieval eval container.
    - EvalLanceDataset / VideoEvalLanceDataset: Lance-backed eval row readers.

Combined datasets:
    - CombinedDataset: Combines multiple datasets with metadata for BatchInterleaveSampler
    - load_combined_dataset: Load CombinedDataset from config

Usage example:
============================================
Training with Lance datasets:
>>> from vlm2emb.auto import AutoDataset
>>> from vlm2emb.data.collators import TrainingCollator
>>>
>>> # Load Lance dataset via registry
>>> dataset = AutoDataset.build(
...     type="vidore_train",
...     path="/path/to/vidore",
... )
>>> sample = dataset[0]  # Returns dict with PIL.Image
>>> collator = TrainingCollator(wrapper=wrapper)
>>> batch = collator([sample])

Evaluation:
>>> from vlm2emb.data.datasets import EvalLanceDataset, LanceDataset, RetrievalEvalDataset
>>> dataset = RetrievalEvalDataset(
...     queries=EvalLanceDataset("/path/to/queries.lance"),
...     candidates=EvalLanceDataset("/path/to/candidates.lance"),
...     qrels=LanceDataset("/path/to/qrels.lance"),
... )
"""

# Base classes
from .base import (
    BaseEvalLanceDataset,
    BaseLanceDataset,
    BaseStructuredSampleLanceDataset,
    EvalLanceDataset,
    LanceDataset,
    LanceKeyedRowResolver,
    LanceTableExtension,
    ResolvedRetrievalEvalSpec,
    ResolvedTrainDatasetSpec,
    SampleLanceDataset,
    SampleTransform,
    VideoEvalLanceDataset,
)

# Evaluation datasets
from .activitynet_captions_train import ActivitynetCaptionsTrainDataset
from .breakfast_train import BreakfastTrainDataset
from .charades_sta_train import CharadesStaTrainDataset

# Combine datasets
from .combine import (
    CombinedDataset,
    load_combined_dataset,
)
from .const import (
    LEGACY_IMAGE_TOKENS,
    LEGACY_VIDEO_TOKENS,
    PHI3V_IMAGE_TOKEN,
    STANDARD_IMAGE_TOKEN,
    STANDARD_VIDEO_TOKEN,
    canonicalize_multimodal_text,
    decode_image,
    extract_image_bytes,
    normalize_text_whitespace,
)
from .country211_train import Country211TrainDataset
from .didemo_train import DidemoTrainDataset
from .eval import RetrievalEvalDataset
from .gqa_train import GqaTrainDataset
from .hmdb51_train import Hmdb51TrainDataset
from .kinetics700_train import Kinetics700TrainDataset
from .llavahound import LlavaHoundTrainDataset
from .mbeir_train import MbeirTrainDataset
from .mmlongbench_doc_train import MMLongBenchDocTrainDataset
from .nextqa_train import NextqaTrainDataset

# Lance-based training datasets
from .mmeb_train import MmebTrainDataset
from .msrvtt_train import MsrvttTrainDataset
from .msvd_train import MsvdTrainDataset
from .place365_train import Place365TrainDataset
from .qvhighlight_train import QvhighlightTrainDataset
from .refcoco_train import RefcocoTrainDataset
from .scienceqa_train import ScienceqaTrainDataset
from .ssv2_train import Ssv2TrainDataset
from .textvqa_train import TextvqaTrainDataset
from .ucf101_train import Ucf101TrainDataset
from .vatex_train import VatexTrainDataset
from .videochat2_it_train import VideoChat2ItTrainDataset
from .vidore import VidoreTrainDataset
from .vidore_rag import VidoreRagTrainDataset
from .vizwiz_train import VizwizTrainDataset
from .visrag import VisragTrainDataset
from .visual7w_pointing import LanceVisual7WPointingTrainDataset, Visual7wPointingTrainDataset
from .wikissnq_train import WikissnqTrainDataset
from .youcook2_train import Youcook2TrainDataset

__all__ = [
    "STANDARD_IMAGE_TOKEN",
    "STANDARD_VIDEO_TOKEN",
    "PHI3V_IMAGE_TOKEN",
    "LEGACY_IMAGE_TOKENS",
    "LEGACY_VIDEO_TOKENS",
    "canonicalize_multimodal_text",
    "normalize_text_whitespace",
    "decode_image",
    "extract_image_bytes",
    # Base classes
    "BaseLanceDataset",
    "BaseStructuredSampleLanceDataset",
    "BaseEvalLanceDataset",
    "LanceDataset",
    "LanceTableExtension",
    "SampleLanceDataset",
    "EvalLanceDataset",
    "LanceKeyedRowResolver",
    "ResolvedTrainDatasetSpec",
    "ResolvedRetrievalEvalSpec",
    "SampleTransform",
    "VideoEvalLanceDataset",
    # Lance-based training datasets
    "MmebTrainDataset",
    "VidoreTrainDataset",
    "VidoreRagTrainDataset",
    "VisragTrainDataset",
    "LlavaHoundTrainDataset",
    "Ssv2TrainDataset",
    "Hmdb51TrainDataset",
    "Ucf101TrainDataset",
    "Kinetics700TrainDataset",
    "BreakfastTrainDataset",
    "DidemoTrainDataset",
    "CharadesStaTrainDataset",
    "QvhighlightTrainDataset",
    "Place365TrainDataset",
    "Country211TrainDataset",
    "ScienceqaTrainDataset",
    "GqaTrainDataset",
    "TextvqaTrainDataset",
    "VizwizTrainDataset",
    "MbeirTrainDataset",
    "WikissnqTrainDataset",
    "MMLongBenchDocTrainDataset",
    "NextqaTrainDataset",
    "Youcook2TrainDataset",
    "ActivitynetCaptionsTrainDataset",
    "VatexTrainDataset",
    "VideoChat2ItTrainDataset",
    "RefcocoTrainDataset",
    "MsvdTrainDataset",
    "MsrvttTrainDataset",
    "Visual7wPointingTrainDataset",
    "LanceVisual7WPointingTrainDataset",
    # Evaluation datasets
    "RetrievalEvalDataset",
    # Combine
    "CombinedDataset",
    "load_combined_dataset",
]
