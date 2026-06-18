from __future__ import annotations

import os

from pathlib import Path

import pytest

from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import BENCHMARK_INDEX_ROWS

MMEB_V2_REAL_ROOT = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2"))

pytestmark = pytest.mark.skipif(
    not MMEB_V2_REAL_ROOT.exists(),
    reason='MMEB-V2 rerun root not available on this machine',
)


def test_mmeb_v2_real_root_all_subsets_smoke_across_modes():
    for mode in ('canonical', 'origin'):
        failures: list[tuple[str, str]] = []
        for dataset_name, path_rel, _, _ in BENCHMARK_INDEX_ROWS:
            dataset_path = MMEB_V2_REAL_ROOT / path_rel
            try:
                dataset = build_mmeb_eval_dataset(
                    dataset_name=dataset_name,
                    artifact_path=str(dataset_path),
                    transform_kwargs={"runtime_mode": mode},
                )
                _ = dataset.queries[0]
                _ = dataset.candidates[0]
            except Exception as exc:  # pragma: no cover - surfaced via assertion payload
                failures.append((dataset_name, f'{type(exc).__name__}: {exc}'))
        assert not failures, failures


def test_mmeb_v2_real_root_visdial_modes_diverge():
    dataset_path = MMEB_V2_REAL_ROOT / 'image-tasks' / 'VisDial'
    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name='VisDial',
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = build_mmeb_eval_dataset(
        dataset_name='VisDial',
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )

    canonical_text = canonical_dataset.queries[0]['query']['text']
    origin_text = origin_dataset.queries[0]['query']['text']

    assert canonical_text.startswith(
        "Represent the given dialogue about an image, which is used for image retrieval:\n"
    )
    assert canonical_text.endswith('\n')
    assert origin_text.endswith('\n')
