from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import TrainingArguments

from vlm2emb.auto import AutoTrainer, AutoTrainingArgs
from vlm2emb.config import load_config
from vlm2emb.data.collators.training_collator import (
    TrainingCollator,
    render_instruction_text,
)
from vlm2emb.training.trainers.gme_trainer import GMEStyleTrainingArgs, GMEStyleTrainer
from vlm2emb.training.trainers.vlm2vec_trainer import VLM2VecTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]


class _RecordingWrapper:
    def __init__(self) -> None:
        self.payload_batches: list[list[dict]] = []

    def process_multimodal_batch(self, payloads: list[dict]) -> list[dict]:
        self.payload_batches.append(payloads)
        return [
            {
                "input_ids": torch.tensor([[index + 1]], dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "text": payload["text"],
            }
            for index, payload in enumerate(payloads)
        ]


def test_gme_training_args_registers_dense_vlm2vec_alias() -> None:
    args = AutoTrainingArgs.from_config(
        {
            "type": "gme",
            "output_dir": "/tmp/gme-style",
            "report_to": "none",
        }
    )

    assert isinstance(args, GMEStyleTrainingArgs)
    assert args.temperature == pytest.approx(0.03)
    assert args.use_explicit_negatives is False
    assert args.hard_negative_mining is False
    assert issubclass(GMEStyleTrainer, VLM2VecTrainer)


def test_gme_training_args_accepts_string_false_overrides() -> None:
    args = AutoTrainingArgs.from_config(
        {
            "type": "gme",
            "output_dir": "/tmp/gme-style",
            "report_to": "none",
            "use_explicit_negatives": "false",
            "hard_negative_mining": "false",
        }
    )

    assert args.use_explicit_negatives is False
    assert args.hard_negative_mining is False


def test_gme_training_args_rejects_mining_for_fair_baseline() -> None:
    with pytest.raises(ValueError, match="hard_negative_mining"):
        AutoTrainingArgs.from_config(
            {
                "type": "gme",
                "output_dir": "/tmp/gme-style",
                "report_to": "none",
                "hard_negative_mining": True,
            }
        )


def test_gme_trainer_alias_instantiates_from_registry(tmp_path: Path) -> None:
    trainer = AutoTrainer.from_config(
        {"type": "gme_trainer"},
        model=torch.nn.Linear(1, 1),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            report_to="none",
        ),
    )

    assert isinstance(trainer, GMEStyleTrainer)
    assert isinstance(trainer, VLM2VecTrainer)


def test_gme_presets_resolve_without_residual_inherit_keys() -> None:
    gme = load_config(REPO_ROOT / "configs/presets/gme_qwen2vl_2b.yaml")
    btoks_gme = load_config(REPO_ROOT / "configs/presets/btoks_gme_qwen2vl_2b_v1.yaml")

    assert gme.train.trainer.type == "gme_trainer"
    assert gme.train.args.type == "gme"
    assert gme.train.args.temperature == pytest.approx(0.03)
    assert gme.train.collator.hard_negative_mining is False
    assert gme.model.modules.pooling.type == "LastTokenPooling"

    assert btoks_gme.train.trainer.type == "btoks_trainer"
    assert btoks_gme.train.args.temperature == pytest.approx(0.03)
    assert btoks_gme.train.args.generation_loss_weight == pytest.approx(0.1)
    assert btoks_gme.train.collator.hard_negative_mining is False
    assert btoks_gme.model.modules.injector.type == "BToksTokenInjector"
    assert btoks_gme.model.modules.pooling.type == "BToksPooling"


def test_render_instruction_text_supports_query_and_corpus_roles() -> None:
    policy = {
        "query_instruction": "Represent the query for retrieval.",
        "corpus_instruction": "",
        "template": "{instruction}\n{text}",
    }

    assert render_instruction_text(
        "find this",
        field_name="query",
        instruction_policy=policy,
    ) == "Represent the query for retrieval.\nfind this"
    assert render_instruction_text(
        "candidate",
        field_name="positive",
        instruction_policy=policy,
    ) == "candidate"


def test_training_collator_applies_gme_query_instruction_without_visual_duplication() -> None:
    wrapper = _RecordingWrapper()
    collator = TrainingCollator(
        wrapper=wrapper,
        instruction_policy={
            "query_instruction": "Represent the query for retrieval.",
            "corpus_instruction": "",
            "template": "{instruction}\n{text}",
        },
    )
    image = object()

    outputs = collator(
        [
            {
                "query": {
                    "text": "<|image_pad|>\nFind matching caption\n",
                    "media": [{"kind": "image", "content": image}],
                },
                "positive": {
                    "text": "A matching caption\n",
                    "media": [],
                },
                "negative": {"text": "", "media": []},
            }
        ]
    )

    query_payload = wrapper.payload_batches[0][0]
    positive_payload = wrapper.payload_batches[1][0]
    assert query_payload["text"].startswith("Represent the query for retrieval.\n")
    assert query_payload["text"].count("<|image_pad|>") == 1
    assert query_payload["media"] == [{"kind": "image", "content": image}]
    assert positive_payload["text"] == "A matching caption\n"
    assert outputs["negative"] == [None]


def test_training_collator_applies_instruction_to_pure_text_query_only() -> None:
    wrapper = _RecordingWrapper()
    collator = TrainingCollator(
        wrapper=wrapper,
        instruction_policy={
            "query_instruction": "Represent the query for retrieval.",
            "corpus_instruction": "",
            "template": "{instruction}\n{text}",
        },
    )

    collator(
        [
            {
                "query": {"text": "find this text", "media": []},
                "positive": {"text": "candidate text", "media": []},
                "negative": {"text": "", "media": []},
            }
        ]
    )

    assert wrapper.payload_batches[0][0]["text"] == (
        "Represent the query for retrieval.\nfind this text"
    )
    assert "<|image_pad|>" not in wrapper.payload_batches[0][0]["text"]
    assert wrapper.payload_batches[1][0]["text"] == "candidate text"


def test_training_collator_supports_corpus_instruction_when_configured() -> None:
    wrapper = _RecordingWrapper()
    collator = TrainingCollator(
        wrapper=wrapper,
        instruction_policy={
            "query_instruction": "Query instruction.",
            "corpus_instruction": "Corpus instruction.",
            "template": "{instruction}\n{text}",
        },
    )

    collator(
        [
            {
                "query": {"text": "query", "media": []},
                "positive": {"text": "positive", "media": []},
                "negative": {"text": "negative", "media": []},
            }
        ]
    )

    assert wrapper.payload_batches[0][0]["text"] == "Query instruction.\nquery"
    assert wrapper.payload_batches[1][0]["text"] == "Corpus instruction.\npositive"
    assert wrapper.payload_batches[2][0]["text"] == "Corpus instruction.\nnegative"


def test_training_collator_accepts_string_false_for_reserved_flags() -> None:
    collator = TrainingCollator.from_config(
        {
            "type": "training",
            "wrapper": _RecordingWrapper(),
            "use_explicit_negatives": "false",
            "hard_negative_mining": "false",
        }
    )

    assert collator.use_explicit_negatives is False
    assert collator.hard_negative_mining is False


def test_training_collator_rejects_mining_flag() -> None:
    with pytest.raises(ValueError, match="hard_negative_mining"):
        TrainingCollator(
            wrapper=_RecordingWrapper(),
            hard_negative_mining=True,
        )
