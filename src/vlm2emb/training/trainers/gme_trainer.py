"""GME-style trainer aliases for fair dense embedding comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

from vlm2emb.auto import AutoTrainer, AutoTrainingArgs
from vlm2emb.training.trainers.vlm2vec_trainer import VLM2VecTrainer, VLM2VecTrainingArgs


def _parse_bool_config(value: object, *, field_name: str) -> bool:
    """Parse bool config values from YAML or CLI overrides."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean, got {value!r}")


@AutoTrainingArgs.register("gme")
@AutoTrainingArgs.register("gme_style")
@dataclass
class GMEStyleTrainingArgs(VLM2VecTrainingArgs):
    """Training arguments for GME-style dense embedding experiments.

    The first GME-style implementation intentionally reuses VLM2Vec's dense
    in-batch contrastive training path. The default temperature follows the
    public GME recipe, while hard-negative mining remains out of scope.
    """

    temperature: float = field(
        default=0.03,
        metadata={"help": "Temperature for GME-style contrastive loss."},
    )
    use_explicit_negatives: bool = field(
        default=False,
        metadata={
            "help": (
                "Reserved for future GME-style explicit-negative training. "
                "The fair baseline keeps this disabled."
            )
        },
    )
    hard_negative_mining: bool = field(
        default=False,
        metadata={
            "help": (
                "Reserved flag for future mined-negative experiments. "
                "No mining is performed by the GME-style fair baseline."
            )
        },
    )

    def __post_init__(self) -> None:
        """Validate fair-comparison GME-style arguments."""
        self.use_explicit_negatives = _parse_bool_config(
            self.use_explicit_negatives,
            field_name="use_explicit_negatives",
        )
        self.hard_negative_mining = _parse_bool_config(
            self.hard_negative_mining,
            field_name="hard_negative_mining",
        )
        super().__post_init__()
        if self.use_explicit_negatives:
            raise ValueError(
                "use_explicit_negatives is reserved for a future GME-style "
                "negative-training change. Keep it false for fair baseline runs."
            )
        if self.hard_negative_mining:
            raise ValueError(
                "hard_negative_mining is reserved for a future data-mining "
                "change. The fair GME-style baseline does not mine negatives."
            )


@AutoTrainer.register("gme")
@AutoTrainer.register("gme_trainer")
@AutoTrainer.register("gme_style")
class GMEStyleTrainer(VLM2VecTrainer):
    """Thin trainer alias for GME-style dense embedding experiments."""


__all__ = ["GMEStyleTrainer", "GMEStyleTrainingArgs"]
