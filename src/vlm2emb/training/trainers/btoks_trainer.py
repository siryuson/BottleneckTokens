"""BToks trainer integrated into the HuggingFace Trainer workflow.

This module provides BToks-specific training arguments and trainer behavior,
while keeping compatibility with the shared retrieval training stack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from transformers import DynamicCache, PreTrainedModel, TrainingArguments

from vlm2emb.auto import AutoTrainer, AutoTrainingArgs
from vlm2emb.data.processors import batch_processor_outputs
from vlm2emb.modules.btoks import (
    extract_last_token_kv,
    extract_btoks_kv,
    get_vision_token_ids,
)
from vlm2emb.training.grad_cache import ChunkContext, grad_cache_accumulate
from vlm2emb.training.losses.contrastive import DistributedContrastiveLoss
from vlm2emb.training.trainers.vlm2vec_trainer import VLM2VecTrainer, VLM2VecTrainingArgs

logger = logging.getLogger(__name__)

GENERATION_KV_MODES = {"compressed", "full"}


def _extract_loss(outputs: Any) -> Tensor | None:
    """Extract loss tensor from model outputs (dict or dataclass)."""
    loss = (
        outputs.get("loss")
        if isinstance(outputs, dict)
        else getattr(outputs, "loss", None)
    )
    if loss is None or not torch.is_tensor(loss):
        return None
    return loss


def _compute_generation_start_positions(
    *,
    encode_out: dict[str, Any],
    features: dict[str, Any],
    batch_idx: Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Return per-sample RoPE starts after BToks injection."""
    encode_mask = encode_out.get("attention_mask")
    if encode_mask is None:
        encode_mask = features.get("attention_mask")

    if encode_mask is not None:
        return cast(Tensor, encode_mask).sum(dim=1).to(device=device).index_select(0, batch_idx)

    encode_seq_len = cast(Tensor, features["input_ids"]).shape[1]
    return torch.full(
        (batch_size,),
        encode_seq_len,
        device=device,
        dtype=torch.long,
    )


def _select_encoder_attention_mask(
    *,
    encode_out: dict[str, Any],
    features: dict[str, Any],
    batch_idx: Tensor,
    expected_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    """Return selected encoder attention mask for full-KV generation."""
    encode_mask = encode_out.get("attention_mask")
    if encode_mask is None:
        encode_mask = features.get("attention_mask")
    if encode_mask is None:
        return None

    mask = cast(Tensor, encode_mask).to(device=device, dtype=dtype)
    if mask.ndim != 2:
        raise ValueError("encoder attention_mask must have shape (B, L)")
    if mask.shape[1] != expected_len:
        raise ValueError(
            "encoder attention_mask length must match selected KV cache length"
        )
    return mask.index_select(0, batch_idx)


def _select_batch_from_cache(past_key_values: Any, batch_idx: Tensor) -> DynamicCache:
    """Select a batch subset from a supported KV cache format."""
    selected = DynamicCache()
    if hasattr(past_key_values, "layers") or (
        hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache")
    ):
        iterator = past_key_values
    elif isinstance(past_key_values, (tuple, list)):
        iterator = past_key_values
    else:
        raise TypeError("Unsupported past_key_values format")

    for layer_idx, layer in enumerate(iterator):
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("Invalid past_key_values layer format")
        k, v = layer[0], layer[1]
        selected.update(
            k.index_select(0, batch_idx),
            v.index_select(0, batch_idx),
            layer_idx=layer_idx,
        )
    return selected


def _btoks_forward_fn(
    unwrapped: nn.Module,
    *,
    selected_indices: list[int],
    opposite_batch: dict[str, Any] | None,
    labels: Tensor | None,
    generation_kv_mode: str = "compressed",
    **features,
) -> dict[str, Any]:
    """Encode + optional generation forward. Pure function, no closure capture.

    Args:
        unwrapped: Unwrapped model instance (passed by model.forward as self).
        selected_indices: Indices into the chunk that need generation.
            Empty list means no generation.
        opposite_batch: Pre-sliced opposite samples for generation.
            Already indexed by selected_indices. None when no generation.
        labels: Pre-built labels for generation loss.  # (G, L_opp)
            Already indexed by selected_indices. None when no generation.
        **features: Model input features forwarded from model.forward().

    Returns:
        Dict with "embeddings" (B, D) and optional "gen_loss" scalar.
    """
    need_gen = bool(selected_indices)
    encode_out = cast(dict[str, Any], cast(Any, unwrapped).encode(**features, return_cache=need_gen))
    embeddings = encode_out["embeddings"]  # (B, D)

    gen_loss = None
    kv_cache_mode: str | None = None
    if need_gen and opposite_batch is not None and labels is not None:
        past_key_values = encode_out.get("past_key_values")
        btoks_token_mask = encode_out.get("btoks_token_mask")  # (B, L)

        if past_key_values is not None:
            device = embeddings.device
            batch_size = embeddings.shape[0]
            if generation_kv_mode == "compressed":
                compressed_kv = None
                if btoks_token_mask is not None:
                    compressed_kv = extract_btoks_kv(
                        past_key_values=past_key_values,
                        btoks_token_mask=btoks_token_mask,
                    )
                if compressed_kv is not None:
                    selected_cache = compressed_kv
                    kv_cache_mode = "btoks_token"
                else:
                    encode_mask = encode_out.get("attention_mask")
                    if encode_mask is None:
                        encode_mask = features.get("attention_mask")
                    if encode_mask is not None:
                        encode_seq_len = cast(Tensor, encode_mask).shape[1]
                    else:
                        encode_seq_len = cast(Tensor, features["input_ids"]).shape[1]
                    selected_cache = extract_last_token_kv(
                        past_key_values=past_key_values,
                        attention_mask=cast(Tensor | None, encode_mask),
                        seq_len=encode_seq_len,
                        batch_size=batch_size,
                        device=device,
                    )
                    kv_cache_mode = "last_token"
            elif generation_kv_mode == "full":
                selected_cache = past_key_values
                kv_cache_mode = "full"
            else:
                raise ValueError(
                    f"generation_kv_mode must be one of {sorted(GENERATION_KV_MODES)}, "
                    f"got {generation_kv_mode!r}"
                )

            if selected_cache is not None:
                gen_batch_size = len(selected_indices)  # G
                # Subset KV cache to selected batch indices
                batch_idx = torch.tensor(
                    selected_indices, device=device, dtype=torch.long,
                )  # (G,)
                selected_kv = _select_batch_from_cache(selected_cache, batch_idx)

                backbone = cast(Any, cast(Any, unwrapped)._modules_dict["backbone"])
                btoks_cache_len = selected_kv.get_seq_length()  # S
                opposite_input_ids = opposite_batch["input_ids"]  # (G, L_opp)
                opposite_seq_len = opposite_input_ids.shape[1]
                opp_mask = opposite_batch.get(
                    "attention_mask",
                    torch.ones_like(opposite_input_ids),
                )  # (G, L_opp)

                start_positions = _compute_generation_start_positions(
                    encode_out=encode_out,
                    features=features,
                    batch_idx=batch_idx,
                    batch_size=gen_batch_size,
                    device=device,
                )  # (G,)

                if kv_cache_mode == "full":
                    prefix_mask = _select_encoder_attention_mask(
                        encode_out=encode_out,
                        features=features,
                        batch_idx=batch_idx,
                        expected_len=btoks_cache_len,
                        dtype=opp_mask.dtype,
                        device=device,
                    )
                else:
                    prefix_mask = None
                if prefix_mask is None:
                    prefix_mask = torch.ones(
                        gen_batch_size, btoks_cache_len,
                        dtype=opp_mask.dtype, device=device,
                    )  # (G, S)
                full_attention_mask = torch.cat(
                    [prefix_mask, opp_mask], dim=1,
                )  # (G, S + L_opp)

                cache_position = torch.arange(
                    btoks_cache_len, btoks_cache_len + opposite_seq_len,
                    device=device,
                )  # (L_opp,)

                position_ids = backbone.build_continuation_position_ids(
                    input_ids=opposite_input_ids,
                    attention_mask=opp_mask,
                    start_positions=start_positions,
                    cache_position=cache_position,
                    image_grid_thw=opposite_batch.get("image_grid_thw"),
                    video_grid_thw=opposite_batch.get("video_grid_thw"),
                )

                gen_forward_inputs = {
                    k: v for k, v in opposite_batch.items()
                    if k not in {"attention_mask", "generation_loss_mask"}
                }

                opposite_outputs = cast(Any, backbone.model)(
                    **gen_forward_inputs,
                    attention_mask=full_attention_mask,
                    past_key_values=selected_kv,
                    labels=labels,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )
                gen_loss = _extract_loss(opposite_outputs)

    return {
        "embeddings": embeddings,
        "gen_loss": gen_loss,
        "kv_cache_mode": kv_cache_mode,
    }


@AutoTrainingArgs.register("btoks_training_args")
@AutoTrainingArgs.register("btoks")
@dataclass
class BToksTrainingArgs(VLM2VecTrainingArgs):
    """Training arguments for BToks trainer.
    - generation_loss_weight: weight for generation loss term
    - generation_loss_duration: number of steps to apply generation loss
    """

    generation_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for generation loss in BToks training"},
    )
    generation_loss_duration: int = field(
        default=0,
        metadata={"help": "Number of global steps to apply generation loss"},
    )
    generation_prefix_text: str = field(
        default="<|im_start|>assistant\n",
        metadata={
            "help": (
                "Generation-target prefix inserted before opposite text so "
                "training matches the cache-decode start protocol"
            )
        },
    )
    generation_suffix_text: str = field(
        default="<|im_end|>",
        metadata={
            "help": "Generation-target suffix appended after opposite text to mark the end boundary"
        },
    )
    include_generation_suffix_in_loss: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether the appended generation end boundary token(s) should "
                "participate in generation loss; the prefix header is always masked"
            )
        },
    )
    generation_kv_mode: str = field(
        default="compressed",
        metadata={
            "help": (
                "KV cache mode for BToks generation loss: 'compressed' uses "
                "btoks-token or last-token KV; 'full' uses full past_key_values."
            )
        },
    )

    def __post_init__(self) -> None:
        """Validate BToks-specific arguments."""
        super().__post_init__()
        if self.generation_loss_duration < 0:
            raise ValueError("generation_loss_duration must be >= 0")
        if self.generation_loss_weight < 0:
            raise ValueError("generation_loss_weight must be >= 0")
        if not self.generation_prefix_text:
            raise ValueError("generation_prefix_text must not be empty")
        if not self.generation_suffix_text:
            raise ValueError("generation_suffix_text must not be empty")
        self.generation_kv_mode = str(self.generation_kv_mode).strip().lower()
        if self.generation_kv_mode not in GENERATION_KV_MODES:
            raise ValueError(
                "generation_kv_mode must be one of "
                f"{sorted(GENERATION_KV_MODES)}, got {self.generation_kv_mode!r}"
            )


@AutoTrainer.register("btoks_trainer")
@AutoTrainer.register("btoks")
class BToksTrainer(VLM2VecTrainer):
    """BToks trainer based on VLM2VecTrainer.

    This trainer keeps GradCache contrastive training from VLM2VecTrainer and
    adds BToks-specific generation loss scheduling. Generation loss is computed
    inside a forward_fn closure injected into model.forward(), ensuring all
    computation happens within a single DDP forward pass.
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: Any | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list | None = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs,
        )

        self.generation_loss_weight = float(getattr(self.args, "generation_loss_weight", 0.0))
        self.generation_loss_duration = int(getattr(self.args, "generation_loss_duration", 0))
        self.generation_kv_mode = str(
            getattr(self.args, "generation_kv_mode", "compressed")
        ).strip().lower()
        if self.generation_kv_mode not in GENERATION_KV_MODES:
            raise ValueError(
                "generation_kv_mode must be one of "
                f"{sorted(GENERATION_KV_MODES)}, got {self.generation_kv_mode!r}"
            )
        logger.info(
            "BToks generation KV mode: %s",
            self.generation_kv_mode,
        )
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        self.generation_prefix_ids = self._encode_generation_text(
            tokenizer,
            cast(str, getattr(self.args, "generation_prefix_text", "<|im_start|>assistant\n")),
            field_name="generation_prefix_text",
        )
        self.generation_suffix_ids = self._encode_generation_text(
            tokenizer,
            cast(str, getattr(self.args, "generation_suffix_text", "<|im_end|>")),
            field_name="generation_suffix_text",
        )
        self.include_generation_suffix_in_loss = bool(
            getattr(self.args, "include_generation_suffix_in_loss", True),
        )
        wrapper_vision_ids = getattr(self.processing_class, "vision_token_ids", None)
        self._generation_visual_token_ids = set(wrapper_vision_ids or [])

    @staticmethod
    def _encode_generation_text(
        tokenizer: Any,
        text: str,
        *,
        field_name: str,
    ) -> list[int]:
        """Encode one generation protocol text into token ids."""
        encoded = tokenizer(text, add_special_tokens=False)
        if hasattr(encoded, "get"):
            token_ids = encoded.get("input_ids")
        elif hasattr(encoded, "input_ids"):
            token_ids = encoded.input_ids
        else:
            token_ids = None

        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim == 0:
                token_ids = [token_ids.item()]
            elif token_ids.ndim == 1:
                token_ids = token_ids.tolist()
            elif token_ids.ndim == 2 and token_ids.shape[0] == 1:
                token_ids = token_ids[0].tolist()
            else:
                raise ValueError(
                    f"{field_name} tokenized to unsupported input_ids shape {tuple(token_ids.shape)}"
                )
        elif hasattr(token_ids, "tolist") and not isinstance(token_ids, (str, bytes)):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, tuple):
            token_ids = list(token_ids)

        if (
            isinstance(token_ids, list)
            and len(token_ids) == 1
            and isinstance(token_ids[0], (list, tuple))
        ):
            token_ids = list(token_ids[0])

        if not isinstance(token_ids, list):
            raise ValueError(f"{field_name} did not tokenize to a list of input_ids")
        if not token_ids:
            raise ValueError(f"{field_name} must tokenize to at least one token")
        return [int(token_id) for token_id in token_ids]

    def _decorate_generation_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Wrap one processed target sample with generation protocol tokens."""
        input_ids = cast(Tensor, sample["input_ids"]).long()
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected sample input_ids shape [1, L], got {tuple(input_ids.shape)}")

        attention_mask = cast(
            Tensor,
            sample.get("attention_mask", torch.ones_like(input_ids)),
        ).long()

        device = input_ids.device
        prefix = torch.tensor(self.generation_prefix_ids, dtype=input_ids.dtype, device=device).view(1, -1)
        suffix = torch.tensor(self.generation_suffix_ids, dtype=input_ids.dtype, device=device).view(1, -1)

        body_loss_mask = torch.ones_like(input_ids, dtype=torch.long)
        if self._generation_visual_token_ids:
            for token_id in self._generation_visual_token_ids:
                body_loss_mask = body_loss_mask.masked_fill(input_ids == token_id, 0)

        prefix_loss_mask = torch.zeros((1, prefix.shape[1]), dtype=torch.long, device=device)
        suffix_loss_value = 1 if self.include_generation_suffix_in_loss else 0
        suffix_loss_mask = torch.full(
            (1, suffix.shape[1]),
            suffix_loss_value,
            dtype=torch.long,
            device=device,
        )

        decorated: dict[str, Any] = {
            **sample,
            "input_ids": torch.cat([prefix, input_ids, suffix], dim=1),
            "attention_mask": torch.cat(
                [
                    torch.ones_like(prefix, dtype=attention_mask.dtype),
                    attention_mask,
                    torch.ones_like(suffix, dtype=attention_mask.dtype),
                ],
                dim=1,
            ),
            "generation_loss_mask": torch.cat(
                [prefix_loss_mask, body_loss_mask, suffix_loss_mask],
                dim=1,
            ),
        }

        mm_token_type_ids = sample.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_tensor = cast(Tensor, mm_token_type_ids).long()
            decorated["mm_token_type_ids"] = torch.cat(
                [
                    torch.zeros((1, prefix.shape[1]), dtype=mm_tensor.dtype, device=mm_tensor.device),
                    mm_tensor,
                    torch.zeros((1, suffix.shape[1]), dtype=mm_tensor.dtype, device=mm_tensor.device),
                ],
                dim=1,
            )

        return decorated

    def _decorate_generation_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply generation protocol decoration to multiple processed target samples."""
        return [self._decorate_generation_sample(sample) for sample in samples]

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> Tensor:
        """Perform one BToks training step."""

        model.train()

        packed_inputs = self._pack_inputs(inputs)
        generation_weight = self._current_generation_weight()

        encode_fn = self._make_encode_fn(model)

        loss_fn = DistributedContrastiveLoss(
            temperature=getattr(self.args, "temperature", 0.02),
            scale_loss=True,
        )

        forward_backward_fn = self._make_forward_backward_fn(
            model=model,
            generation_weight=generation_weight,
            batch_size_per_device=len(packed_inputs[0]),
        )

        contrastive_loss, group_results = grad_cache_accumulate(
            inputs=packed_inputs,
            encode_fns=[encode_fn, encode_fn],
            loss_fn=loss_fn,
            chunk_sizes=[self.gc_chunk_size, self.gc_chunk_size],
            forward_backward_fns=[forward_backward_fn, forward_backward_fn],
            aligned_sort_key_fn=(
                self._packed_self_token_length if self.gc_sort_by_length else None
            ),
            aligned_sort_group_idx=0,
        )

        # Aggregate generation loss from group_results
        weighted_generation_loss: Tensor | None = None
        if generation_weight > 0:
            total_gen_loss = torch.tensor(0.0, device=self.accelerator.device)
            kv_cache_mode: str | None = None
            for group in group_results:
                for result in group:
                    if isinstance(result, dict):
                        mode = result.get("kv_cache_mode")
                        if mode is not None:
                            kv_cache_mode = str(mode)
                        gl = result.get("weighted_gen_loss")
                        if gl is not None:
                            total_gen_loss = total_gen_loss + gl.detach()
            weighted_generation_loss = total_gen_loss
        else:
            kv_cache_mode = None

        # Normalize contrastive_loss: DistributedContrastiveLoss returns
        # CE_mean_global × W (due to scale_loss=True for gradient correctness).
        # Divide by W so the returned loss = CE_mean_global, invariant to W.
        world_size = self.accelerator.num_processes
        normalized_contrastive = contrastive_loss / world_size

        self._log_btoks_metrics(
            contrastive_loss=normalized_contrastive,
            weighted_generation_loss=weighted_generation_loss,
            generation_weight=generation_weight,
            kv_cache_mode=kv_cache_mode,
        )

        return normalized_contrastive

    def _current_generation_weight(self) -> float:
        """Return generation loss weight for current global step."""
        if self.generation_loss_weight <= 0:
            return 0.0

        if self.generation_loss_duration <= 0:
            return 0.0

        if self.state.global_step >= self.generation_loss_duration:
            return 0.0

        return self.generation_loss_weight

    def _pack_inputs(self, inputs: dict[str, Any]) -> list[list[dict[str, Any]]]:
        """Pack collator outputs into query/positive groups for GradCache.

        Converts ntp_side target-side strings into per-sample should_generate
        booleans. After this point, data uses self/opposite semantics: the
        selected self side provides btoks KV and its opposite side is generated.
        """
        query_samples = inputs["query"]
        positive_samples = inputs["positive"]
        if len(query_samples) != len(positive_samples):
            raise ValueError(
                "BToksTrainer requires query and positive groups to have the "
                "same number of samples; got "
                f"query={len(query_samples)}, positive={len(positive_samples)}"
            )
        batch_size = len(query_samples)

        metadata = inputs.get("metadata", {})
        ntp_side_values = metadata.get("ntp_side") if isinstance(metadata, dict) else None
        modes = self._normalize_generate_modes(ntp_side_values, batch_size)

        query_group: list[dict[str, Any]] = []
        positive_group: list[dict[str, Any]] = []
        for index in range(batch_size):
            mode = modes[index]
            query_group.append(
                {
                    "self": query_samples[index],
                    "opposite": positive_samples[index],
                    "should_generate": mode in {"positive", "both"},
                }
            )
            positive_group.append(
                {
                    "self": positive_samples[index],
                    "opposite": query_samples[index],
                    "should_generate": mode in {"query", "both"},
                }
            )

        return [query_group, positive_group]

    @staticmethod
    def _packed_self_token_length(sample: dict[str, Any]) -> int:
        """Return effective token length for the self side of a packed sample."""
        self_sample = sample.get("self")
        if not isinstance(self_sample, dict):
            raise ValueError("Packed BToks sample must include a dict-valued 'self' field")
        return VLM2VecTrainer._input_token_length(self_sample)

    def _normalize_generate_modes(
        self,
        ntp_side_values: Any,
        batch_size: int,
    ) -> list[str]:
        """Normalize ntp_side target-side values to none/query/positive/both modes."""
        if not isinstance(ntp_side_values, (list, tuple)):
            return ["none"] * batch_size

        modes: list[str] = []
        for index in range(batch_size):
            raw_value = ntp_side_values[index] if index < len(ntp_side_values) else "none"
            value = str(raw_value).strip().lower()

            if value in {"none", "off", "disabled"}:
                modes.append("none")
            elif value in {"query", "qry"}:
                modes.append("query")
            elif value in {"positive", "pos", "target", "tgt"}:
                modes.append("positive")
            elif value in {"both", "all"}:
                modes.append("both")
            else:
                modes.append("none")

        return modes

    def _make_encode_fn(self, model) -> Callable[[list[dict[str, Any]]], Tensor]:
        """Build GradCache phase-A encode closure."""
        wrapper = self.processing_class

        def encode_fn(chunk: list[dict[str, Any]]) -> Tensor:
            with self.accelerator.autocast():
                my_samples = [sample["self"] for sample in chunk]
                batch = batch_processor_outputs(my_samples, wrapper)
                batch = self._prepare_inputs(batch)
                with self.accelerator.no_sync(model):
                    outputs = model(**batch)
                return outputs["embeddings"]

        return encode_fn

    def _make_forward_backward_fn(
        self,
        model: nn.Module,
        generation_weight: float,
        batch_size_per_device: int,
    ) -> Callable[[list[dict[str, Any]], Tensor, ChunkContext], dict[str, Any]]:
        """Build GradCache phase-B forward_backward closure.

        This closure handles:
        1. Organize inputs (my_batch, opposite_batch, labels, selected_indices)
        2. Invoke model.forward() with _btoks_forward_fn
        3. Compute surrogate loss + optional weighted generation loss
        4. DDP sync control (no_sync wraps both forward and backward)
        5. Backward

        Generation loss is normalized by (num_gen_samples / batch_size_per_device)
        so that each sample contributes equally regardless of chunk_size or
        world_size. This makes the effective gradient = mean over global batch.
        """
        wrapper = self.processing_class

        def forward_backward_fn(
            chunk: list[dict[str, Any]],
            cache: Tensor,
            ctx: ChunkContext,
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}

            is_last = (
                ctx.group_idx == ctx.num_groups - 1
                and ctx.chunk_idx == ctx.num_chunks - 1
            )

            # ---- 1. Organize inputs ----
            my_samples = [sample["self"] for sample in chunk]
            my_batch = batch_processor_outputs(my_samples, wrapper)
            my_batch = self._prepare_inputs(my_batch)

            selected_indices: list[int] = []
            opposite_batch: dict[str, Any] | None = None
            labels: Tensor | None = None

            if generation_weight > 0:
                selected_indices = [
                    i for i, sample in enumerate(chunk)
                    if sample.get("should_generate", False)
                ]
                if selected_indices:
                    opposite_samples = [
                        chunk[i]["opposite"] for i in selected_indices
                    ]
                    opposite_samples = self._decorate_generation_samples(opposite_samples)
                    opposite_batch = batch_processor_outputs(
                        opposite_samples, wrapper,
                    )
                    opposite_batch = self._prepare_inputs(opposite_batch)
                    labels = self._build_generation_labels(
                        opposite_batch, model,
                    )

            # ---- 2. Forward + backward with DDP sync control ----
            my_batch["forward_fn"] = lambda unwrapped, **features: (
                _btoks_forward_fn(
                    unwrapped,
                    selected_indices=selected_indices,
                    opposite_batch=opposite_batch,
                    labels=labels,
                    generation_kv_mode=self.generation_kv_mode,
                    **features,
                )
            )

            sync_ctx = nullcontext() if is_last else self.accelerator.no_sync(model)
            with sync_ctx:
                # autocast wraps forward + surrogate only; backward runs outside
                # per PyTorch best practice
                with self.accelerator.autocast():
                    outputs = model(**my_batch)
                    embeddings = outputs["embeddings"]  # (B, D)

                    surrogate = torch.dot(embeddings.flatten(), cache.flatten())
                    gen_loss = outputs.get("gen_loss")
                    kv_cache_mode = outputs.get("kv_cache_mode")
                    if kv_cache_mode is not None:
                        result["kv_cache_mode"] = kv_cache_mode
                    if gen_loss is not None and selected_indices:
                        # Normalize: each sample gets equal weight in the
                        # global batch. scale = G_chunk / B_local so that
                        # accumulated gradient = mean over device batch,
                        # invariant to chunk_size and world_size.
                        scale = len(selected_indices) / batch_size_per_device
                        weighted = gen_loss * generation_weight * scale
                        result["weighted_gen_loss"] = weighted
                        surrogate = surrogate + weighted

                self.accelerator.backward(surrogate)

            return result

        return forward_backward_fn

    def _build_generation_labels(
        self,
        opposite_batch: dict[str, Any],
        model: nn.Module,
    ) -> Tensor:
        """Build labels for generation loss from opposite batch.

        Args:
            opposite_batch: Already sliced to selected indices.  # (G, L_opp)
            model: DDP-wrapped model (unwrapped internally).

        Returns:
            Labels tensor of shape (G, L_opp) with -100 at masked positions.
        """
        input_ids = opposite_batch["input_ids"]  # (G, L_opp)
        labels = input_ids.clone()

        # Mask padding positions
        attention_mask = opposite_batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        generation_loss_mask = opposite_batch.get("generation_loss_mask")
        if generation_loss_mask is not None:
            labels = labels.masked_fill(cast(Tensor, generation_loss_mask) == 0, -100)

            btoks_token_mask = opposite_batch.get("btoks_token_mask")
            if btoks_token_mask is not None:
                labels = labels.masked_fill(btoks_token_mask, -100)

            return labels

        # Mask vision token positions
        unwrapped = self.accelerator.unwrap_model(model)
        backbone = unwrapped._modules_dict["backbone"]

        vision_token_ids = []
        if hasattr(backbone.model, "config"):
            vision_token_ids = get_vision_token_ids(backbone.model.config)

        for vision_token_id in vision_token_ids:
            labels = labels.masked_fill(input_ids == vision_token_id, -100)

        # Mask btoks token positions if present
        btoks_token_mask = opposite_batch.get("btoks_token_mask")
        if btoks_token_mask is not None:
            labels = labels.masked_fill(btoks_token_mask, -100)

        return labels

    def _log_btoks_metrics(
        self,
        contrastive_loss: Tensor,
        weighted_generation_loss: Tensor | None,
        generation_weight: float,
        kv_cache_mode: str | None = None,
    ) -> None:
        """Log BToks-specific metrics via Trainer.log() for wandb/tensorboard.

        Uses HuggingFace Trainer's built-in logging so metrics are automatically
        sent to all configured loggers (wandb, tensorboard, etc.).

        Args:
            contrastive_loss: Already normalized (CE_mean_global, ÷W applied).
            weighted_generation_loss: Already normalized per sample via
                G_chunk / B_local scaling.
            generation_weight: Current generation loss weight.
        """
        generation_loss_value = (
            float(weighted_generation_loss.detach().item())
            if weighted_generation_loss is not None
            else 0.0
        )
        contrastive_loss_value = float(contrastive_loss.detach().item())

        metrics = {
            "btoks/contrastive_loss": contrastive_loss_value,
            "btoks/generation_loss": generation_loss_value,
            "btoks/generation_weight": generation_weight,
            "btoks/generation_kv_mode_compressed": float(
                self.generation_kv_mode == "compressed"
            ),
            "btoks/generation_kv_mode_full": float(self.generation_kv_mode == "full"),
            "btoks/kv_cache_mode_btoks_token": float(kv_cache_mode == "btoks_token"),
            "btoks/kv_cache_mode_last_token": float(kv_cache_mode == "last_token"),
            "btoks/kv_cache_mode_full": float(kv_cache_mode == "full"),
            "btoks/total_loss": contrastive_loss_value + generation_loss_value,
        }
        self.log(metrics)


__all__ = [
    "BToksTrainer",
    "BToksTrainingArgs",
]
