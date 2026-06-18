import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import Tensor

from .context_managers import RandContext

__all__ = ["grad_cache_accumulate", "ChunkContext"]

logger = logging.getLogger(__name__)


# =============================================================================
# grad_cache_accumulate: pure functional gradient-cache implementation.
# =============================================================================


class ChunkContext(NamedTuple):
    """Context passed to forward_backward_fns for progress tracking.

    Attributes:
        group_idx: Current active group ordinal (0-based)
        chunk_idx: Current chunk index within the group (0-based)
        num_groups: Total number of active groups
        num_chunks: Total number of chunks in the current group
    """
    group_idx: int
    chunk_idx: int
    num_groups: int
    num_chunks: int


def _default_split_fn(samples: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split samples into consecutive chunks of at most chunk_size items."""
    return [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]


def _get_input_tensors(model_input: Any) -> list[Tensor]:
    """Recursively collect tensors from nested model inputs for RNG capture."""
    if isinstance(model_input, Tensor):
        return [model_input]
    elif isinstance(model_input, (list, tuple)):
        tensors = []
        for x in model_input:
            tensors.extend(_get_input_tensors(x))
        return tensors
    elif isinstance(model_input, dict):
        tensors = []
        for x in model_input.values():
            tensors.extend(_get_input_tensors(x))
        return tensors
    else:
        return []


def _sort_aligned_inputs(
    inputs: list[list[Any]],
    *,
    sort_key_fn: Callable[[Any], Any],
    sort_group_idx: int,
) -> list[list[Any]]:
    """Sort aligned input groups with one stable permutation.

    Args:
        inputs: Input groups whose rows are aligned by index.
        sort_key_fn: Function that returns the sort key for samples in the
            baseline group.
        sort_group_idx: Baseline group used to compute the permutation.

    Returns:
        New input groups sorted by the same permutation.

    Raises:
        ValueError: If groups cannot be sorted with one shared permutation.
    """
    if sort_group_idx < 0 or sort_group_idx >= len(inputs):
        raise ValueError(
            f"aligned_sort_group_idx ({sort_group_idx}) is out of range for "
            f"{len(inputs)} input groups"
        )

    group_lengths = [len(group) for group in inputs]
    if not any(group_lengths):
        return [list(group) for group in inputs]

    sort_group_len = group_lengths[sort_group_idx]
    if sort_group_len == 0 or any(length != sort_group_len for length in group_lengths):
        raise ValueError(
            "aligned sorting requires all input groups to have the same number "
            f"of samples; got group lengths {group_lengths}"
        )

    sort_group = inputs[sort_group_idx]
    permutation = sorted(
        range(sort_group_len),
        key=lambda idx: (sort_key_fn(sort_group[idx]), idx),
    )
    return [[group[idx] for idx in permutation] for group in inputs]


def grad_cache_accumulate(
    inputs: list[list[Any]],
    encode_fns: list[Callable[[list[Any]], Tensor]],
    loss_fn: Callable[..., Tensor],
    chunk_sizes: list[int],
    *,
    forward_backward_fns: list[Callable[[list[Any], Tensor, ChunkContext], Any]] | None = None,
    split_fn: Callable[[list[Any], int], list[list[Any]]] | None = None,
    aligned_sort_key_fn: Callable[[Any], Any] | None = None,
    aligned_sort_group_idx: int = 0,
) -> tuple[Tensor, list[list[Any]]]:
    """Gradient cache accumulation for losses requiring full-batch representations.

    Algorithm:
        Phase A: Run no-grad encoding to collect all representations.
        Phase A->B: Compute the full-batch loss and build gradient caches.
        Phase B: Replay chunks with gradients and call forward_backward_fns.

    Args:
        inputs: N input groups, each a list of samples. Empty groups are skipped.
        encode_fns: N encode functions with signature
            ``(samples: list[Any]) -> reps: Tensor(batch, dim)``.
        loss_fn: Phase-A loss function with signature
            ``(*reps: Tensor) -> loss: Tensor``.
        chunk_sizes: Chunk size for each input group.
        forward_backward_fns: Optional Phase-B functions with signature
            ``(chunk: list[Any], cache: Tensor, ctx: ChunkContext) -> Any``.
            If None, uses the default surrogate-loss backward path.
        split_fn: Optional custom split function with signature
            ``(samples: list[Any], chunk_size: int) -> list[list[Any]]``.
        aligned_sort_key_fn: Optional sort key for aligned sorting before chunking.
                             The key is evaluated on samples from
                             aligned_sort_group_idx and the resulting stable
                             ascending permutation is applied to every group.
        aligned_sort_group_idx: Input group used to compute the aligned sort
                                permutation. Defaults to the query group (0).

    Returns:
        tuple of:
            - loss: Detached contrastive loss tensor
            - group_results: list[list[Any]], per-group list of forward_backward_fn return values.
              Empty lists for groups using default backward.
    """
    split = split_fn or _default_split_fn
    working_inputs = (
        _sort_aligned_inputs(
            inputs,
            sort_key_fn=aligned_sort_key_fn,
            sort_group_idx=aligned_sort_group_idx,
        )
        if aligned_sort_key_fn is not None
        else inputs
    )

    # =========================================================================
    # Phase A: no-grad forward pass to collect representations and RNG states.
    # =========================================================================

    all_chunks: list[list[list[Any]]] = []  # [group][chunk][sample]
    all_reps: list[list[Tensor]] = []       # [group][chunk] -> Tensor
    all_rand_states: list[list[RandContext]] = []

    # Track which groups are active (non-empty)
    active_group_indices: list[int] = []

    for group_idx, (group_input, chunk_size) in enumerate(zip(working_inputs, chunk_sizes, strict=True)):
        if not group_input:
            all_chunks.append([])
            all_reps.append([])
            all_rand_states.append([])
            continue

        active_group_indices.append(group_idx)
        chunks = split(group_input, chunk_size)
        all_chunks.append(chunks)

        chunk_reps = []
        chunk_states = []
        for chunk in chunks:
            input_tensors = _get_input_tensors(chunk)
            state = RandContext(*input_tensors)
            with state:
                with torch.no_grad():
                    reps = encode_fns[group_idx](chunk)
                reps = reps.detach().requires_grad_(True)
                chunk_reps.append(reps)
            chunk_states.append(state)

        all_reps.append(chunk_reps)
        all_rand_states.append(chunk_states)

    if not active_group_indices:
        zero = torch.tensor(0.0)
        return zero, [[] for _ in inputs]

    # =========================================================================
    # Phase A->B: compute loss and build gradient cache tensors.
    # =========================================================================

    # Concatenate chunk reps per group for loss computation
    full_reps = []
    for group_idx in active_group_indices:
        full_reps.append(torch.cat(all_reps[group_idx], dim=0))

    loss = loss_fn(*full_reps)

    # Build gradient cache: grad of loss w.r.t. each chunk's reps
    all_cache_grads: list[list[Tensor]] = [[] for _ in range(len(inputs))]
    cache_inputs = []
    cache_group_map = []
    for group_idx in active_group_indices:
        for chunk_reps in all_reps[group_idx]:
            cache_inputs.append(chunk_reps)
            cache_group_map.append(group_idx)

    grads = torch.autograd.grad(loss, cache_inputs)

    grad_offset = 0
    for group_idx in active_group_indices:
        num_chunks = len(all_reps[group_idx])
        for _j in range(num_chunks):
            all_cache_grads[group_idx].append(grads[grad_offset])
            grad_offset += 1

    # =========================================================================
    # Phase B: replay chunks with gradients and run backward.
    # =========================================================================

    group_results: list[list[Any]] = [[] for _ in range(len(inputs))]
    num_active = len(active_group_indices)

    for active_group_idx, group_idx in enumerate(active_group_indices):
        chunks = all_chunks[group_idx]
        cache_chunks = all_cache_grads[group_idx]
        rand_states = all_rand_states[group_idx]
        num_chunks_in_group = len(chunks)

        has_custom_fn = (
            forward_backward_fns is not None
            and group_idx < len(forward_backward_fns)
            and forward_backward_fns[group_idx] is not None
        )

        for chunk_idx, (chunk, cache, state) in enumerate(
            zip(chunks, cache_chunks, rand_states, strict=True)
        ):
            ctx = ChunkContext(
                group_idx=active_group_idx,
                chunk_idx=chunk_idx,
                num_groups=num_active,
                num_chunks=num_chunks_in_group,
            )

            if has_custom_fn:
                # Trainer-provided forward_backward_fn handles everything
                # Restore random state so dropout etc. match Phase A
                fb_fn = forward_backward_fns[group_idx]  # type: ignore[index]
                with state:
                    result = fb_fn(chunk, cache, ctx)
                group_results[group_idx].append(result)
            else:
                # Default: surrogate loss backward
                with state:
                    reps = encode_fns[group_idx](chunk)

                if isinstance(reps, dict):
                    raise TypeError(
                        "Expected Tensor reps when forward_backward_fns is None, got dict. "
                        "Please provide forward_backward_fns for dict outputs."
                    )
                surrogate = torch.dot(reps.flatten(), cache.flatten())
                surrogate.backward()
                surrogate = None

    return loss.detach(), group_results
