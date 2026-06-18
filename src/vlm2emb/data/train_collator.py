"""GradCache utilities for splitting VLM inputs.

This module provides utility functions used by GradCache in BToksTrainer
for splitting model inputs into smaller chunks during gradient accumulation.
"""

from itertools import repeat

import torch


def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    """Split VLM inputs into chunks for GradCache.

    Args:
        model_input: Dict with a single key containing model inputs.
        chunk_size: Number of samples per chunk.

    Returns:
        List of chunked input dicts.
    """
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [
                arg_val[k][i : i + chunk_size]
                for i in range(0, len(arg_val[k]), chunk_size)
            ]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [
        dict(zip(kk, tt, strict=True))
        for kk, tt in zip(repeat(keys), zip(*chunked_tensors, strict=True), strict=True)
    ]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def get_dense_rep(x: dict) -> torch.Tensor:
    """Get dense representation from model output.

    Args:
        x: Model output dict with 'qry_reps' and/or 'tgt_reps'.

    Returns:
        Query representations if available, otherwise target representations.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    return x["qry_reps"]
