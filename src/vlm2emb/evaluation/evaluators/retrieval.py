"""Retrieval evaluator.

Provides evaluation functionality for retrieval tasks with explicit shard and
gather helpers instead of implicit accelerator dataloader splitting.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import torch
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vlm2emb.auto import AutoEvaluator
from vlm2emb.data.collators import EvalCollator
from vlm2emb.evaluation.distributed import (
    ShardSpec,
    build_shard_position_tensor,
    build_shard_spec,
    build_sharded_dataset_view,
    gather_tensor_first_dim,
    reorder_gathered_tensor,
    trim_gathered_tensor,
)
from vlm2emb.evaluation.evaluators.base import BaseEvaluator
from vlm2emb.evaluation.metrics import DEFAULT_RETRIEVAL_METRICS, RankingMetrics
from vlm2emb.utils.logging import RankLogger

if TYPE_CHECKING:
    import torch.nn as nn
    from accelerate import Accelerator

    from vlm2emb.data.datasets.eval import RetrievalEvalDataset

logger = RankLogger(__name__)

PROGRESS_LOG_INTERVAL_SECONDS = 5 * 60


@AutoEvaluator.register("retrieval")
class RetrievalEvaluator(BaseEvaluator):
    """Evaluator for retrieval tasks.

    Constructor receives evaluation config only. Runtime resources
    (model, processor_wrapper, accelerator) are passed to __call__.

    Args:
        dataset: Retrieval evaluation dataset
        metrics: Complete metric names to compute (e.g., ("hit@1", "ndcg@5"))
        batch_size: Default batch size for encoding (overridable at call time)
        num_workers: Default number of DataLoader workers (overridable at call time)
        show_progress: Whether to show progress bar
    """

    def __init__(
        self,
        dataset: RetrievalEvalDataset,
        metrics: list[str] | tuple[str, ...] | None = None,
        batch_size: int = 32,
        num_workers: int = 1,
        show_progress: bool = True,
    ):
        super().__init__(
            name=dataset.name,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        self.dataset = dataset
        self.metrics = tuple(metrics) if metrics is not None else DEFAULT_RETRIEVAL_METRICS
        self.num_workers = num_workers
        self._metrics_computer = RankingMetrics(metrics=self.metrics)

    def __call__(
        self,
        model: nn.Module,
        processor_wrapper: Any,
        accelerator: Accelerator | None = None,
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate model on the dataset.

        Args:
            model: Model to evaluate
            processor_wrapper: ProcessorWrapper for data preprocessing
            accelerator: Optional accelerator for distributed evaluation
            batch_size: Override default batch size
            num_workers: Override default num_workers

        Returns:
            Dictionary of metric scores {"hit@1": x, "ndcg@5": y, ...}
        """
        bs = batch_size if batch_size is not None else self.batch_size
        nw = num_workers if num_workers is not None else self.num_workers

        logger.info(
            f"{self.dataset.name}: "
            f"queries={len(self.dataset.queries)}, "
            f"candidates={len(self.dataset.candidates)}"
        )
        logger.debug(
            f"Distributed: {self._is_distributed(accelerator)}, "
            f"Process: {self._process_index(accelerator)}/{self._num_processes(accelerator)}"
        )

        model.eval()

        # Keep query and candidate embeddings on device; metrics consume tensors directly.
        logger.info("[1/3] Encoding queries...")
        query_embeds = self._encode_with_dataloader(
            model,
            processor_wrapper,
            accelerator,
            self.dataset.queries,
            "queries",
            batch_size=bs,
            num_workers=nw,
        )

        logger.info("[2/3] Encoding candidates...")
        cand_embeds = self._encode_with_dataloader(
            model,
            processor_wrapper,
            accelerator,
            self.dataset.candidates,
            "candidates",
            batch_size=bs,
            num_workers=nw,
        )

        logger.info("[3/3] Computing metrics...")

        if self._is_main_process(accelerator):
            pred_tensors = self._compute_predictions(query_embeds, cand_embeds)
            scores = self._metrics_computer.compute(
                pred_tensors["relevance"],
                pred_tensors["n_relevant"],
                pred_tensors["all_rel_scores"],
            )
            logger.info(f"{self.dataset.name}: hit@1={scores.get('hit@1', 0):.4f}")
        else:
            scores = {}

        scores = self._broadcast_scores(scores, accelerator)

        # Keep all ranks aligned before the benchmark moves to the next evaluator.
        if self._is_distributed(accelerator) and accelerator is not None:
            accelerator.wait_for_everyone()

        logger.debug(f"Done: {self.dataset.name}")
        return scores

    # ================================================================
    # DataLoader-based encoding
    # ================================================================

    def _broadcast_scores(
        self,
        scores: dict[str, float],
        accelerator: Accelerator | None,
    ) -> dict[str, float]:
        """Broadcast single-dataset scores from global rank 0 to all ranks."""
        if not self._is_distributed(accelerator):
            return scores
        return cast(dict[str, float], broadcast_object_list([scores], from_process=0)[0])

    def _log_protocol_state(
        self,
        *,
        desc: str,
        stage: str,
        shard: ShardSpec,
        local_count: int,
        gather_counts: list[int] | None = None,
        tensor_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Log one retrieval shard/gather protocol state via the rank-aware logger.

        RankLogger defaults to rank 0, so this stays DEBUG-only without flooding
        the permanent INFO surface.
        """
        logger.debug(
            (
                "retrieval_protocol dataset=%s desc=%s stage=%s rank=%s world_size=%s "
                "shard_start=%s shard_stop=%s shard_blocks=%s local_count=%s "
                "gather_counts=%s trimmed_shape=%s"
            ),
            self.dataset.name,
            desc,
            stage,
            shard.rank,
            shard.world_size,
            shard.start,
            shard.stop,
            shard.block_ranges,
            local_count,
            gather_counts,
            tensor_shape,
        )

    def _encode_with_dataloader(
        self,
        model: nn.Module,
        processor_wrapper: Any,
        accelerator: Accelerator | None,
        dataset: Any,
        desc: str,
        batch_size: int = 16,
        num_workers: int = 1,
    ) -> torch.Tensor:
        """Encode dataset using DataLoader with distributed support.

        Embeddings stay as ``torch.Tensor`` objects on the active model device.
        Distributed runs shard data explicitly and gather tensors without using
        ``accelerator.prepare()`` or implicit dataloader splitting.

        Args:
            model: Model to use for encoding
            processor_wrapper: ProcessorWrapper for data preprocessing
            accelerator: Optional accelerator for distributed evaluation
            dataset: Dataset to encode (LanceDataset)
            desc: Description for progress bar
            batch_size: Batch size for DataLoader
            num_workers: Number of DataLoader workers

        Returns:
            Embeddings tensor ``[N, D]`` on the main process. Non-main processes
            return an empty ``[0, D]`` tensor after participating in gather.
        """
        total_size = len(dataset)
        if accelerator is not None and not hasattr(accelerator, "process_index"):
            raise TypeError(
                "RetrievalEvaluator requires an Accelerator-compatible runtime with process_index"
            )

        # Assign dataloader-sized blocks to ranks and keep original sample
        # positions so rank 0 can restore global order after gather.
        shard = build_shard_spec(
            total_size=total_size,
            world_size=self._num_processes(accelerator),
            rank=self._process_index(accelerator),
            block_size=batch_size,
        )
        self._log_protocol_state(
            desc=desc,
            stage="start",
            shard=shard,
            local_count=shard.real_count,
        )
        dataset_view = build_sharded_dataset_view(dataset, shard)

        collator = EvalCollator(processor_wrapper=processor_wrapper)
        dataloader = DataLoader(
            dataset_view,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

        local_embeds_list: list[torch.Tensor] = []
        device = self._get_model_device(model)
        local_positions = build_shard_position_tensor(shard, device=device)

        num_batches = len(dataloader)
        iterator = dataloader
        if self.show_progress:
            iterator = tqdm(
                dataloader,
                desc=f"Encoding {desc}",
                total=num_batches,
                disable=not self._is_main_process(accelerator),
            )

        if accelerator is not None:
            autocast_ctx = accelerator.autocast()
        elif torch.cuda.is_available():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

        encode_started_at = perf_counter()
        next_progress_log_at = encode_started_at + PROGRESS_LOG_INTERVAL_SECONDS
        first_batch_wait_seconds: float | None = None

        with torch.no_grad(), autocast_ctx:
            for batch_index, batch in enumerate(iterator, 1):
                if first_batch_wait_seconds is None:
                    first_batch_wait_seconds = perf_counter() - encode_started_at
                    logger.debug(
                        "%s: first %s batch ready after %.2fs (batch_size=%d, num_workers=%d)",
                        self.dataset.name,
                        desc,
                        first_batch_wait_seconds,
                        batch_size,
                        num_workers,
                    )
                inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in batch["inputs"].items()
                }
                embeddings = self._get_embeddings(model, inputs)
                local_embeds_list.append(embeddings)
                now = perf_counter()
                if now >= next_progress_log_at:
                    processed_samples = min(batch_index * batch_size, shard.real_count)
                    logger.info(
                        (
                            "%s: %s encode progress rank=%d batches=%d/%d "
                            "samples=%d/%d elapsed=%.2fs first_batch_wait=%.2fs "
                            "batch_size=%d num_workers=%d"
                        ),
                        self.dataset.name,
                        desc,
                        shard.rank,
                        batch_index,
                        num_batches,
                        processed_samples,
                        shard.real_count,
                        now - encode_started_at,
                        first_batch_wait_seconds or 0.0,
                        batch_size,
                        num_workers,
                        ranks=[-1],
                    )
                    next_progress_log_at = now + PROGRESS_LOG_INTERVAL_SECONDS

        total_encode_seconds = perf_counter() - encode_started_at
        logger.info(
            (
                "%s: %s encode finished rank=%d in %.2fs "
                "(first_batch_wait=%.2fs, local_batches=%d, local_samples=%d/%d)"
            ),
            self.dataset.name,
            desc,
            shard.rank,
            total_encode_seconds,
            first_batch_wait_seconds or 0.0,
            len(local_embeds_list),
            sum(int(embeds.shape[0]) for embeds in local_embeds_list),
            shard.real_count,
            ranks=[-1],
        )

        if local_embeds_list:
            local_embeds = torch.cat(local_embeds_list, dim=0)
        else:
            local_embeds = torch.empty(0, 0, device=device)

        self._log_protocol_state(
            desc=desc,
            stage="local-ready",
            shard=shard,
            local_count=int(local_embeds.shape[0]),
            tensor_shape=tuple(local_embeds.shape),
        )

        logger.debug(
            "%s: shard_blocks=%s local_embeds=%s local_positions=%s",
            desc,
            shard.block_ranges,
            tuple(local_embeds.shape),
            tuple(local_positions.shape),
        )

        result_embeds = local_embeds

        if accelerator is not None and self._is_distributed(accelerator):
            logger.debug("Gathering from all processes with explicit counts handshake...")
            # Gather embeddings and positions with the same first-dimension counts.
            # Positions are the source of truth for restoring block-cyclic order.
            gathered_embeds, embed_counts = gather_tensor_first_dim(local_embeds)
            gathered_positions, position_counts = gather_tensor_first_dim(local_positions)
            if position_counts != embed_counts:
                raise ValueError(
                    f"{desc}: gathered position counts {position_counts} do not match embedding counts {embed_counts}"
                )
            self._log_protocol_state(
                desc=desc,
                stage="pre-trim",
                shard=shard,
                local_count=int(local_embeds.shape[0]),
                gather_counts=embed_counts,
                tensor_shape=tuple(gathered_embeds.shape),
            )
            rank_major_embeds = trim_gathered_tensor(
                gathered_embeds,
                counts=embed_counts,
                total_size=total_size,
            )
            self._log_protocol_state(
                desc=desc,
                stage="post-trim",
                shard=shard,
                local_count=int(local_embeds.shape[0]),
                gather_counts=embed_counts,
                tensor_shape=tuple(rank_major_embeds.shape),
            )
            logger.debug(
                "%s: gathered_embeds=%s gathered_positions=%s counts=%s",
                desc,
                tuple(gathered_embeds.shape),
                tuple(gathered_positions.shape),
                embed_counts,
            )
            if self._is_main_process(accelerator):
                result_embeds = reorder_gathered_tensor(
                    gathered_embeds,
                    positions=gathered_positions,
                    counts=embed_counts,
                    total_size=total_size,
                )
                self._log_protocol_state(
                    desc=desc,
                    stage="reordered",
                    shard=shard,
                    local_count=int(local_embeds.shape[0]),
                    gather_counts=embed_counts,
                    tensor_shape=tuple(result_embeds.shape),
                )
            else:
                result_embeds = rank_major_embeds
        else:
            self._log_protocol_state(
                desc=desc,
                stage="post-trim",
                shard=shard,
                local_count=int(local_embeds.shape[0]),
                gather_counts=[int(local_embeds.shape[0])],
                tensor_shape=tuple(local_embeds.shape),
            )

        if self._is_main_process(accelerator):
            if result_embeds.shape[0] != total_size:
                raise ValueError(
                    f"{desc}: gathered embeddings shape {result_embeds.shape} does not match total_size={total_size}"
                )
            logger.debug("Final ordered %s embeddings: %s", desc, tuple(result_embeds.shape))
            return result_embeds

        embed_dim = 0
        if result_embeds.ndim == 2:
            embed_dim = int(result_embeds.shape[1])
        return torch.empty(0, embed_dim, device=device)

    # ================================================================
    # Prediction computation (GPU tensor path)
    # ================================================================

    @staticmethod
    def _build_per_query_relevance(
        *,
        qrel_candidate_ids: list[Any],
        qrel_candidate_scores: list[Any],
        cand_id_to_idx: dict[Any, int],
        n_query: int,
    ) -> tuple[list[dict[int, float]], int]:
        """Convert qrels from candidate IDs to candidate-index relevance maps."""
        per_query_rel: list[dict[int, float]] = []
        max_n_rel = 0

        for i in range(n_query):
            cand_id_list = qrel_candidate_ids[i] or []
            cand_score_list = qrel_candidate_scores[i] or []
            if len(cand_id_list) != len(cand_score_list):
                raise ValueError(
                    f"qrels[{i}] candidate_ids/candidate_scores length mismatch: "
                    f"{len(cand_id_list)} != {len(cand_score_list)}"
                )

            missing_candidate_ids = [
                cid for cid in cand_id_list if cid is not None and cid not in cand_id_to_idx
            ]
            if missing_candidate_ids:
                raise ValueError(
                    f"qrels[{i}] references missing candidate ids: {missing_candidate_ids!r}"
                )

            rel_map: dict[int, float] = {}
            for cid, score in zip(cand_id_list, cand_score_list, strict=True):
                if cid is None:
                    continue
                rel_score = float(score) if score is not None else 0.0
                if rel_score > 0:
                    rel_map[cand_id_to_idx[cid]] = rel_score

            per_query_rel.append(rel_map)
            max_n_rel = max(max_n_rel, len(rel_map))

        return per_query_rel, max(max_n_rel, 1)

    @staticmethod
    def _write_query_metric_inputs(
        *,
        relevance: torch.Tensor,
        all_rel_scores: torch.Tensor,
        query_index: int,
        ordered_candidate_indices: torch.Tensor | list[int],
        rel_map: dict[int, float],
    ) -> None:
        """Populate metric input tensors for one query."""
        for rank, cand_idx_raw in enumerate(ordered_candidate_indices):
            cand_idx = (
                int(cand_idx_raw.item())
                if isinstance(cand_idx_raw, torch.Tensor)
                else int(cand_idx_raw)
            )
            if cand_idx in rel_map:
                relevance[query_index, rank] = rel_map[cand_idx]

        # RankingMetrics sorts this row for ideal DCG, so insertion order is enough.
        for rank, score in enumerate(rel_map.values()):
            all_rel_scores[query_index, rank] = score

    def _compute_predictions(
        self,
        query_embeds: torch.Tensor,
        cand_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute top-K prediction tensors using GPU matmul and topk.

        Returns GPU tensors for ``RankingMetrics.compute()``.

        Args:
            query_embeds: Query embeddings [N_query, D] on GPU
            cand_embeds: Candidate embeddings [N_cand, D] on GPU

        Returns:
            dict with keys:
                - relevance: [N_query, max_k] relevance scores at top-K positions
                - n_relevant: [N_query] total relevant docs per query
                - all_rel_scores: [N_query, max_n_rel] for ideal DCG
        """
        device = query_embeds.device
        n_query = query_embeds.shape[0]
        n_cand = cand_embeds.shape[0]
        max_k = self._metrics_computer.max_k

        # Bulk table reads keep Lance access out of the per-query scoring loop.
        cand_ids_table = self.dataset.candidates.to_table(columns=["id"])
        cand_ids = cand_ids_table.column("id").to_pylist()

        query_ids_table = self.dataset.queries.to_table(columns=["id"])
        query_ids = query_ids_table.column("id").to_pylist()
        cand_id_to_idx = {cid: idx for idx, cid in enumerate(cand_ids)}

        if n_query != len(query_ids):
            raise ValueError(
                f"query embeddings count ({n_query}) != query ids count ({len(query_ids)})"
            )
        if n_cand != len(cand_ids):
            raise ValueError(
                f"candidate embeddings count ({n_cand}) != candidate ids count ({len(cand_ids)})"
            )

        qrels_table = self.dataset.qrels.to_table()
        qrel_query_ids = qrels_table.column("query_id").to_pylist()
        qrel_modes = qrels_table.column("mode").to_pylist()
        qrel_cand_ids_col = qrels_table.column("candidate_ids").to_pylist()
        qrel_cand_scores_col = qrels_table.column("candidate_scores").to_pylist()

        # Qrels are row-aligned with queries; metric computation depends on that contract.
        if len(qrel_query_ids) != len(query_ids):
            raise ValueError(
                f"qrels count ({len(qrel_query_ids)}) != queries count ({len(query_ids)})"
            )
        for i, (qrel_qid, query_qid) in enumerate(zip(qrel_query_ids, query_ids, strict=True)):
            if str(qrel_qid) != str(query_qid):
                raise ValueError(
                    f"qrels[{i}].query_id={qrel_qid!r} != queries[{i}].id={query_qid!r}"
                )

        per_query_rel, max_n_rel = self._build_per_query_relevance(
            qrel_candidate_ids=qrel_cand_ids_col,
            qrel_candidate_scores=qrel_cand_scores_col,
            cand_id_to_idx=cand_id_to_idx,
            n_query=n_query,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        similarity_started_at = perf_counter()
        similarity = query_embeds @ cand_embeds.T  # [N_query, N_cand]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.debug(
            "%s: similarity matmul finished in %.2fs (shape=%s)",
            self.dataset.name,
            perf_counter() - similarity_started_at,
            tuple(similarity.shape),
        )

        # Sparse mode ranks against all candidates. Exhaustive mode ranks only
        # within the candidate subset listed in qrels for that query.
        has_exhaustive = any(
            (qrel_modes[i] or "sparse") == "exhaustive"
            for i in range(n_query)
        )
        topk_started_at = perf_counter()

        if not has_exhaustive:
            effective_k = min(max_k, n_cand)
            _, topk_indices = torch.topk(
                similarity, k=effective_k, dim=1, sorted=True
            )  # [N_query, effective_k]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            logger.debug(
                "%s: sparse topk finished in %.2fs (effective_k=%d, shape=%s)",
                self.dataset.name,
                perf_counter() - topk_started_at,
                effective_k,
                tuple(topk_indices.shape),
            )
            del similarity

            relevance = torch.zeros(n_query, effective_k, device=device)
            n_relevant = torch.zeros(n_query, device=device)
            all_rel_scores = torch.zeros(n_query, max_n_rel, device=device)

            for i in range(n_query):
                rel_map = per_query_rel[i]
                n_relevant[i] = len(rel_map)
                self._write_query_metric_inputs(
                    relevance=relevance,
                    all_rel_scores=all_rel_scores,
                    query_index=i,
                    ordered_candidate_indices=topk_indices[i],
                    rel_map=rel_map,
                )

        else:
            effective_k = max_k
            relevance = torch.zeros(n_query, effective_k, device=device)
            n_relevant = torch.zeros(n_query, device=device)
            all_rel_scores = torch.zeros(n_query, max_n_rel, device=device)

            for i in range(n_query):
                mode = qrel_modes[i] if qrel_modes[i] else "sparse"
                rel_map = per_query_rel[i]
                n_relevant[i] = len(rel_map)

                if mode == "exhaustive":
                    ordered_indices: torch.Tensor | list[int] = []
                    cand_id_list = qrel_cand_ids_col[i] or []
                    subset_indices = [
                        cand_id_to_idx[cid] for cid in cand_id_list if cid is not None
                    ]
                    if subset_indices:
                        subset_idx_t = torch.tensor(
                            subset_indices, device=device, dtype=torch.long
                        )
                        subset_sim = similarity[i, subset_idx_t]  # [subset_size]
                        sub_k = min(effective_k, len(subset_indices))
                        _, sub_topk = torch.topk(subset_sim, k=sub_k, sorted=True)
                        ordered_indices = [subset_indices[int(pos.item())] for pos in sub_topk]
                else:
                    row_k = min(effective_k, n_cand)
                    _, row_topk = torch.topk(similarity[i], k=row_k, sorted=True)
                    ordered_indices = row_topk

                self._write_query_metric_inputs(
                    relevance=relevance,
                    all_rel_scores=all_rel_scores,
                    query_index=i,
                    ordered_candidate_indices=ordered_indices,
                    rel_map=rel_map,
                )

            del similarity
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            logger.debug(
                "%s: mixed-mode topk finished in %.2fs (effective_k=%d)",
                self.dataset.name,
                perf_counter() - topk_started_at,
                effective_k,
            )

        if relevance.shape[1] < max_k:
            pad = torch.zeros(
                n_query, max_k - relevance.shape[1], device=device
            )
            relevance = torch.cat([relevance, pad], dim=1)

        logger.debug(
            "%s: prediction tensors ready relevance=%s n_relevant=%s all_rel_scores=%s",
            self.dataset.name,
            tuple(relevance.shape),
            tuple(n_relevant.shape),
            tuple(all_rel_scores.shape),
        )

        return {
            "relevance": relevance,      # [N_query, max_k]
            "n_relevant": n_relevant,     # [N_query]
            "all_rel_scores": all_rel_scores,  # [N_query, max_n_rel]
        }

    # ================================================================
    # Helper methods
    # ================================================================

    def _get_model_device(self, model: nn.Module) -> torch.device:
        """Get device from model."""
        if hasattr(model, "device"):
            return cast(torch.device, model.device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_embeddings(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """Get embeddings from model.

        Args:
            model: Model to use
            inputs: Model inputs

        Returns:
            Embeddings tensor [batch_size, D]
        """
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            if "embeddings" in outputs:
                embeddings = outputs["embeddings"]
            elif "sentence_embedding" in outputs:
                embeddings = outputs["sentence_embedding"]
            else:
                raise ValueError(
                    f"Dict output missing 'embeddings' key. Keys: {outputs.keys()}"
                )
        elif isinstance(outputs, torch.Tensor):
            embeddings = outputs
        else:
            raise ValueError(f"Unknown output format: {type(outputs)}")

        return embeddings.float()


__all__ = ["RetrievalEvaluator"]
