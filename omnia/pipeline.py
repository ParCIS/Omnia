"""Coarse-to-fine reranking, chunk pruning and progressive prefilling."""

import asyncio
import math
import time
from dataclasses import dataclass, field
from contextlib import nullcontext

from .config import PipelineConfig


@dataclass
class RankingState:
    top_k: int
    scores: list[float] = field(default_factory=list)
    stable_candidates: int = 0

    def top_indices(self):
        # Stable sorting makes ties deterministic in coarse-rank order.
        return sorted(range(len(self.scores)), key=self.scores.__getitem__, reverse=True)[:self.top_k]

    def extend(self, scores, *, track_stability=True):
        for score in scores:
            previous = self.top_indices()
            self.scores.append(score)
            current = self.top_indices()
            if track_stability and len(previous) == self.top_k and current == previous:
                self.stable_candidates += 1
            else:
                self.stable_candidates = 0


def checked_scores(scores, expected):
    if len(scores) != expected or any(not math.isfinite(s) for s in scores):
        raise ValueError("Reranker returned an incomplete or non-finite score list")
    return scores


class OmniaPipeline:
    """Model clients expose async score(query, chunks) / generate(query, chunks, n)."""

    def __init__(self, coarse, fine, generator, config=None, bars=None):
        self.coarse = coarse
        self.fine = fine
        self.generator = generator
        self.config = config or PipelineConfig()
        self.bars = bars

    async def run(self, query, chunks, top_k=5, mode="omnia"):
        if mode not in ("omnia", "baseline"):
            raise ValueError("mode must be 'omnia' or 'baseline'")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("request must be a non-empty string")
        if not isinstance(chunks, list) or not chunks or any(
            not isinstance(c, str) or not c.strip() for c in chunks
        ):
            raise ValueError("chunks must be a non-empty list of non-empty strings")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        arrival = time.time()
        top_k = min(top_k, len(chunks))
        rerank_begin = time.time()
        original_indices = list(range(len(chunks)))
        coarse_end = None
        prefill_count = 0
        rollback_count = 0
        previous_prefix = []
        prefill_task = None
        group_size = self.config.group_size

        try:
            if mode == "baseline":
                scores = checked_scores(await self.fine.score(query, chunks), len(chunks))
                state = RankingState(top_k, scores)
            else:
                coarse_scores = checked_scores(await self.coarse.score(query, chunks), len(chunks))
                original_indices.sort(key=coarse_scores.__getitem__, reverse=True)
                chunks = [chunks[i] for i in original_indices]
                coarse_end = time.time()
                if self.bars is not None:
                    group_size = await self.bars.choose_group_size(group_size)
                state = RankingState(top_k)
                first_size = min(len(chunks), max(top_k, math.ceil(self.config.initial_fraction * len(chunks))))
                stable_needed = max(self.config.min_stable_candidates,
                                    math.ceil(self.config.stability_fraction * len(chunks)))
                cursor = 0
                while cursor < len(chunks):
                    size = first_size if cursor == 0 else group_size
                    batch = chunks[cursor:cursor + size]
                    scores = checked_scores(await self.fine.score(query, batch), len(batch))
                    state.extend(scores, track_stability=cursor > 0)
                    cursor += len(batch)
                    if prefill_task is not None and prefill_task.done():
                        # Propagate errors; failed warmups must not disappear in background tasks.
                        prefill_task.result()
                    if cursor == len(chunks) or (
                        self.config.enable_pruning and state.stable_candidates >= stable_needed
                    ):
                        break
                    if self.config.enable_prefill and (prefill_task is None or prefill_task.done()):
                        count = min(top_k, len(previous_prefix) + 1)
                        prefix = state.top_indices()[:count]
                        if previous_prefix and prefix[:len(previous_prefix)] != previous_prefix:
                            rollback_count += 1
                        if prefix != previous_prefix:
                            # A new prompt reuses the backend's matching prefix; a changed
                            # suffix is recomputed. No direct KV cache mutation is required.
                            prefill_task = asyncio.create_task(self.generator.generate(
                                query, [chunks[i] for i in prefix], 1
                            ))
                            prefill_count += 1
                            previous_prefix = prefix
                        # Ensure the warmup is dispatched before the next fine group.
                        await asyncio.sleep(0)

            rerank_end = time.time()
            selected = state.top_indices()
            if previous_prefix and selected[:len(previous_prefix)] != previous_prefix:
                rollback_count += 1
            if prefill_task is not None:
                await prefill_task
            with self.bars.generation() if self.bars is not None else nullcontext():
                result = await self.generator.generate(
                    query, [chunks[i] for i in selected], self.config.max_tokens
                )
            return {
                "request": query,
                "mode": mode,
                "top_k": top_k,
                "arrival_time": arrival,
                "rrk_beg_time": rerank_begin,
                "coarse_end_time": coarse_end,
                "rrk_end_time": rerank_end,
                "gen_beg_time": result["gen_beg_time"],
                "gen_1st_time": result["ttft_timestamp"],
                "gen_end_time": result["gen_end_time"],
                "response": result["content"],
                "chunk_num": len(chunks),
                "compute_chunk_num": len(state.scores),
                "pruned_chunk_num": len(chunks) - len(state.scores),
                "selected_indices": [original_indices[i] for i in selected],
                "selected_scores": [state.scores[i] for i in selected],
                "prefill_count": prefill_count,
                "rollback_count": rollback_count,
                "fine_group_size": group_size if mode == "omnia" else None,
                "bars_enabled": self.bars is not None and mode == "omnia",
            }
        finally:
            if prefill_task is not None:
                if not prefill_task.done():
                    prefill_task.cancel()
                await asyncio.gather(prefill_task, return_exceptions=True)
