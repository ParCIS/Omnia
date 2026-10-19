import asyncio
import unittest

from omnia.config import PipelineConfig
from omnia.pipeline import OmniaPipeline, RankingState


class Reranker:
    def __init__(self, scores):
        self.scores = scores
        self.batches = []

    async def score(self, query, chunks):
        self.batches.append(list(chunks))
        await asyncio.sleep(0)
        return [self.scores[c] for c in chunks]


class Generator:
    def __init__(self):
        self.calls = []

    async def generate(self, query, chunks, n):
        self.calls.append((list(chunks), n))
        return dict(content="answer", gen_beg_time=1, ttft_timestamp=2, gen_end_time=3)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_scoring_keeps_earlier_groups_and_original_indices(self):
        coarse = Reranker(dict(a=1, b=3, c=2, d=0))
        fine = Reranker(dict(a=0.8, b=0.9, c=0.1, d=0.2))
        gen = Generator()
        pipeline = OmniaPipeline(coarse, fine, gen, PipelineConfig(
            enable_pruning=False, enable_prefill=False, group_size=1))
        result = await pipeline.run("q", list("abcd"), 2)
        self.assertEqual(fine.batches, [["b", "c"], ["a"], ["d"]])
        self.assertEqual(result["selected_indices"], [1, 0])
        self.assertEqual(result["selected_scores"], [0.9, 0.8])
        self.assertEqual(result["compute_chunk_num"], 4)
        self.assertEqual(gen.calls, [(["b", "a"], 512)])

    async def test_pruning_stops_tail_after_consecutive_stability(self):
        chunks = list("abcdefghij")
        rrk = Reranker({c: 10 - i for i, c in enumerate(chunks)})
        result = await OmniaPipeline(rrk, rrk, Generator()).run("q", chunks, 2)
        self.assertEqual(result["compute_chunk_num"], 4)
        self.assertEqual(result["pruned_chunk_num"], 6)
        self.assertEqual(result["selected_indices"], [0, 1])

    async def test_group_results_are_all_consumed_before_pruning(self):
        chunks = list("abcdefg")
        coarse = Reranker({c: 10 - i for i, c in enumerate(chunks)})
        fine = Reranker(dict(a=0.8, b=0.7, c=0.1, d=0.2, e=0.99, f=0.1, g=0.1))
        pipeline = OmniaPipeline(coarse, fine, Generator(), PipelineConfig(group_size=3))
        result = await pipeline.run("q", chunks, 2)
        # c,d would satisfy stability, but e in the same computed group replaces a.
        self.assertEqual(result["selected_indices"], [4, 0])
        self.assertEqual(result["compute_chunk_num"], 7)

    async def test_baseline_skips_coarse_and_prefill(self):
        coarse, fine, gen = Reranker({}), Reranker(dict(a=0.2, b=0.8)), Generator()
        result = await OmniaPipeline(coarse, fine, gen).run("q", ["a", "b"], 1, "baseline")
        self.assertEqual(coarse.batches, [])
        self.assertEqual(gen.calls, [(["b"], 512)])
        self.assertEqual(result["prefill_count"], 0)

    async def test_small_input_clamps_top_k(self):
        rrk = Reranker(dict(a=1))
        result = await OmniaPipeline(rrk, rrk, Generator()).run("q", ["a"], 5)
        self.assertEqual(result["top_k"], 1)
        self.assertEqual(result["compute_chunk_num"], 1)
        self.assertEqual(result["prefill_count"], 0)

    async def test_prefill_overlaps_fine_and_final_prompt_corrects_prediction(self):
        started, fine_continued = asyncio.Event(), asyncio.Event()

        class OverlappingFine(Reranker):
            async def score(self, query, chunks):
                if self.batches:
                    await asyncio.wait_for(started.wait(), 1)
                    fine_continued.set()
                return await super().score(query, chunks)

        class SlowGenerator(Generator):
            async def generate(self, query, chunks, n):
                if n == 1:
                    started.set()
                    await asyncio.wait_for(fine_continued.wait(), 1)
                return await super().generate(query, chunks, n)

        coarse = Reranker(dict(a=4, b=3, c=2, d=1))
        fine = OverlappingFine(dict(a=0.8, b=0.7, c=0.99, d=0.1))
        gen = SlowGenerator()
        pipeline = OmniaPipeline(coarse, fine, gen, PipelineConfig(enable_pruning=False))
        result = await asyncio.wait_for(pipeline.run("q", list("abcd"), 2), 2)
        self.assertEqual(gen.calls, [(["a"], 1), (["c", "a"], 512)])
        self.assertEqual(result["rollback_count"], 1)

    async def test_fine_failure_cancels_inflight_prefill(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        class BrokenFine(Reranker):
            async def score(self, query, chunks):
                if self.batches:
                    await asyncio.wait_for(started.wait(), 1)
                    raise RuntimeError("backend down")
                return await super().score(query, chunks)

        class PendingGenerator(Generator):
            async def generate(self, *args):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        scores = dict(a=4, b=3, c=2, d=1)
        pipeline = OmniaPipeline(Reranker(scores), BrokenFine(scores), PendingGenerator())
        with self.assertRaisesRegex(RuntimeError, "backend down"):
            await asyncio.wait_for(pipeline.run("q", list("abcd"), 2), 2)
        self.assertTrue(cancelled.is_set())

    async def test_prefill_failure_propagates(self):
        class BrokenGenerator(Generator):
            async def generate(self, *args):
                raise RuntimeError("prefill failed")

        rrk = Reranker(dict(a=4, b=3, c=2, d=1))
        with self.assertRaisesRegex(RuntimeError, "prefill failed"):
            await OmniaPipeline(rrk, rrk, BrokenGenerator()).run("q", list("abcd"), 2)

    async def test_invalid_inputs(self):
        pipeline = OmniaPipeline(None, None, None)
        for query, chunks, k, mode in [
            ("", ["a"], 1, "omnia"), ("q", [], 1, "omnia"),
            ("q", [""], 1, "omnia"), ("q", ["a"], 0, "omnia"),
            ("q", ["a"], True, "omnia"), ("q", ["a"], 1, "unknown"),
        ]:
            with self.assertRaises(ValueError):
                await pipeline.run(query, chunks, k, mode)

    async def test_incomplete_scores_rejected(self):
        class IncompleteReranker:
            async def score(self, *args):
                return []

        with self.assertRaisesRegex(ValueError, "incomplete"):
            await OmniaPipeline(None, IncompleteReranker(), Generator()).run(
                "q", ["a"], 1, "baseline")


class StabilityTests(unittest.TestCase):
    def test_change_resets_stability(self):
        state = RankingState(1)
        state.extend([0.8], track_stability=False)
        state.extend([0.1, 0.2])
        self.assertEqual(state.stable_candidates, 2)
        state.extend([0.9])
        self.assertEqual(state.stable_candidates, 0)
        state.extend([0.3])
        self.assertEqual(state.stable_candidates, 1)

    def test_invalid_configuration(self):
        for values in [dict(group_size=0), dict(initial_fraction=0),
                       dict(stability_fraction=2), dict(enable_pruning="false")]:
            with self.assertRaises(ValueError):
                PipelineConfig(**values)
