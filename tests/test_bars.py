import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from omnia.bars import BarsScheduler, parse_kv_usage
from omnia.config import BarsConfig, PipelineConfig
from omnia.pipeline import OmniaPipeline


class MetricsTests(unittest.TestCase):
    def test_vllm_metrics_select_busiest_engine(self):
        text = '''# HELP vllm:kv_cache_usage_perc KV cache utilization
vllm:kv_cache_usage_perc{model_name="qwen",engine="0"} 0.25
vllm:kv_cache_usage_perc{model_name="qwen",engine="1"} 0.75 1000
'''
        self.assertEqual(parse_kv_usage(text), 0.75)

    def test_old_vllm_metric(self):
        self.assertEqual(parse_kv_usage('vllm:gpu_cache_usage_perc 0.4'), 0.4)

    def test_sglang_ratio_matches_labels_regardless_of_order(self):
        text = '''sglang:num_used_tokens{engine_type="unified",rank="0"} 60
sglang:max_total_num_tokens{rank="0",engine_type="unified"} 100
sglang:num_used_tokens{engine_type="unified",rank="1"} 10
sglang:max_total_num_tokens{engine_type="unified",rank="1"} 100
'''
        self.assertEqual(parse_kv_usage(text), 0.6)

    def test_sglang_utilization_fraction(self):
        self.assertEqual(parse_kv_usage('sglang:token_usage{engine_type="unified"} 0.8'), 0.8)

    def test_invalid_metrics_are_not_treated_as_idle(self):
        for text in ('unrelated_metric 1', 'vllm:kv_cache_usage_perc NaN',
                     'sglang:num_used_tokens 5\nsglang:max_total_num_tokens 0'):
            with self.assertRaises(ValueError):
                parse_kv_usage(text)

    def test_original_formula_and_saturation(self):
        scheduler = BarsScheduler(None, "unused", BarsConfig(enabled=True))
        self.assertEqual(scheduler.group_size(0), 2)
        scheduler.active_generations = 8
        self.assertEqual(scheduler.group_size(0.5), 5)
        self.assertGreater(scheduler.group_size(1), 1000)


class BarsTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_metrics_and_failure_fallback(self):
        response = AsyncMock()
        response.raise_for_status = Mock()
        response.text.return_value = 'vllm:kv_cache_usage_perc 0.75'
        response.__aenter__.return_value = response
        session = Mock()
        session.get.return_value = response
        scheduler = BarsScheduler(session, 'http://generator/metrics', BarsConfig(enabled=True))
        self.assertEqual(await scheduler.choose_group_size(2), 3)
        session.get.assert_called_once()
        response.text.side_effect = asyncio.TimeoutError()
        with self.assertLogs('omnia.bars', level='WARNING'):
            self.assertEqual(await scheduler.choose_group_size(7), 7)

    async def test_cancellation_releases_generation_count(self):
        scheduler = BarsScheduler(None, "unused", BarsConfig())
        started = asyncio.Event()

        async def generate():
            with scheduler.generation():
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(generate())
        await started.wait()
        self.assertEqual(scheduler.active_generations, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(scheduler.active_generations, 0)

    async def test_pipeline_samples_once_and_tracks_only_final_generation(self):
        scheduler = BarsScheduler(None, "unused", BarsConfig(enabled=True))
        scheduler.choose_group_size = AsyncMock(return_value=3)
        chunks = list("abcdefghij")
        scores = {chunk: len(chunks) - i for i, chunk in enumerate(chunks)}
        coarse, fine, generator = AsyncMock(), AsyncMock(), AsyncMock()
        coarse.score.return_value = list(scores.values())
        fine.score.side_effect = lambda query, batch: [scores[chunk] for chunk in batch]

        async def generate(query, batch, n):
            self.assertEqual(scheduler.active_generations, int(n != 1))
            return dict(content="answer", gen_beg_time=1, ttft_timestamp=2, gen_end_time=3)

        generator.generate.side_effect = generate
        pipeline = OmniaPipeline(coarse, fine, generator, PipelineConfig(enable_pruning=False), scheduler)
        result = await pipeline.run("q", chunks, 2)
        scheduler.choose_group_size.assert_awaited_once_with(2)
        self.assertEqual([len(call.args[1]) for call in fine.score.await_args_list], [2, 3, 3, 2])
        self.assertEqual(result["fine_group_size"], 3)
        self.assertTrue(result["bars_enabled"])
        self.assertEqual(scheduler.active_generations, 0)

    async def test_baseline_does_not_poll_and_failure_releases_counter(self):
        scheduler = BarsScheduler(None, "unused", BarsConfig(enabled=True))
        scheduler.choose_group_size = AsyncMock()
        fine, generator = AsyncMock(), AsyncMock()
        fine.score.return_value = [0.9]

        async def fail(*args):
            self.assertEqual(scheduler.active_generations, 1)
            raise RuntimeError("generation failed")

        generator.generate.side_effect = fail
        pipeline = OmniaPipeline(None, fine, generator, bars=scheduler)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await pipeline.run("q", ["a"], 1, "baseline")
        scheduler.choose_group_size.assert_not_awaited()
        self.assertEqual(scheduler.active_generations, 0)

    def test_invalid_configuration(self):
        for values in (dict(alpha=-1), dict(beta=float('nan')),
                       dict(metrics_timeout_seconds=0), dict(min_group_size=0)):
            with self.assertRaises(ValueError):
                BarsConfig(**values)
