"""The original lightweight BARS heuristic, sampled once before fine reranking."""

import asyncio
import logging
import math
import re
from contextlib import contextmanager

import aiohttp

LOGGER = logging.getLogger(__name__)
SAMPLE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+(\S+)')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_kv_usage(text):
    """Read a cache fraction from vLLM or SGLang Prometheus metrics."""
    metrics = {}
    for line in text.splitlines():
        match = SAMPLE.match(line.strip())
        if match is None or line.lstrip().startswith("#"):
            continue
        name, labels, raw = match.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            key = tuple(sorted(LABEL.findall(labels or "")))
            metrics.setdefault(name, {})[key] = value
    for name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"):
        values = list(metrics.get(name, {}).values())
        if values:
            return max(0.0, min(1.0, max(values)))
    used = metrics.get("sglang:num_used_tokens", {})
    totals = metrics.get("sglang:max_total_num_tokens", {})
    ratios = [value / totals[key] for key, value in used.items()
              if key in totals and totals[key] > 0]
    if ratios:
        return max(0.0, min(1.0, max(ratios)))
    raise ValueError("No supported KV cache metrics found")


class BarsScheduler:
    def __init__(self, session, metrics_url, config):
        self.session = session
        self.metrics_url = metrics_url
        self.config = config
        self.active_generations = 0

    @contextmanager
    def generation(self):
        # Like the original CLIENT_NUM, count final generation calls, including
        # backend queueing/prefill, rather than trying to identify exact decode phases.
        self.active_generations += 1
        try:
            yield
        finally:
            self.active_generations -= 1

    def group_size(self, usage):
        # Saturation must not cause division by zero.
        usage = max(0.0, min(1.0 - 1e-6, usage))
        return max(self.config.min_group_size, int(
            self.config.alpha * self.active_generations
            + self.config.beta * usage / (1.0 - usage)
        ))

    async def choose_group_size(self, fallback):
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.metrics_timeout_seconds)
            async with self.session.get(self.metrics_url, timeout=timeout) as response:
                response.raise_for_status()
                usage = parse_kv_usage(await response.text())
            return self.group_size(usage)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            LOGGER.warning("BARS metrics unavailable; using group_size=%s: %s", fallback, error)
            return fallback
