"""Run with python -m omnia.server --config config.json."""

import argparse
import asyncio
import json
import logging
from uuid import uuid4

import aiohttp
from aiohttp import web

from .clients import Generator, QwenReranker
from .bars import BarsScheduler
from .config import BarsConfig, PipelineConfig, load_config
from .pipeline import OmniaPipeline

LOGGER = logging.getLogger(__name__)


def create_app(config):
    app = web.Application()
    corpus = {}
    if config.get("dataset_path"):
        with open(config["dataset_path"], encoding="utf-8") as source:
            corpus = json.load(source)

    pipeline = None

    async def lifecycle(app):
        nonlocal pipeline
        timeout = aiohttp.ClientTimeout(total=config.get("timeout_seconds", 600))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            bars_config = BarsConfig(**config.get("bars", {}))
            bars = None
            if bars_config.enabled:
                base_url = config["generator"]["base_url"].rstrip("/").removesuffix("/v1")
                bars = BarsScheduler(session, bars_config.metrics_url or base_url + "/metrics", bars_config)
            pipeline = OmniaPipeline(
                QwenReranker(session, config["coarse_reranker"]),
                QwenReranker(session, config["fine_reranker"]),
                Generator(session, config["generator"]),
                PipelineConfig(**config.get("pipeline", {})),
                bars=bars,
            )
            yield

    async def generate(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            query = body.get("request")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("request must be a non-empty string")
            chunks = body.get("chunks")
            if chunks is None:
                if query not in corpus:
                    raise ValueError("Provide chunks or a request present in dataset_path")
                chunks = corpus[query]["chunks"]
            top_k = body.get("top_k", 5)
            mode = body.get("mode", "omnia")
            # Validate inputs before dispatch so model failures remain server errors.
            if (type(top_k) is not int or top_k < 1
                    or mode not in ("omnia", "baseline")
                    or not isinstance(chunks, list) or not chunks
                    or any(not isinstance(c, str) or not c.strip() for c in chunks)):
                raise ValueError("Invalid top_k, mode or chunks")
        except (ValueError, TypeError, KeyError) as error:
            return web.json_response({"error": str(error)}, status=400)
        try:
            result = await pipeline.run(query, chunks, top_k, mode)
            result["id"] = body.get("id", str(uuid4()))
            return web.json_response(result)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError, RuntimeError):
            LOGGER.exception("RAG backend failed")
            return web.json_response({"error": "Model backend failed; see server log"}, status=502)

    app.cleanup_ctx.append(lifecycle)
    app.router.add_post("/generate", generate)
    return app


def main():
    parser = argparse.ArgumentParser(description="Omnia core service with optional lightweight BARS")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO)
    server = config.get("server", {})
    web.run_app(create_app(config), host=server.get("host", "127.0.0.1"),
                port=server.get("port", 9876), handler_cancellation=True)


if __name__ == "__main__":
    main()
