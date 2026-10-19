import json
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from omnia.server import create_app


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.coarse = AsyncMock()
        self.fine = AsyncMock()
        self.coarse.score.return_value = [0.9, 0.1]
        self.fine.score.return_value = [0.9, 0.1]
        self.generator = AsyncMock()
        self.generator.generate.return_value = dict(
            content="Paris", gen_beg_time=1, ttft_timestamp=2, gen_end_time=3)
        app = create_app(dict(coarse_reranker={}, fine_reranker={}, generator={}))
        self.runner = web.AppRunner(app)
        # Exercise startup/cleanup with a real aiohttp session, but no model
        # download, network listener or GPU service.
        with patch("omnia.server.QwenReranker", side_effect=[self.coarse, self.fine]), \
                patch("omnia.server.Generator", return_value=self.generator):
            await self.runner.setup()
        self.handler = next(iter(app.router.routes())).handler
        self.addAsyncCleanup(self.runner.cleanup)

    async def call(self, body):
        request = make_mocked_request("POST", "/generate")
        request.json = AsyncMock(return_value=body)
        return await self.handler(request)

    async def test_generate_endpoint_both_modes(self):
        for mode in ("omnia", "baseline"):
            response = await self.call(dict(
                id="test", request="capital?", chunks=["Paris", "Berlin"], top_k=2, mode=mode))
            self.assertEqual(response.status, 200)
            body = json.loads(response.text)
            self.assertEqual(body["id"], "test")
            self.assertEqual(body["response"], "Paris")
            self.assertEqual(body["selected_indices"], [0, 1])
        self.assertEqual(self.coarse.score.await_count, 1)

    async def test_invalid_request_is_400(self):
        for body in [[], {}, dict(request="unknown"), dict(request="q", chunks=[])]:
            self.assertEqual((await self.call(body)).status, 400)
        self.coarse.score.assert_not_awaited()

    async def test_backend_failure_is_502(self):
        self.fine.score.side_effect = RuntimeError("backend down")
        with self.assertLogs("omnia.server", level="ERROR"):
            response = await self.call(dict(
                request="q", chunks=["a", "b"], mode="baseline"))
        self.assertEqual(response.status, 502)
