import asyncio
import json
import unittest
from unittest.mock import patch

from omnia.clients import Generator, QwenReranker, build_messages


class Response:
    def __init__(self, data=None, lines=()):
        self.data = data
        self.lines = iter(lines)
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return self.data

    async def readline(self):
        return next(self.lines, b"")


class Session:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, url, json):
        self.requests.append((url, json))
        return self.response


class Tokenizer:
    def encode(self, word, **kwargs):
        return [1 if word == "yes" else 2]

    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\n" + messages[-1]["content"] + "<|im_end|>\n"


def event(delta):
    return ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n").encode()


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_token_ignores_role_and_usage_events(self):
        session = Session(Response(lines=[
            event({"role": "assistant"}), event({"content": "hello"}),
            event({"content": " world"}), b'data: {"choices": [], "usage": {}}\n',
            b"data: [DONE]\n",
        ]))
        generator = Generator(session, dict(base_url="http://model/v1", model="test"))
        with patch("omnia.clients.time.time", side_effect=[10, 20, 30]):
            result = await generator.generate("q", ["a"], 20)
        self.assertEqual(result, dict(content="hello world", gen_beg_time=10,
                                      ttft_timestamp=20, gen_end_time=30))

    async def test_reranker_stable_softmax_and_prompt(self):
        session = Session(Response(data={"choices": [{"logprobs": {
            "top_logprobs": [{"yes": -1000, "no": -1001}]
        }}]}))
        client = QwenReranker(session, dict(base_url="http://model/v1", model="test"), Tokenizer())
        self.assertAlmostEqual((await client.score("q", ["a"]))[0], 0.7310585786)
        prompt = session.requests[0][1]["prompt"]
        self.assertEqual(session.requests[0][1]["logprob_token_ids"], [1, 2])
        self.assertNotIn("<|im_end|>\n<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    async def test_missing_score_is_not_silently_defaulted(self):
        session = Session(Response(data={"choices": [{"logprobs": {
            "top_logprobs": [{"yes": -1}]
        }}]}))
        client = QwenReranker(session, dict(base_url="http://model/v1", model="test"), Tokenizer())
        with self.assertRaises(KeyError):
            await client.score("q", ["a"])

    async def test_sglang_selected_logprobs_are_matched_by_token_id(self):
        session = Session(Response(data={"meta_info": {"output_token_ids_logprobs": [
            [[-1001, 2, None], [-1000, 1, None]]
        ]}}))
        client = QwenReranker(session, dict(
            base_url="http://model/v1/", model="test", backend="sglang"), Tokenizer())
        self.assertAlmostEqual((await client.score("q", ["a"]))[0], 0.7310585786)
        url, payload = session.requests[0]
        self.assertEqual(url, "http://model/generate")
        self.assertEqual(payload["token_ids_logprob"], [1, 2])
        self.assertEqual(payload["sampling_params"]["temperature"], 1)
        self.assertNotIn("allowed_token_ids", payload)

    async def test_sglang_missing_label_fails(self):
        session = Session(Response(data={"meta_info": {"output_token_ids_logprobs": [
            [[-1, 1, None]]
        ]}}))
        client = QwenReranker(session, dict(
            base_url="http://model/v1", model="test", backend="sglang"), Tokenizer())
        with self.assertRaises(KeyError):
            await client.score("q", ["a"])

    async def test_stream_error_propagates(self):
        session = Session(Response(lines=[b'data: {"error": "failed"}\n']))
        client = Generator(session, dict(base_url="http://model/v1", model="test"))
        with self.assertRaises(RuntimeError):
            await client.generate("q", ["a"], 1)

    def test_appending_context_preserves_text_prefix(self):
        first = build_messages("q", ["a"])[0]["content"]
        second = build_messages("q", ["a", "b"])[0]["content"]
        self.assertTrue(second.startswith(first))


if __name__ == "__main__":
    unittest.main()
