"""HTTP adapters for the Qwen rerankers and a chat-completion generator."""

import asyncio
import json
import math
import time


class QwenReranker:
    def __init__(self, session, config, tokenizer=None):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer", config["model"]))
        self.session = session
        self.tokenizer = tokenizer
        self.model = config["model"]
        self.backend = config.get("backend", "vllm")
        if self.backend not in ("vllm", "sglang"):
            raise ValueError("Reranker backend must be 'vllm' or 'sglang'")
        base_url = config["base_url"].rstrip("/")
        self.url = (base_url + "/completions" if self.backend == "vllm"
                    else base_url.removesuffix("/v1") + "/generate")
        concurrency = config.get("max_concurrency", 64)
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.semaphore = asyncio.Semaphore(concurrency)
        self.token_ids = []
        for word in ("yes", "no"):
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Reranker tokenizer must encode {word!r} as one token")
            self.token_ids.append(ids[0])

    def format_prompt(self, query, chunk):
        messages = [
            {"role": "system", "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'},
            {"role": "user", "content": f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n<Query>: {query}\n\n<Document>: {chunk}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    async def _score_one(self, query, chunk):
        async with self.semaphore:
            prompt = self.format_prompt(query, chunk)
            if self.backend == "sglang":
                payload = {
                    "text": prompt, "stream": False, "return_logprob": True,
                    "logprob_start_len": -1, "token_ids_logprob": self.token_ids,
                    "sampling_params": {"temperature": 1, "max_new_tokens": 1},
                }
            else:
                payload = {
                    "model": self.model, "prompt": prompt,
                    "temperature": 0, "max_tokens": 1, "logprobs": 2,
                    "logprob_token_ids": self.token_ids,
                    "allowed_token_ids": self.token_ids,
                }
            async with self.session.post(self.url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
            if self.backend == "sglang":
                entries = data["meta_info"]["output_token_ids_logprobs"][0]
                logprobs = {entry[1]: entry[0] for entry in entries}
                yes, no = [logprobs[token_id] for token_id in self.token_ids]
            else:
                logprobs = data["choices"][0]["logprobs"]["top_logprobs"][0]
                yes, no = logprobs["yes"], logprobs["no"]
            maximum = max(yes, no)
            positive, negative = math.exp(yes - maximum), math.exp(no - maximum)
            return positive / (positive + negative)

    async def score(self, query, chunks):
        tasks = [asyncio.create_task(self._score_one(query, c)) for c in chunks]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def build_messages(query, chunks):
    # Keep the instructions and question fixed, then append context. No chunk
    # count or timestamp may precede the context, or prefix reuse would break.
    context = "\n\n".join(chunks)
    return [{"role": "user", "content": (
        "Answer the question using the provided context. Be concise.\n\n"
        f"Question:\n{query}\n\nContext:\n{context}"
    )}]


class Generator:
    def __init__(self, session, config):
        self.session = session
        self.model = config["model"]
        self.url = config["base_url"].rstrip("/") + "/chat/completions"
        self.extra_body = config.get("extra_body", {})

    async def generate(self, query, chunks, max_tokens):
        payload = dict(self.extra_body)
        payload.update(model=self.model, messages=build_messages(query, chunks),
                       stream=True, temperature=0, max_tokens=max_tokens)
        begin = time.time()
        first = None
        content = []
        async with self.session.post(self.url, json=payload) as response:
            response.raise_for_status()
            # readline handles transport chunks that split an SSE JSON record.
            while raw := await response.content.readline():
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                event = line[5:].strip()
                if event == "[DONE]":
                    break
                data = json.loads(event)
                if "error" in data:
                    raise RuntimeError(f"Generator stream error: {data['error']}")
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if first is None and (text or reasoning):
                        first = time.time()
                    if text:
                        content.append(text)
        return {"content": "".join(content), "gen_beg_time": begin,
                "ttft_timestamp": first, "gen_end_time": time.time()}
