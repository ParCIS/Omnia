# Omnia

Omnia reduces RAG serving latency through **coarse-to-fine reranking, chunk pruning and progressive prefilling**. This repository packages the core experimental pipeline with the original **lightweight BARS heuristic**, HTTP adapters for vLLM and SGLang, a serial baseline, and a small corpus sample.

## Pipeline

```text
Candidate chunks -> coarse reranking -> reordered chunks -> grouped fine reranking
                                                               |
                                  +----------------------------+----------------+
                                  |                            |                |
                           stable Top-k?                speculative prefill     |
                           prune the tail               overlaps next group     |
                                  |                                             |
                                  +------------------> final Top-k -> answer generation
```

1. A small Qwen3-Reranker scores all candidates and determines the fine-reranking order.
2. The fine reranker processes an initial group of `max(top_k, ceil(initial_fraction * N))` candidates, capped at `N`. Subsequent groups use the size selected by BARS, or the fixed `pipeline.group_size` when BARS is disabled.
3. After the initial group establishes a reference ranking, unchanged ordered Top-k results increment a consecutive-stability counter. Any ranking change resets it. Once the count reaches `max(min_stable_candidates, ceil(stability_fraction * N))`, fine reranking stops at a group boundary. Every score in an already-computed group participates in selection.
4. After the first group, the system speculatively warms the current Top-1 prompt with a one-token generation request, then progressively expands toward Top-k while subsequent fine-reranking groups execute.
5. If the ranking changes, a corrected prompt lets the backend reuse its matching prefix and recompute the changed suffix. After fine reranking and any outstanding warmup finish, the final selected chunks are used to generate the answer.

There is at most one outstanding warmup per request. A group completed while a warmup is still running does not submit another warmup; a later group can use the updated ranking. Warmup output is discarded. Actual reuse depends on token-prefix matches, cache block boundaries and eviction.

The `baseline` mode skips coarse reranking, scores every candidate with the fine reranker and generates once. It performs no pruning or speculative prefilling.

## Repository layout

```text
ParCIS_Omnia/
├── omnia/
│   ├── bars.py               # Lightweight BARS and backend metrics parsing
│   ├── config.py             # Configuration and parameter validation
│   ├── pipeline.py           # Core pipeline and serial baseline
│   ├── clients.py            # vLLM/SGLang reranking and streaming generation
│   └── server.py             # POST /generate service
├── deployment/
│   ├── serve.sh              # Launch either backend in coarse/fine/llm roles
│   └── README.md             # Deployment, GPU placement and API requirements
├── data/
│   ├── frames_sample.json    # Two complete records, about 16 KiB
│   ├── manifest.json         # Source, selection rule and checksums
│   └── README.md             # Corpus collection, sizes and usage
├── tests/                    # Tests without GPU models or downloads
├── config.example.json
└── requirements.txt          # Orchestration dependencies only
```

## Quick start

Use Python 3.10+ for the orchestration service. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
```

Follow [the deployment guide](deployment/README.md) to install and launch the three model backends. Model engines can run in separate environments or on remote GPU machines. For example, after installing vLLM, run these in separate terminals:

```bash
bash deployment/serve.sh vllm coarse
bash deployment/serve.sh vllm fine
bash deployment/serve.sh vllm llm
```

For SGLang, replace `vllm` with `sglang` and set `backend` to `sglang` in both reranker configurations. Both backends use the same chat-completion interface for the answer generator. Backend-specific reranker adapters are included; see the deployment guide for their required logprob interfaces and official documentation.

Edit `config.json` to match the model names, URLs and tokenizer locations, then start the service:

```bash
python -m omnia.server --config config.json
```

Send retrieved chunks directly:

```bash
curl http://127.0.0.1:9876/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "example-1",
    "request": "What is the capital of France?",
    "top_k": 1,
    "mode": "omnia",
    "chunks": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      "Madrid is the capital of Spain.",
      "Rome is the capital of Italy."
    ]
  }'
```

`id` is optional. `top_k` defaults to 5 and is capped at the candidate count. Set `mode` to `baseline` for the serial comparison. The API returns a complete JSON response; streaming is used internally to measure the first generated token.

To use the included corpus, set `dataset_path` to `data/frames_sample.json` and omit `chunks` from the request. The question must exactly match a corpus key. The [data guide](data/README.md) includes a ready-to-run sample request and describes full-corpus paths.

## Lightweight BARS

The example configuration enables the heuristic from the original `OMNIA/omnia.py`:

```text
G = max(G_min, int(alpha * active_generations + beta * M / (1 - M)))
```

`M` is the generator's KV cache utilization as a fraction. Defaults are `G_min=2`, `alpha=0.5` and `beta=1.0`. Omnia queries the generator's `/metrics` **once after coarse reranking and before fine reranking**, then uses `G` for all groups after the initial group. The initial fraction is unchanged. Higher load generally produces larger groups and less frequent opportunities to submit warmups.

`active_generations` counts outstanding final generation calls in this Omnia process, including their backend queueing and prefilling, as the original `CLIENT_NUM` did. It excludes speculative warmups and is not an exact backend decoding count. Calls from other Omnia processes or external clients are not counted, though their cache usage may appear in backend metrics. Baseline calls through this process contribute to the count but do not use BARS to schedule reranking. The counter is decremented on success, failure and cancellation.

Supported metrics are `vllm:kv_cache_usage_perc`, the older `vllm:gpu_cache_usage_perc`, and SGLang's `sglang:token_usage` or the ratio of matching `sglang:num_used_tokens` and `sglang:max_total_num_tokens` series. If multiple utilization series are present, the largest is used. Metrics requests are asynchronous with a one-second default timeout. Missing or failed metrics produce a warning and fall back to `pipeline.group_size`; a saturated cache is clamped just below 1 to avoid division by zero.

Set `bars.enabled` to `false` to use fixed groups and avoid metric requests. Omitting the `bars` section also disables the heuristic for compatibility with minimal configurations. `bars.metrics_url` can override the default URL derived from the generator's base URL.

This restores the original simple scheduling rule. It does not implement per-group resampling, distributed client accounting or the complete scheduling design described in the paper.

## Corpus provenance

The main experimental corpora were built from **benchmark questions and their associated source URLs**, fetched with **Jina Reader**, cleaned with **Mistune**, then split into token windows using the **Qwen3-Reranker-4B tokenizer**. These datasets replay collected pages; the serving path does not perform live search. Separate exploratory scripts use Qwen3 query rewriting, DDGS search and LlamaIndex sentence splitting.

The seven full 512-token artifacts total about **276 MiB**, so they are not duplicated here. Two complete FRAMES records with source URLs and reference answers are included for smoke testing. See [data/README.md](data/README.md) for dataset-specific provenance, measured sizes, and the distinction between benchmark-associated pages and search-collected pages.

## Configuration

| Setting | Example/default | Meaning |
| --- | --- | --- |
| `pipeline.group_size` | 2 | Fixed group size, also used if BARS metrics fail |
| `pipeline.initial_fraction` | 0.2 | Initial fine-reranking fraction, covering at least Top-k |
| `pipeline.stability_fraction` | 0.1 | Required consecutive-stability fraction |
| `pipeline.min_stable_candidates` | 2 | Minimum consecutive-stability count |
| `pipeline.enable_pruning` | true | Enable early termination of fine reranking |
| `pipeline.enable_prefill` | true | Enable speculative one-token warmups |
| `pipeline.max_tokens` | 512 | Final generation token limit |
| `bars.enabled` | true in the example | Enable the lightweight BARS rule |
| `bars.min_group_size`, `bars.alpha`, `bars.beta` | 2, 0.5, 1.0 | BARS formula parameters |
| `bars.metrics_timeout_seconds` | 1.0 | Timeout for the generator metrics request |
| `coarse_reranker.backend`, `fine_reranker.backend` | `vllm` | Select `vllm` or `sglang` independently |
| `max_concurrency` in each reranker | 128 / 64 in the example | Shared HTTP concurrency limit for that backend |

The generator must have prefix caching enabled for warmups to save work. The question-first prompt keeps instructions and the question fixed while context grows. Inputs are not silently truncated; configure sufficient backend context length. Dataset paths are resolved relative to the configuration file. Tokenizer paths are passed directly to Transformers.

## Response fields

Responses include `response`, `selected_indices` in the original input order, fine-reranker `selected_scores`, `chunk_num`, `compute_chunk_num`, `pruned_chunk_num`, `prefill_count`, `rollback_count`, `fine_group_size` and `bars_enabled`. `compute_chunk_num` counts only fine-reranked candidates; coarse reranking still scores all candidates. `fine_group_size` is the requested group size before the final slice is clipped to the remaining candidates, and is null for baseline. `bars_enabled` indicates that the heuristic is enabled for this request, even if its metrics lookup falls back.

`rollback_count` counts changes to already-speculated chunk prefixes, including a correction needed by final generation. It does not count backend KV operations. This version uses strict ranking correction rather than the paper's tolerance-based rollback policy.

Timestamps are Unix seconds: `arrival_time`, `rrk_beg_time`, `coarse_end_time`, `rrk_end_time`, `gen_beg_time`, `gen_1st_time` and `gen_end_time`. `gen_beg_time` starts the final generation call, not the first warmup. `gen_1st_time` excludes role-only stream events and is null if no output token is observed. End-to-end TTFT is `gen_1st_time - arrival_time`; response latency is `gen_end_time - arrival_time`. `coarse_end_time` is null for baseline.

## Validation and scope

```bash
python -m unittest discover -s tests -v
python -m compileall -q omnia tests
bash -n deployment/serve.sh
OMNIA_DRY_RUN=1 bash deployment/serve.sh vllm coarse
OMNIA_DRY_RUN=1 bash deployment/serve.sh sglang llm
```

Tests cover score assembly, original-index mapping, consecutive stability, group boundaries, reranking/prefill overlap, correction, error cleanup, backend adapters, BARS and the HTTP handler. They require no GPU model or download. Deployment commands are checked with dry runs. Real GPU compatibility, accuracy and performance remain to be evaluated in the target environment; this repository does not claim to reproduce the paper's reported speedups from these tests.

This cleanup also fixes the original full-reranking path returning only the last group's scores and the stability counter failing to reset after a ranking change. It keeps one-token cache warmups and a lightweight scheduler, without direct KV manipulation or cross-request fairness guarantees. The original parent `OMNIA/` implementation and full datasets remain untouched.

## License

Copyright (c) 2026 Rongtian Fu.

This project's original code and documentation are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Please credit **Rongtian Fu**. See [LICENSE](LICENSE) for the full license text.

Third-party materials, including the source passages in the corpus sample,
retain their respective rights and licenses; see [corpus provenance](data/README.md).

## Citation

```bibtex
@inproceedings{fu2026omnia,
  title={Omnia: Efficient RAG Serving through Speculative Scheduling},
  author={Fu, Rongtian and Li, Shigang and Xu, Youxuan and Wu, Tong and Ma, Zhi and Shi, Jinliang},
  booktitle={Proceedings of the 2026 The 35th International Symposium on High-Performance Parallel and Distributed Computing},
  pages={165--178},
  year={2026}
}
```
