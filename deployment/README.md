# Model deployment

Run the coarse reranker, fine reranker and answer generator as three separate model services. The orchestration service in `omnia/` communicates with them over HTTP. Either backend can serve each role, so mixed deployments are also supported.

## Environment

Install the model backend on the GPU machine in an environment separate from the lightweight orchestration dependencies. Use the backend's installation instructions for your CUDA, PyTorch and hardware combination: [vLLM installation](https://docs.vllm.ai/en/stable/getting_started/installation/) or [SGLang installation](https://docs.sglang.io/get_started/install.html). Do not assume `requirements.txt` installs either model engine.

The launchers target the documented generation interfaces and options below. Use a vLLM release that supports `logprob_token_ids` in `/v1/completions`, or an SGLang release that supports `token_ids_logprob` in `/generate`. Record the installed engine version with your experiment results. These launchers have been checked with dry runs, not executed against GPU models in the preparation environment.

All examples below run from the `ParCIS_Omnia` directory. Run each launch command in its own terminal and wait for that service to finish loading.

## vLLM

```bash
bash deployment/serve.sh vllm coarse
bash deployment/serve.sh vllm fine
bash deployment/serve.sh vllm llm
```

Use `"backend": "vllm"` in both reranker entries in `config.json`. This is the default in `config.example.json`.

The rerankers run as `Qwen3ForCausalLM` with the generation runner, rather than as sequence-classification/pooling models. Omnia builds the Qwen reranking prompt, requests the `yes` and `no` token log probabilities from `/v1/completions`, and normalizes those two values. Both token probabilities must be returned; the client fails if one is missing. See the [vLLM completion request protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/) for `logprob_token_ids`.

The launcher explicitly enables prefix caching and keeps `/metrics` available. These options and the generation runner are described in the [vLLM serve reference](https://docs.vllm.ai/en/stable/cli/serve/).

## SGLang

```bash
bash deployment/serve.sh sglang coarse
bash deployment/serve.sh sglang fine
bash deployment/serve.sh sglang llm
```

Change the two reranker entries to `"backend": "sglang"`. Keep their `base_url` values ending in `/v1`; the adapter derives the native `/generate` URL by removing that suffix. The LLM continues to use `/v1/chat/completions`, so no generator backend selector is required.

The SGLang adapter requests `token_ids_logprob` for the exact `yes` and `no` token IDs, takes their first-position log probabilities from `meta_info.output_token_ids_logprobs`, and computes the same two-label softmax. It discards the generated token text. The sampling temperature is 1 to preserve the probability ratio. This avoids depending on a small top-logprob list containing both labels. See [SGLang sampling parameters](https://docs.sglang.io/docs/basic_usage/sampling_params).

Rerankers are launched as causal language models, without embedding mode. For the LLM, keep Radix prefix caching enabled and enable metrics for BARS. The launcher passes `--enable-metrics` and does not pass `--disable-radix-cache`. See [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments).

## Defaults and overrides

| Role | Model | Port | GPU IDs | Tensor parallelism | Context length |
| --- | --- | --- | --- | --- | --- |
| `coarse` | `Qwen/Qwen3-Reranker-0.6B` | 9877 | `0` | 1 | 2048 |
| `fine` | `Qwen/Qwen3-Reranker-8B` | 9878 | `1` | 1 | 2048 |
| `llm` | `Qwen/Qwen3-32B` | 9879 | `2,3` | 2 | 8192 |

These are placement examples, not minimum hardware requirements. Adjust GPU IDs, tensor parallelism, context length and memory fraction to fit your hardware. The reranker context must include the query, one chunk and the instruction template; the LLM context must fit all selected chunks, the question and the answer budget.

```bash
OMNIA_GPUS=4,5 OMNIA_TP_SIZE=2 \
OMNIA_MODEL=/models/Qwen3-32B OMNIA_SERVED_MODEL=Qwen/Qwen3-32B \
bash deployment/serve.sh sglang llm
```

Supported environment overrides:

| Variable | Purpose |
| --- | --- |
| `OMNIA_MODEL` | Model ID or local weights directory |
| `OMNIA_SERVED_MODEL` | Name exposed by the model service; defaults to the model path/ID |
| `OMNIA_GPUS` | Value assigned to `CUDA_VISIBLE_DEVICES` |
| `OMNIA_TP_SIZE` | Tensor-parallel degree |
| `OMNIA_HOST`, `OMNIA_PORT` | Bind address and port; default host is `127.0.0.1` |
| `OMNIA_CONTEXT_LENGTH` | Maximum model context length |
| `OMNIA_MEMORY_FRACTION` | Backend memory fraction; default 0.9 |
| `OMNIA_DRY_RUN` | Set to `1` to print the command without launching a model |

Additional backend arguments can be appended after the role. Launchers use argument arrays and do not evaluate shell text. Model paths containing spaces remain one argument.

For remote model servers, set `OMNIA_HOST` to a reachable bind address and update the three `base_url` values in `config.json`. The reranker `model` must match the served model name; set its separate `tokenizer` field to the matching local tokenizer path or Hugging Face model ID if the served name is an alias. Update `bars.metrics_url` to point to the generator's `/metrics` endpoint.

## Verify and start Omnia

```bash
curl -f http://127.0.0.1:9877/v1/models
curl -f http://127.0.0.1:9878/v1/models
curl -f http://127.0.0.1:9879/v1/models
curl -f http://127.0.0.1:9879/metrics
python -m omnia.server --config config.json
```

The example generator configuration disables Qwen3 thinking through `chat_template_kwargs.enable_thinking=false`. Adjust `generator.extra_body` when switching model families. The generation client shares the same prompt construction for warmups and final generation, allowing the backend to reuse matching token prefixes.

Model-list checks verify reachability, not scoring correctness. Follow them with the `/generate` example in the main README. If scoring reports missing probabilities, check that the reranker is running in generation mode and that the installed backend supports the selected-token logprob field described above.
