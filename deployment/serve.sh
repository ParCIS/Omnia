#!/usr/bin/env bash
# Launch one model process. Run coarse, fine and llm in separate terminals.
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^(vllm|sglang)$ || ! "$2" =~ ^(coarse|fine|llm)$ ]]; then
    echo "Usage: bash deployment/serve.sh {vllm|sglang} {coarse|fine|llm} [backend arguments...]" >&2
    exit 2
fi
omnia_backend=$1
omnia_role=$2
shift 2

case "$omnia_role" in
    coarse)
        omnia_default_model=Qwen/Qwen3-Reranker-0.6B
        omnia_default_port=9877
        omnia_default_gpus=0
        omnia_default_tp=1
        omnia_default_length=2048
        ;;
    fine)
        omnia_default_model=Qwen/Qwen3-Reranker-8B
        omnia_default_port=9878
        omnia_default_gpus=1
        omnia_default_tp=1
        omnia_default_length=2048
        ;;
    llm)
        omnia_default_model=Qwen/Qwen3-32B
        omnia_default_port=9879
        omnia_default_gpus=2,3
        omnia_default_tp=2
        omnia_default_length=8192
        ;;
esac

omnia_model=${OMNIA_MODEL:-$omnia_default_model}
omnia_port=${OMNIA_PORT:-$omnia_default_port}
omnia_host=${OMNIA_HOST:-127.0.0.1}
omnia_tp=${OMNIA_TP_SIZE:-$omnia_default_tp}
omnia_length=${OMNIA_CONTEXT_LENGTH:-$omnia_default_length}
omnia_memory=${OMNIA_MEMORY_FRACTION:-0.9}
export CUDA_VISIBLE_DEVICES=${OMNIA_GPUS:-$omnia_default_gpus}

if [[ "$omnia_backend" == vllm ]]; then
    omnia_command=(python -m vllm.entrypoints.openai.api_server
        --model "$omnia_model" --served-model-name "${OMNIA_SERVED_MODEL:-$omnia_model}"
        --host "$omnia_host" --port "$omnia_port"
        --tensor-parallel-size "$omnia_tp" --max-model-len "$omnia_length"
        --gpu-memory-utilization "$omnia_memory" --runner generate --convert none
        --enable-prefix-caching)
    if [[ "$omnia_role" != llm ]]; then
        omnia_command+=(--hf-overrides '{"architectures":["Qwen3ForCausalLM"]}')
    fi
else
    # Radix prefix caching remains enabled by default; metrics are required by BARS.
    omnia_command=(python -m sglang.launch_server
        --model-path "$omnia_model" --served-model-name "${OMNIA_SERVED_MODEL:-$omnia_model}"
        --host "$omnia_host" --port "$omnia_port"
        --tp-size "$omnia_tp" --context-length "$omnia_length"
        --mem-fraction-static "$omnia_memory" --enable-metrics)
    if [[ "$omnia_role" != llm ]]; then
        omnia_command+=(--json-model-override-args '{"architectures":["Qwen3ForCausalLM"]}')
    fi
fi
omnia_command+=("$@")

if [[ ${OMNIA_DRY_RUN:-0} == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${omnia_command[@]}"
    printf '\n'
else
    exec "${omnia_command[@]}"
fi
