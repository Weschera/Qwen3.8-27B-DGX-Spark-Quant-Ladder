#!/usr/bin/env bash
# Qwen3.8-27B BF16 on vLLM — full-precision reference lane (~10 tok/s single-request w/ MTP3); use a quant for daily serving
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
docker rm -f qwen38-27b 2>/dev/null || true
docker run -d --name qwen38-27b --gpus all --ipc=host -p 8219:8000 \
  -v "$MODEL_DIR":/model:ro \
  vllm/vllm-openai:nightly \
  --model /model --served-model-name Qwen3.8-27B \
  --attention-backend triton_attn \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gpu-memory-utilization 0.80 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --trust-remote-code
echo "launching on :8219 — verify with scripts/verify.sh"
