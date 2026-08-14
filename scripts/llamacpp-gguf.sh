#!/usr/bin/env bash
# Qwen3.8-27B UD-Q4_K_XL on llama.cpp (b10430+) — no MTP in mainline yet (~11.5 tok/s)
set -euo pipefail
GGUF="${GGUF:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
pkill -x llama-server 2>/dev/null || true
nohup llama-server -m "$GGUF" -ngl 99 --jinja -c 131072 -fa on \
  --host 0.0.0.0 --port 8219 > llama-server.log 2>&1 &
echo "PID $! on :8219 — KV stays f16 (quantized KV + flash-attn aborts on GB10)"
