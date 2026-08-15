# Qwen3.8-27B on DGX Spark — the quant ladder recipe

Serve **Qwen3.8-27B** (hybrid Gated-DeltaNet VLM, 262K native ctx, native MTP head) on a
single DGX Spark (GB10, 128 GB unified memory) — at the quant that fits *your* use, with
speculative decoding on, thinking effort under control, and a verifier that proves what
you're actually getting.

All numbers below were measured on release day (2026-08-14) on this hardware, single
request, from stream artifacts — not from marketing tables.

## Pick your quant

Same prompt (a large one-shot creative coding task, thinking at the model's default
`xhigh` effort, no output cap), one attempt each, one Spark each:

| Quant | Engine | Spec decode | Decode tok/s (measured) | Disk | Notes |
|---|---|---|---:|---:|---|
| NVFP4 (unsloth) | vLLM nightly | MTP K3 | **23.7** | 22 G | best speed; fp8 KV capable |
| FP8 (official) | vLLM nightly | MTP K3 | 15.0 | 29 G | best quality-per-byte pedigree (Qwen's own calib) |
| UD-Q4_K_XL (unsloth) | llama.cpp ≥ b10430 | none yet | ~11.5 | 17 G | mainline llama.cpp can't drive the MTP head yet |
| bf16 (official) | vLLM nightly | MTP K3 | ~10 | 49 G | bandwidth-bound reference; use for parity checks, not serving |

Single-request decode on GB10 is memory-bandwidth-bound: smaller weights ≈ proportionally
faster, and the MTP head claws back the rest. Output quality on the creative task was
strong on every quant we rendered (artifacts in `examples/`).

> **Measurement honesty:** the rates above came from deliberately conservative
> first-light launches (util 0.60, default attention backend, bf16 KV, 131K window).
> The launch commands below are the *tuned* config — fp8 KV + triton should only help,
> but run `scripts/verify.sh` and trust your own number over ours.

## vLLM launch (NVFP4, the daily driver)

```bash
docker run -d --name qwen38-27b --gpus all --ipc=host -p 8219:8000 \
  -v $HOME/models/Qwen3.8-27B-NVFP4:/model:ro \
  vllm/vllm-openai:nightly \
  --model /model --served-model-name Qwen3.8-27B-NVFP4 \
  --quantization compressed-tensors \
  --attention-backend triton_attn \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gpu-memory-utilization 0.80 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --trust-remote-code
```

Swap the volume + `--served-model-name` for FP8/bf16; drop `--quantization` and
`--kv-cache-dtype` for bf16. See `scripts/`.

### Why each non-obvious flag

- **`--reasoning-parser qwen3`** — without it, the entire thinking trace arrives inline in
  `content` (the chat template emits no opening `<think>`, just a closing tag). Every
  client that splits on `reasoning_content` silently breaks. We learned this from the
  artifacts; don't skip it.
- **`--kv-cache-dtype fp8` + `--attention-backend triton_attn`** — fp8 KV roughly doubles
  KV token capacity, but on GB10 (sm_121) FlashAttention can't serve fp8 KV — triton can.
  Credit where due: the triton requirement was first documented in
  [MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000).
  Only 16 of 64 layers are full attention (rest are Gated DeltaNet with constant-size
  state), so KV is cheap here to begin with — 131K ctx at util 0.60 already fits 247K
  KV tokens.
- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder`** — without these the server
  never emits structured `tool_calls`, and every agentic client (and benchmark) silently
  degrades. Our own benchmark preflight caught this omission on the first run — that's
  why the verify step below exists.
- **`--speculative-config mtp`** — the checkpoint ships its own MTP head (`mtp.*`
  tensors); no draft model needed. See K-tuning below.
- **`--gpu-memory-utilization 0.80`** — unified memory: the GPU pool *is* system RAM.
  0.84 works headless on this model; leave margin if anything else runs on the box. A
  driver-level OOM on GB10 can wedge the whole node, not just the process — err low if
  unsure.
- **`--max-num-seqs 4`** — single-user box; admit a few streams, keep batch-1 latency.

### Flags that did nothing (so you don't have to test them)

`--enable-flashinfer-autotune` + `FLASHINFER_CUDA_ARCH_LIST=12.1a` (seen in other GB10
recipes): measured zero effect on `vllm/vllm-openai:nightly` ≥ mid-Aug 2026 — the nightly
already picks correct sm_121 kernels. Also mind your accounting when comparing decode
claims: TTFT-inclusive vs decode-only differs by ~3-5% on short runs, and code vs prose
differs by ~60% under MTP (acceptance-bound). State your basis.

## Atlas lane: the 30 tok/s protocol (verified 2×, 2026-08-15)

With the LUT-staging repair kernels (Atlas PR #519 lineage), a single GB10 Spark serves
this model **above 30 tok/s** server-side decode. Full build + deploy walkthrough:
[Atlas-DGX-Spark-Quickstart](https://github.com/Weschera/Atlas-DGX-Spark-Quickstart).

Measured (server's `usage."response_token/s"`, MinHeap code prompt, temp 0, 3 repeats,
two independent labs): effort-none **32.0 / 29.8 / 30.5** (ours) · 31.6 (maintainer
fingerprint, incl. one deterministic-twin 964-token run) · thinking default 24.1.
Matched protocol on vLLM: 28.2 / 23.1 — Atlas holds the single-stream decode crown;
vLLM keeps prefill (~2,000 vs ~530 tok/s), concurrency, and the 262K–1M window.

What costs tok/s vs the fast protocol: temp 0.6 (−~2), prose vs code (−~4 via MTP
acceptance), bf16-KV override (−~2), pre-#519 kernels (−~5). State temp + prompt class
next to any number you quote.

## Apple Silicon + the MTP sidecar: ~45 tok/s on an M4 Max

`mlx-community/Qwen3.8-27B-MTP-4bit` packages the checkpoint's MTP head as a 239 MB
draft model. With `mlx-vlm` (plain `mlx-lm` cannot load `qwen3_5_mtp`):

```bash
mlx_vlm.generate --model mlx-community/Qwen3.8-27B-4bit \
  --draft-model mlx-community/Qwen3.8-27B-MTP-4bit --draft-kind mtp --draft-block-size 3 ...
```

Measured on M4 Max 128 GB: **~45 tok/s** generation (55% draft acceptance, prose,
temp 0.7) vs 29.5 plain MLX — currently the fastest single-box decode we have measured
for this model on any hardware. (Single-session measurement; repeats pending.)

## MTP tuning (measured acceptance)

Per-position draft acceptance we observed at K3 during a long code generation:
position 1 ≈ 0.61–0.74, position 2 ≈ 0.24–0.35, position 3 ≈ 0.06–0.22; mean accepted
length ~2.0–2.7 (higher in code, lower in prose).

Position 3 rarely pays for itself outside heavy code. **K3 for coding workloads, K2 for
mixed/chat** is our working rule; A/B on your own traffic with the verifier below —
acceptance is workload-dependent.

## Thinking effort: the 75K-token surprise

Default effort is **`xhigh`** and this model *uses* it: our one-shot creative task
produced 75,291 tokens on NVFP4 (≈160K chars of reasoning before the first line of
output). Glorious for quality, brutal for latency. Control it per request:

```json
{"chat_template_kwargs": {"reasoning_effort": "medium"}}   // or "low"
{"chat_template_kwargs": {"enable_thinking": false}}        // off
```

Valid efforts are `xhigh` (default), `medium`, `low` — passing `high` 400s.
Recommended sampling (model card): thinking → temp 1.0, top-p 0.95, top-k 20;
non-thinking → temp 0.7, top-p 0.80, presence 1.5.

## llama.cpp lane (GGUF)

Needs **b10430 or newer** (qwen3_5 support landed release-day).

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 --jinja -c 131072 -fa on \
  --host 0.0.0.0 --port 8219
```

- Keep KV at default f16 — on GB10, flash-attention aborts on *quantized* KV
  (`fattn.cu` assert); f16 KV + `-fa on` is fine.
- Mainline can't drive the native MTP head yet — no speculation, so vLLM+NVFP4 is
  ~2× faster today. Revisit when MTP lands.
- Vision needs the `mmproj-F16.gguf` sidecar (`--mmproj`).

## 1M context (optional)

Native window is 262,144. For 1M, add static YaRN (factor 4) — per the model card this
can slightly degrade short-context quality, so keep a native-window service for daily use:

```bash
--max-model-len 1000000 \
--hf-overrides '{"text_config":{"max_position_embeddings":1000000,"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'
```

## Verify what you launched (don't trust the launch)

`scripts/verify.sh` hits the endpoint and prints: HTTP health, a timed 400-token
non-thinking probe (tok/s), and the server's own MTP acceptance metrics. If acceptance
metrics are absent, your speculative config didn't take. If `reasoning_content` is empty
on a thinking request, your reasoning parser isn't set. Two minutes, no guessing.

## Provenance

- Measured 2026-08-14 (release day) on 4× DGX Spark (GB10, 128 GB), one model per node.
- Engines: `vllm/vllm-openai:nightly` (post-f8d03e77), llama.cpp b10430.
- Rates are single-request decode from streamed artifacts (`response.sse` timing), not
  server-side averages.
- Prior art: [MiaAI-Lab's NVFP4 recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000)
  (fp8-KV/triton finding, YaRN overrides) — different scope; this recipe adds the
  measured quant ladder, MTP acceptance tuning, thinking-effort ops, the reasoning-parser
  pitfall, and the llama.cpp lane.
