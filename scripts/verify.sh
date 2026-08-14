#!/usr/bin/env bash
# Verify a Qwen3.8-27B endpoint actually delivers what its launch flags promised.
# Usage: verify.sh [ENDPOINT]   (default http://localhost:8219)
set -euo pipefail
EP="${1:-http://localhost:8219}"

echo "== health =="
curl -sf --max-time 5 "$EP/v1/models" >/dev/null && echo "OK $EP" || { echo "FAIL: no /v1/models"; exit 1; }
MODEL=$(curl -s "$EP/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "model: $MODEL"

echo "== timed 400-token probe (thinking off) =="
python3 - "$EP" "$MODEL" <<'EOF'
import json,sys,time,urllib.request
ep,model=sys.argv[1],sys.argv[2]
body=json.dumps({"model":model,"messages":[{"role":"user","content":"Write a 300-word story about a lighthouse keeper."}],
                 "max_tokens":400,"temperature":0.7,"top_p":0.8,
                 "chat_template_kwargs":{"enable_thinking":False}}).encode()
t0=time.time()
r=json.load(urllib.request.urlopen(urllib.request.Request(f"{ep}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=600))
dt=time.time()-t0; u=r["usage"]
print(f"{u['completion_tokens']} tok in {dt:.1f}s = {u['completion_tokens']/dt:.1f} tok/s (incl TTFT)")
EOF

echo "== thinking round-trip (reasoning parser check) =="
python3 - "$EP" "$MODEL" <<'EOF'
import json,sys,urllib.request
ep,model=sys.argv[1],sys.argv[2]
body=json.dumps({"model":model,"messages":[{"role":"user","content":"What is 17*23?"}],
                 "max_tokens":2048,"chat_template_kwargs":{"reasoning_effort":"low"}}).encode()
r=json.load(urllib.request.urlopen(urllib.request.Request(f"{ep}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=600))
m=r["choices"][0]["message"]
rc=m.get("reasoning_content") or m.get("reasoning") or ""
inline="<think>" in (m.get("content") or "") or "</think>" in (m.get("content") or "")
if rc and not inline: print(f"OK: reasoning_content present ({len(rc)} chars), none leaked into content")
elif inline: print("WARN: think tags inline in content — launch is missing --reasoning-parser qwen3")
else: print("WARN: no reasoning returned — template kwargs may not be reaching the server")
EOF

echo "== MTP acceptance (vLLM only; needs docker access on the serving host) =="
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q qwen38; then
  docker logs "$(docker ps --format '{{.Names}}' | grep qwen38 | head -1)" 2>&1 \
    | grep "SpecDecoding metrics" | tail -2 || echo "WARN: no SpecDecoding metrics — speculative config did not take"
else
  echo "skip (run on the serving host to see acceptance metrics)"
fi
