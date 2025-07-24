#!/bin/bash

# (Optional: add this if you want to force FlashInfer)
# export VLLM_ATTENTION_BACKEND=FLASHINFER

python3 -m vllm.entrypoints.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --port 8000
