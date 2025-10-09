#!/bin/bash
set -e
lm-eval \
  --model vllm \
  --model_args "pretrained=google/gemma-3-270m-it,dtype=bfloat16,tensor_parallel_size=2,distributed_executor_backend=mp" \
  --tasks squadv2,triviaqa,nq_open,boolq,ag_news,sst2,hellaswag,arc_easy,piqa \
  --num_fewshot 0 \
  --gen_kwargs "temperature=0,do_sample=False,max_gen_toks=64" \
  --batch_size auto \
  --seed 42 \
  --output_path results \
 
pkill -f "vllm" || true
pkill -f "engine_core" || true
pkill -f "torchrun" || true
sleep 2
fuser -k /dev/nvidia* || true