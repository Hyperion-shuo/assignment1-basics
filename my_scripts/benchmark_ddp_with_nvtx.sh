#!/bin/bash

# Benchmark DDP communication strategies with NVTX + Nsight Systems profiling
# Medium config: d_model=1024, d_ff=4096, num_layers=24, num_heads=16
# Usage: bash my_scripts/benchmark_ddp_with_nvtx.sh

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="${SCRIPT_DIR}/nsys_results"
mkdir -p "${RESULT_DIR}"

# Shared training args (medium config, reduced iters for profiling)
COMMON_ARGS=(
    --data_set_name tinystories
    --vocab_size 10000
    --context_length 256
    --d_model 1024
    --num_layers 24
    --num_heads 16
    --d_ff 4096
    --batch_size 8
    --max_grad_norm 1.0
    --alpha_max 1e-3
    --alpha_min 1e-4
    --t_w 1000
    --t_c 10000
    --betas 0.9 0.999
    --adamw_eps 1e-8
    --weight_decay 0.01
    --num_epoch 1
    --max_iter 30
    --save_num_iter 99999
    --log_num_iter 10
    --generate_num_iter 99999
    --eval_num_iter 99999
    --eval_num_batches 1
    --world_size 4
    --use_mixed_precision
    --no_wandb
    --use_nvtx
)

echo "============================================"
echo "  1/2  Profiling: Naive DDP (baseline)"
echo "============================================"
uv run nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o "${RESULT_DIR}/ddp_naive" \
    --force-overwrite true \
    python -m my_scripts.ddp_trainer \
    "${COMMON_ARGS[@]}"

echo ""
echo "============================================"
echo "  2/2  Profiling: Overlap comm (per-param async)"
echo "============================================"
uv run nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o "${RESULT_DIR}/ddp_overlap" \
    --force-overwrite true \
    python -m my_scripts.ddp_trainer \
    "${COMMON_ARGS[@]}" \
    --use_overlap_comm

echo ""
echo "============================================"
echo "  Done! Results saved to:"
echo "    ${RESULT_DIR}/ddp_naive.nsys-rep"
echo "    ${RESULT_DIR}/ddp_overlap.nsys-rep"
echo "  Open with: nsys-ui <file>.nsys-rep"
echo "============================================"
