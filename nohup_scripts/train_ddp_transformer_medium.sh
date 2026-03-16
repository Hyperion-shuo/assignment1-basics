#!/bin/bash

# DDP training with Transformer Medium config on TinyStories using 4 A10 GPUs
# Medium: d_model=1024, d_ff=4096, num_layers=24, num_heads=16

# bash nohup_scripts/train_ddp_transformer_large.sh

# ============ Experiment configs (uncomment ONE block) ============
# Experiment 1: baseline (no flatten)
# WANDB_RUN="ddp_baseline"
# EXTRA_ARGS=""

# Experiment 2: with flatten dense tensors
# WANDB_RUN="ddp_flatten"
# EXTRA_ARGS="--flatten_dense_tensors"

# Experiment 3: with overlap communication and computation (per-param async all-reduce)
# WANDB_RUN="ddp_overlap"
# EXTRA_ARGS="--use_overlap_comm"

# Experiment 4: with bucket-based gradient communication
# WANDB_RUN="ddp_bucket"
# EXTRA_ARGS="--use_bucket_comm --bucket_size_mb 25"

# Experiment 5: with bucket-based gradient communication
# WANDB_RUN="ddp_bucket_1MB"
# EXTRA_ARGS="--use_bucket_comm --bucket_size_mb 1"

# Experiment 6: with bucket-based gradient communication
WANDB_RUN="ddp_bucket_10MB"
EXTRA_ARGS="--use_bucket_comm --bucket_size_mb 10"

# Experiment 7: with bucket-based gradient communication
# WANDB_RUN="ddp_bucket_100MB"
# EXTRA_ARGS="--use_bucket_comm --bucket_size_mb 100"

# Experiment 8: with bucket-based gradient communication
# WANDB_RUN="ddp_bucket_1000MB"
# EXTRA_ARGS="--use_bucket_comm --bucket_size_mb 1000"
# ==================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/nohup_scripts/logs"
mkdir -p "${LOG_DIR}"

nohup uv run python -m my_scripts.ddp_trainer \
    --data_set_name tinystories \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --d_ff 4096 \
    --batch_size 8 \
    --max_grad_norm 1.0 \
    --alpha_max 1e-3 \
    --alpha_min 1e-4 \
    --t_w 1000 \
    --t_c 10000 \
    --betas 0.9 0.999 \
    --adamw_eps 1e-8 \
    --weight_decay 0.01 \
    --num_epoch 1 \
    --max_iter 2001 \
    --save_num_iter 5000 \
    --log_num_iter 100 \
    --generate_num_iter 2000 \
    --eval_num_iter 500 \
    --eval_num_batches 20 \
    --world_size 4 \
    --use_mixed_precision \
    --wandb_project_name ddp-medium-ablation \
    --wandb_run_name "${WANDB_RUN}" \
    ${EXTRA_ARGS} \
    > "${LOG_DIR}/ddp_transformer_medium.log" 2>&1 &

echo "Launched DDP Transformer Medium training (4 GPUs)"
echo "  PID=$!, log: ${LOG_DIR}/ddp_transformer_medium.log"
echo "Use 'tail -f ${LOG_DIR}/ddp_transformer_medium.log' to monitor."
