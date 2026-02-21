#!/bin/bash

# Train TinyStories - single experiment with nohup
# Usage:
#   bash nohup_scripts/train_tinystories_single.sh [GPU_ID] [LEARNING_RATE]
#   bash nohup_scripts/train_tinystories_single.sh 0 3e-4

GPU_ID=${1:-0}
LR=${2:-3e-4}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAINER="${SCRIPT_DIR}/my_scripts/trainer.py"
LOG_DIR="${SCRIPT_DIR}/nohup_scripts/logs"
mkdir -p "${LOG_DIR}"

RUN_NAME="tinystories_lr${LR}_gpu${GPU_ID}"
CKPT_DIR="${SCRIPT_DIR}/models/tinystories_lr${LR}"
LOG_FILE="${LOG_DIR}/train_tinystories_gpu${GPU_ID}_lr${LR}.log"

echo "Launching training on GPU ${GPU_ID} with alpha_max=${LR} ..."
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "  Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python "${TRAINER}" \
    --data_set_name tinystories \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000 \
    --batch_size 64 \
    --max_grad_norm 1.0 \
    --alpha_max ${LR} \
    --alpha_min 1e-5 \
    --t_w 1000 \
    --t_c 10000 \
    --betas 0.9 0.999 \
    --adamw_eps 1e-8 \
    --weight_decay 0.01 \
    --num_epoch 1 \
    --save_num_iter 500 \
    --log_num_iter 100 \
    --generate_num_iter 500 \
    --eval_num_iter 500 \
    --eval_num_batches 20 \
    --wandb_run_name "${RUN_NAME}" \
    --checkpoint_dir "${CKPT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

echo "  PID=$!, use 'tail -f ${LOG_FILE}' to monitor."
