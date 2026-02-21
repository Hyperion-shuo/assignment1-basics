#!/bin/bash

# Train TinyStories on 4 A10 GPUs with different learning rates
# Each GPU runs an independent training process

# bash nohup_scripts/train_tinystories_4gpu.sh to run
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAINER="${SCRIPT_DIR}/my_scripts/trainer.py"
LOG_DIR="${SCRIPT_DIR}/nohup_scripts/logs"
mkdir -p "${LOG_DIR}"

# Common hyperparameters
COMMON_ARGS="
    --data_set_name tinystories
    --vocab_size 10000
    --context_length 256
    --d_model 512
    --num_layers 4
    --num_heads 16
    --d_ff 1344
    --rope_theta 10000
    --batch_size 128
    --max_grad_norm 1.0
    --alpha_min 1e-5
    --t_w 1000
    --t_c 10000
    --betas 0.9 0.999
    --adamw_eps 1e-8
    --weight_decay 0.01
    --num_epoch 1
    --save_num_iter 500
    --log_num_iter 100
    --generate_num_iter 500
    --eval_num_iter 500
    --eval_num_batches 20
"

# Different learning rates for each GPU
LR_LIST=(1e-4 3e-4 1e-3 3e-3)
LR_MIN_LIST=(1e-5 3e-5 1e-4 3e-4)

for i in 0 1 2 3; do
    LR=${LR_LIST[$i]}
    LR_MIN=${LR_MIN_LIST[$i]}
    RUN_NAME="tinystories_lrmax${LR}_lrmin${LR_MIN}_gpu${i}"
    CKPT_DIR="${SCRIPT_DIR}/models/tinystories_lrmax${LR}_lrmin${LR_MIN}"

    echo "Launching GPU ${i} with alpha_max=${LR} ..."

    CUDA_VISIBLE_DEVICES=${i} nohup python "${TRAINER}" \
        ${COMMON_ARGS} \
        --alpha_max ${LR} \
        --alpha_min ${LR_MIN} \
        --wandb_run_name "${RUN_NAME}" \
        --checkpoint_dir "${CKPT_DIR}" \
        > "${LOG_DIR}/train_tinystories_gpu${i}_lrmax${LR}_lrmin${LR_MIN}.log" 2>&1 &

    echo "  PID=$!, log: ${LOG_DIR}/train_tinystories_gpu${i}_lrmax${LR}_lrmin${LR_MIN}.log"
done

echo ""
echo "All 4 training processes launched."
echo "Use 'tail -f ${LOG_DIR}/train_tinystories_gpu*.log' to monitor."
echo "Use 'nvidia-smi' to check GPU utilization."
