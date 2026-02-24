#!/bin/bash

# Ablation study on TinyStories using 4 A10 GPUs
# GPU 0: baseline (normal settings)
# GPU 1: no RMSNorm
# GPU 2: post-norm
# GPU 3: no RoPE (NoPE)

# bash nohup_scripts/train_tinystories_ablation.sh to run
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
    --alpha_max 1e-3
    --alpha_min 1e-4
    --t_w 1000
    --t_c 10000
    --betas 0.9 0.999
    --adamw_eps 1e-8
    --weight_decay 0.01
    --num_epoch 1
    --save_num_iter 5000
    --log_num_iter 100
    --generate_num_iter 2000
    --eval_num_iter 500
    --eval_num_batches 20
    --wandb_project_name tinystories-ablation
"

# Ablation configs: (name, extra_args)
NAMES=("baseline" "no_rms_norm" "post_norm" "no_rope")
EXTRA_ARGS=(
    ""
    "--no-with_rms_norm"
    "--no-pre_norm"
    "--no-use_rope"
)

for i in 0 1 2 3; do
    NAME=${NAMES[$i]}
    EXTRA=${EXTRA_ARGS[$i]}
    RUN_NAME="ablation_${NAME}"
    CKPT_DIR="${SCRIPT_DIR}/models/ablation_${NAME}"

    echo "Launching GPU ${i}: ${NAME} ..."

    CUDA_VISIBLE_DEVICES=${i} nohup python "${TRAINER}" \
        ${COMMON_ARGS} \
        ${EXTRA} \
        --wandb_run_name "${RUN_NAME}" \
        --checkpoint_dir "${CKPT_DIR}" \
        > "${LOG_DIR}/ablation_${NAME}.log" 2>&1 &

    echo "  PID=$!, log: ${LOG_DIR}/ablation_${NAME}.log"
done

echo ""
echo "All 4 ablation experiments launched."
echo "Use 'tail -f ${LOG_DIR}/ablation_*.log' to monitor."
echo "Use 'nvidia-smi' to check GPU utilization."
