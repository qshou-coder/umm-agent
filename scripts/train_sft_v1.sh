export WANDB_API_KEY=0ca6e7d97bd6aea9de4b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

# W&B input logging controls
WANDB_LOG_INPUT_DATA=${WANDB_LOG_INPUT_DATA:-True}
WANDB_LOG_INPUT_EVERY=${WANDB_LOG_INPUT_EVERY:-100}
WANDB_INPUT_PREVIEW_TOKENS=${WANDB_INPUT_PREVIEW_TOKENS:-128}
WANDB_LOG_INPUT_IMAGES=${WANDB_LOG_INPUT_IMAGES:-True}
# -1 means log all reference/gen images in the batch
WANDB_MAX_LOGGED_IMAGES=${WANDB_MAX_LOGGED_IMAGES:--1}

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train/pretrain_unified_navit.py \
    --model_path /apdcephfs_zwfy2/share_303944931/shawncschen/models/Bagel-7B \
    --vit_path /apdcephfs_zwfy2/share_303944931/shawncschen/models/siglip-so400m-14-980-flash-attn2-navit \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --dataset_config_file ./data/configs/agent_data.yaml \
    --auto_resume True \
    --resume_model_only True \
    --finetune_from_hf True \
    --finetune_from_ema True \
    --results_dir ./sft/interleaved_v1 \
    --checkpoint_dir ./sft/interleaved_v1 \
    --num_workers 1 \
    --log_every 20 \
    --save_every 1000 \
    --num_shard 8 \
    --num_replicate $NNODES \
    --lr 5e-5 \
    --warmup_steps 500 \
    --expected_num_tokens 30240 \
    --max_num_tokens 31520 \
    --max_num_tokens_per_sample 30240 \
    --wandb_project bagel_agent-SFT \
    --wandb_name interleaved_v1_0212 \
    --wandb_offline True \
    --wandb_log_input_data ${WANDB_LOG_INPUT_DATA} \
    --wandb_log_input_every ${WANDB_LOG_INPUT_EVERY} \
    --wandb_input_preview_tokens ${WANDB_INPUT_PREVIEW_TOKENS} \
    --wandb_log_input_images ${WANDB_LOG_INPUT_IMAGES} \
    --wandb_max_logged_images ${WANDB_MAX_LOGGED_IMAGES}

# for debugging
# bash scripts/train_sft_v1.sh 1 0 28.49.32.176 29501