set -euo pipefail

export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=bond1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# W&B input logging controls
WANDB_LOG_INPUT_DATA=${WANDB_LOG_INPUT_DATA:-False}
WANDB_LOG_INPUT_EVERY=${WANDB_LOG_INPUT_EVERY:-100}
WANDB_INPUT_PREVIEW_TOKENS=${WANDB_INPUT_PREVIEW_TOKENS:-128}
WANDB_LOG_INPUT_IMAGES=${WANDB_LOG_INPUT_IMAGES:-False}
# -1 means log all reference/gen images in the batch
WANDB_MAX_LOGGED_IMAGES=${WANDB_MAX_LOGGED_IMAGES:--1}

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/train_sft_v1.sh <nnodes> <node_rank> <master_addr> <master_port>"
    exit 1
fi

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if ! [[ "${NNODES}" =~ ^[0-9]+$ && "${NODE_RANK}" =~ ^[0-9]+$ && "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
    echo "[launcher] invalid numeric args: nnodes=${NNODES} node_rank=${NODE_RANK} master_port=${MASTER_PORT}"
    exit 1
fi

if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "[launcher] node_rank out of range: node_rank=${NODE_RANK}, nnodes=${NNODES}"
    exit 1
fi

CHECK_PORT=$((MASTER_PORT + 17))
if (( CHECK_PORT > 65535 )); then
    echo "[launcher] preflight check port overflow: master_port=${MASTER_PORT}, check_port=${CHECK_PORT}"
    exit 1
fi

echo "[launcher] $(date '+%F %T') host=$(hostname) nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launcher] nproc_per_node=${NPROC_PER_NODE}"

NNODES="${NNODES}" NODE_RANK="${NODE_RANK}" MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" python - <<'PY'
import os
import socket
import time

nnodes = int(os.environ["NNODES"])
node_rank = int(os.environ["NODE_RANK"])
master_addr = os.environ["MASTER_ADDR"]
master_port = int(os.environ["MASTER_PORT"])

print(
    f"[netcheck] host={socket.gethostname()} rank={node_rank}/{nnodes} master={master_addr}:{master_port}",
    flush=True,
)

try:
    resolved = socket.gethostbyname(master_addr)
    print(f"[netcheck] resolve {master_addr} -> {resolved}", flush=True)
except Exception as e:
    print(f"[netcheck] FAIL resolve {master_addr}: {e}", flush=True)
    raise SystemExit(2)

if node_rank != 0:
    ok = False
    last_err = ""
    for _ in range(20):
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((master_addr, master_port))
            ok = True
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5)
        finally:
            try:
                s.close()
            except Exception:
                pass
    if ok:
        print(f"[netcheck] OK connect {master_addr}:{master_port}", flush=True)
    else:
        print(f"[netcheck] WARN cannot connect {master_addr}:{master_port}: {last_err}", flush=True)
else:
    print("[netcheck] rank0 skip connect probe.", flush=True)
PY

python scripts/dist_preflight_check.py \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --timeout-sec 60

torchrun \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
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
    --num_shard $NPROC_PER_NODE \
    --num_replicate $NNODES \
    --lr 5e-5 \
    --warmup_steps 500 \
    --expected_num_tokens 30240 \
    --max_num_tokens 31520 \
    --max_num_tokens_per_sample 30240 \
    --wandb_project bagel_agent-SFT \
    --wandb_name interleaved_v1_0213 \
    --wandb_runid 0 \
    --wandb_offline True \
    --wandb_log_input_data ${WANDB_LOG_INPUT_DATA} \
    --wandb_log_input_every ${WANDB_LOG_INPUT_EVERY} \
    --wandb_input_preview_tokens ${WANDB_INPUT_PREVIEW_TOKENS} \
    --wandb_log_input_images ${WANDB_LOG_INPUT_IMAGES} \
    --wandb_max_logged_images ${WANDB_MAX_LOGGED_IMAGES}

# for debugging
# bash scripts/train_sft_v1.sh 1 0 28.49.32.176 29501
# for 64 GPU testing
# bash scripts/train_sft_v1.sh 8 {num} 28.49.32.176 29501