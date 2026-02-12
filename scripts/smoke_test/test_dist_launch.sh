#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash scripts/test_dist_launch.sh <nnodes> <node_rank> <master_addr> <master_port> [nproc_per_node]"
    exit 1
fi

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
NPROC_PER_NODE=${5:-8}

if ! [[ "${NNODES}" =~ ^[0-9]+$ && "${NODE_RANK}" =~ ^[0-9]+$ && "${MASTER_PORT}" =~ ^[0-9]+$ && "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
    echo "[test] invalid numeric args"
    exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "[test] node_rank out of range: node_rank=${NODE_RANK}, nnodes=${NNODES}"
    exit 1
fi
if (( NPROC_PER_NODE <= 0 )); then
    echo "[test] nproc_per_node must be > 0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "[test] $(date '+%F %T') host=$(hostname) nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

python scripts/dist_preflight_check.py \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --timeout-sec 30

torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/dist_smoke_worker.py \
    --backend auto \
    --timeout-sec 120

