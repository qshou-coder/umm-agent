#!/usr/bin/env python3
import argparse
import os
import socket
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed smoke test worker.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "nccl", "gloo"],
        help="Distributed backend to use.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Process group init timeout in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    host = socket.gethostname()

    if args.backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    else:
        backend = args.backend

    device = torch.device("cpu")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=nccl but CUDA is not available.")
        cuda_count = torch.cuda.device_count()
        if cuda_count <= 0:
            raise RuntimeError("backend=nccl but cuda device count is 0.")
        torch.cuda.set_device(local_rank % cuda_count)
        device = torch.device("cuda", local_rank % cuda_count)

    print(
        f"[smoke] host={host} rank={rank}/{world_size} local_rank={local_rank} "
        f"backend={backend} device={device}",
        flush=True,
    )

    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=args.timeout_sec),
    )
    print(f"[smoke] rank={rank} init_process_group done", flush=True)

    tensor = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = (world_size * (world_size + 1)) / 2
    ok = abs(tensor.item() - expected) < 1e-5
    print(
        f"[smoke] rank={rank} all_reduce result={tensor.item():.1f} expected={expected:.1f} ok={ok}",
        flush=True,
    )
    if not ok:
        raise RuntimeError(
            f"all_reduce mismatch on rank={rank}: got {tensor.item()}, expected {expected}"
        )

    dist.barrier()
    if rank == 0:
        print("[smoke] PASS all ranks reached barrier", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[smoke] FAIL with exception:", flush=True)
        traceback.print_exc()
        sys.exit(2)

