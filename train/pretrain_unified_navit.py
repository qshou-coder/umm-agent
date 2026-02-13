# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import json
import os
import socket
import signal
import importlib
import sys
import threading
import traceback
from itertools import chain
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from time import time
from typing import Optional

print("[debug-train-import] stdlib imports done", flush=True)

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

print("[debug-train-import] torch/transformers imports done", flush=True)

_debug_trace = importlib.import_module("data.debug_trace")
_agent_log = _debug_trace.agent_log
_import_with_log = _debug_trace.import_with_log


_dataset_base = _import_with_log(
    "data.dataset_base",
    "H1",
    "train/pretrain_unified_navit.py:project-imports",
    "before import data.dataset_base",
)
DataConfig = _dataset_base.DataConfig
PackedDataset = _dataset_base.PackedDataset
collate_wrapper = _dataset_base.collate_wrapper

_data_utils = _import_with_log(
    "data.data_utils",
    "H2",
    "train/pretrain_unified_navit.py:project-imports",
    "before import data.data_utils",
)
add_special_tokens = _data_utils.add_special_tokens

_autoencoder = _import_with_log(
    "modeling.autoencoder",
    "H3",
    "train/pretrain_unified_navit.py:project-imports",
    "before import modeling.autoencoder",
)
load_ae = _autoencoder.load_ae

_bagel = _import_with_log(
    "modeling.bagel",
    "H4",
    "train/pretrain_unified_navit.py:project-imports",
    "before import modeling.bagel",
)
BagelConfig = _bagel.BagelConfig
Bagel = _bagel.Bagel
Qwen2Config = _bagel.Qwen2Config
Qwen2ForCausalLM = _bagel.Qwen2ForCausalLM
SiglipVisionConfig = _bagel.SiglipVisionConfig
SiglipVisionModel = _bagel.SiglipVisionModel

_qwen2 = _import_with_log(
    "modeling.qwen2",
    "H5",
    "train/pretrain_unified_navit.py:project-imports",
    "before import remaining project modules",
)
Qwen2Tokenizer = _qwen2.Qwen2Tokenizer
_train_utils = importlib.import_module("train.train_utils")
create_logger = _train_utils.create_logger
get_latest_ckpt = _train_utils.get_latest_ckpt
_fsdp_utils = importlib.import_module("train.fsdp_utils")
FSDPCheckpoint = _fsdp_utils.FSDPCheckpoint
FSDPConfig = _fsdp_utils.FSDPConfig
grad_checkpoint_check_fn = _fsdp_utils.grad_checkpoint_check_fn
fsdp_wrapper = _fsdp_utils.fsdp_wrapper
fsdp_ema_setup = _fsdp_utils.fsdp_ema_setup
fsdp_ema_update = _fsdp_utils.fsdp_ema_update

# region agent log
_agent_log("H0", "train/pretrain_unified_navit.py:project-imports", "project imports done")
# endregion

print("[debug-train-import] project imports done", flush=True)


_ACTIVE_TRAIN_LOADER_ITER = None
_ACTIVE_LOGGER = None
_WANDB_ACTIVE = False


def _dbg(msg: str):
    print(f"[debug-train] {msg}", flush=True)


def _wandb_token_metric_key(token: str) -> str:
    token_key = token.replace("<", "").replace(">", "").replace("/", "close_")
    token_key = token_key.replace("|", "_").strip("_")
    return f"special_token_loss/{token_key}"


def _local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0


def _is_node_leader() -> bool:
    return _local_rank() == 0


def _stage_enter(stage_name: str, logger=None):
    """Log stage entry once per node to avoid log storms on large clusters."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    msg = (
        f"[stage-enter] stage={stage_name} rank={rank}/{world_size} "
        f"host={socket.gethostname()} local_rank={_local_rank()}"
    )
    if _is_node_leader() or world_size <= 16:
        _dbg(msg)
        if logger is not None:
            logger.info(msg)
    # region agent log
    _agent_log(
        "SYNC",
        "train/pretrain_unified_navit.py:stage-enter",
        msg,
        {
            "stage": stage_name,
            "rank": rank,
            "world_size": world_size,
            "host": socket.gethostname(),
            "local_rank": _local_rank(),
        },
    )
    # endregion


def _sync_stage(stage_name: str, logger=None, local_elapsed_sec: Optional[float] = None, topk: int = 8):
    """Synchronize all ranks and print rank0 summary for straggler diagnosis."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    wait_start = time()
    dist.barrier()
    wait_sec = time() - wait_start
    msg = (
        f"[stage-sync] stage={stage_name} rank={rank}/{world_size} "
        f"wait_sec={wait_sec:.3f} local_elapsed_sec="
        f"{(local_elapsed_sec if local_elapsed_sec is not None else -1.0):.3f}"
    )
    if _is_node_leader() or world_size <= 16:
        _dbg(msg)
        if logger is not None:
            logger.info(msg)

    records = [None for _ in range(world_size)]
    dist.all_gather_object(
        records,
        {
            "rank": rank,
            "host": socket.gethostname(),
            "local_rank": _local_rank(),
            "wait_sec": float(wait_sec),
            "local_elapsed_sec": (float(local_elapsed_sec) if local_elapsed_sec is not None else None),
        },
    )

    if rank == 0:
        wait_values = sorted(float(r["wait_sec"]) for r in records if r is not None)
        idx95 = min(len(wait_values) - 1, int(len(wait_values) * 0.95))
        summary_msg = (
            f"[stage-sync-summary] stage={stage_name} world={world_size} "
            f"wait_sec(min/p95/max)="
            f"{wait_values[0]:.3f}/{wait_values[idx95]:.3f}/{wait_values[-1]:.3f}"
        )
        _dbg(summary_msg)
        if logger is not None:
            logger.info(summary_msg)

        elapsed_records = [r for r in records if r is not None and r["local_elapsed_sec"] is not None]
        if elapsed_records:
            elapsed_sorted = sorted(float(r["local_elapsed_sec"]) for r in elapsed_records)
            eidx95 = min(len(elapsed_sorted) - 1, int(len(elapsed_sorted) * 0.95))
            elapsed_summary_msg = (
                f"[stage-sync-summary] stage={stage_name} local_elapsed_sec(min/p95/max)="
                f"{elapsed_sorted[0]:.3f}/{elapsed_sorted[eidx95]:.3f}/{elapsed_sorted[-1]:.3f}"
            )
            _dbg(elapsed_summary_msg)
            if logger is not None:
                logger.info(elapsed_summary_msg)

            top_slow = sorted(
                elapsed_records, key=lambda x: float(x["local_elapsed_sec"]), reverse=True
            )[: max(1, min(topk, len(elapsed_records)))]
            for rec in top_slow:
                slow_msg = (
                    f"[stage-sync-top] stage={stage_name} rank={rec['rank']} "
                    f"host={rec['host']} local_rank={rec['local_rank']} "
                    f"local_elapsed_sec={float(rec['local_elapsed_sec']):.3f} "
                    f"wait_sec={float(rec['wait_sec']):.3f}"
                )
                _dbg(slow_msg)
                if logger is not None:
                    logger.info(slow_msg)
    # region agent log
    _agent_log(
        "SYNC",
        "train/pretrain_unified_navit.py:stage-sync",
        msg,
        {
            "stage": stage_name,
            "rank": rank,
            "world_size": world_size,
            "host": socket.gethostname(),
            "local_rank": _local_rank(),
            "wait_sec": round(wait_sec, 3),
            "local_elapsed_sec": (round(local_elapsed_sec, 3) if local_elapsed_sec is not None else None),
        },
    )
    # endregion


def _shutdown_dataloader_workers(logger=None):
    """Best-effort worker shutdown to avoid orphan dataloader processes."""
    global _ACTIVE_TRAIN_LOADER_ITER
    it = _ACTIVE_TRAIN_LOADER_ITER
    if it is None:
        return
    try:
        if hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
            _dbg("[cleanup] dataloader workers shut down")
            if logger is not None:
                logger.info("[cleanup] dataloader workers shut down")
    except Exception as e:
        _dbg(f"[cleanup] dataloader worker shutdown failed: {e}")
        if logger is not None:
            logger.warning(f"[cleanup] dataloader worker shutdown failed: {e}")
    finally:
        _ACTIVE_TRAIN_LOADER_ITER = None


def _cleanup_runtime(logger=None):
    global _WANDB_ACTIVE
    active_logger = logger if logger is not None else _ACTIVE_LOGGER
    if _WANDB_ACTIVE:
        try:
            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                wandb.finish()
            _WANDB_ACTIVE = False
        except Exception as e:
            _dbg(f"[cleanup] wandb finish failed: {e}")
            if active_logger is not None:
                active_logger.warning(f"[cleanup] wandb finish failed: {e}")
    _shutdown_dataloader_workers(active_logger)
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
            _dbg("[cleanup] process group destroyed")
            if active_logger is not None:
                active_logger.info("[cleanup] process group destroyed")
        except Exception as e:
            _dbg(f"[cleanup] destroy_process_group failed: {e}")
            if active_logger is not None:
                active_logger.warning(f"[cleanup] destroy_process_group failed: {e}")


def _install_termination_handlers(logger=None):
    def _handle_signal(signum, _frame):
        _dbg(f"[cleanup] received signal={signum}, request graceful shutdown")
        if logger is not None:
            logger.warning(f"[cleanup] received signal={signum}, request graceful shutdown")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def _start_init_watchdog(timeout_sec: int):
    """
    Guard against hard hangs inside dist.init_process_group.
    If timeout is reached before being stopped, force exit with code 75 so elastic can restart.
    """
    stop_event = threading.Event()

    # region agent log
    _agent_log(
        "H12",
        "train/pretrain_unified_navit.py:init-watchdog",
        "init_watchdog_start",
        {
            "timeout_sec": int(timeout_sec),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "env_rank": os.environ.get("RANK"),
            "env_local_rank": os.environ.get("LOCAL_RANK"),
            "env_world_size": os.environ.get("WORLD_SIZE"),
            "elastic_restart_count": _elastic_restart_count(),
        },
    )
    # endregion

    def _watch():
        if stop_event.wait(timeout=max(1, int(timeout_sec))):
            return
        # region agent log
        _agent_log(
            "H12",
            "train/pretrain_unified_navit.py:init-watchdog",
            "init_watchdog_timeout",
            {
                "timeout_sec": int(timeout_sec),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "env_rank": os.environ.get("RANK"),
                "env_local_rank": os.environ.get("LOCAL_RANK"),
                "env_world_size": os.environ.get("WORLD_SIZE"),
                "elastic_restart_count": _elastic_restart_count(),
                "exit_code": 75,
            },
        )
        # endregion
        os._exit(75)

    th = threading.Thread(target=_watch, name="dist-init-watchdog", daemon=True)
    th.start()
    return stop_event


def _classify_recoverable_exception(exc: Exception) -> tuple[bool, str]:
    """Classify whether a distributed failure is safe to auto-restart."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    recoverable_patterns = (
        "timeout",
        "timed out",
        "rendezvous",
        "tcpstore",
        "broken pipe",
        "connection reset",
        "connection refused",
        "connection closed",
        "socket",
        "nccl",
        "collective",
        "allreduce",
        "barrier",
        "watchdog",
    )
    for pat in recoverable_patterns:
        if pat in msg:
            return True, pat
    return False, "non_recoverable"


def _elastic_restart_count() -> int:
    try:
        return int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    except ValueError:
        return -1


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


def build_wandb_input_log(
    data,
    data_indexes,
    tokenizer,
    max_preview_tokens=128,
    max_preview_chars=800,
    max_index_items=8,
    log_images=False,
    max_logged_images=-1,
):
    """
    Build a compact input snapshot for W&B logging.
    Keep this lightweight to avoid affecting training throughput.
    """
    sample_lens = data.get("sample_lens", [])
    input_log = {
        "input/sequence_length": int(data.get("sequence_length", 0)),
        "input/num_samples_in_batch": int(len(sample_lens)),
        "input/num_text_tokens": int(data["packed_text_ids"].numel()) if "packed_text_ids" in data else 0,
        "input/num_vit_tokens": int(data["packed_vit_token_indexes"].numel()) if "packed_vit_token_indexes" in data else 0,
        "input/num_vae_tokens": int(data["packed_vae_token_indexes"].numel()) if "packed_vae_token_indexes" in data else 0,
    }

    if len(sample_lens) > 0:
        input_log["input/sample_len_min"] = int(min(sample_lens))
        input_log["input/sample_len_max"] = int(max(sample_lens))
        input_log["input/sample_len_avg"] = float(sum(sample_lens) / len(sample_lens))

    if data_indexes:
        dataset_counter = {}
        reference_images = []
        generation_images = []
        for item in data_indexes:
            dataset_name = item.get("dataset_name", "unknown")
            dataset_counter[dataset_name] = dataset_counter.get(dataset_name, 0) + 1
            reference_images.extend(item.get("reference_images", []))
            generation_images.extend(item.get("generation_images", []))
        input_log["input/dataset_mix"] = json.dumps(dataset_counter, ensure_ascii=False)
        input_log["input/data_indexes_preview"] = json.dumps(
            data_indexes[:max_index_items], ensure_ascii=False
        )
        if log_images:
            if max_logged_images > 0:
                reference_images = reference_images[:max_logged_images]
                generation_images = generation_images[:max_logged_images]
            if reference_images:
                input_log["input/reference_images"] = [
                    wandb.Image(img_path, caption=os.path.basename(img_path))
                    for img_path in reference_images
                ]
            if generation_images:
                input_log["input/generation_images"] = [
                    wandb.Image(img_path, caption=os.path.basename(img_path))
                    for img_path in generation_images
                ]

    try:
        if "packed_text_ids" in data and tokenizer is not None:
            token_ids = data["packed_text_ids"][:max_preview_tokens].detach().cpu().tolist()
            preview_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            input_log["input/text_preview"] = preview_text[:max_preview_chars]
    except Exception:
        # Never let preview logging affect training.
        pass

    return input_log


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Pin CPU memory for faster H2D copy; disable for stability when debugging large clusters."}
    )
    persistent_workers: bool = field(
        default=False,
        metadata={"help": "Keep DataLoader workers alive across iterations; disable to reduce stale worker risk."}
    )
    dataloader_timeout_sec: int = field(
        default=0,
        metadata={"help": "DataLoader timeout in seconds. 0 means wait forever."}
    )
    multiprocessing_context: Optional[str] = field(
        default=None,
        metadata={"help": "DataLoader worker start method: spawn/fork/forkserver. None keeps PyTorch default."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )
    wandb_log_input_data: bool = field(
        default=False,
        metadata={"help": "Log a compact snapshot of model input data to W&B."}
    )
    wandb_log_input_every: int = field(
        default=200,
        metadata={"help": "Log input snapshot every N steps when wandb_log_input_data=True."}
    )
    wandb_input_preview_tokens: int = field(
        default=128,
        metadata={"help": "Number of packed text tokens to decode for input preview."}
    )
    wandb_log_input_images: bool = field(
        default=False,
        metadata={"help": "Log reference/gen images from input batch to W&B."}
    )
    wandb_max_logged_images: int = field(
        default=-1,
        metadata={"help": "Max number of images per type to log; -1 means log all."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=1000_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )


def main():
    global _ACTIVE_LOGGER
    global _WANDB_ACTIVE
    # region agent log
    _agent_log(
        "H12",
        "train/pretrain_unified_navit.py:main",
        "main_enter",
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "env_rank": os.environ.get("RANK"),
            "env_local_rank": os.environ.get("LOCAL_RANK"),
            "env_world_size": os.environ.get("WORLD_SIZE"),
            "elastic_restart_count": _elastic_restart_count(),
        },
    )
    # endregion
    _dbg("entered main()")
    # region agent log
    _agent_log(
        "H12",
        "train/pretrain_unified_navit.py:main",
        "before_assert_cuda_available",
        {
            "cuda_available": bool(torch.cuda.is_available()),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "env_rank": os.environ.get("RANK"),
            "env_local_rank": os.environ.get("LOCAL_RANK"),
            "env_world_size": os.environ.get("WORLD_SIZE"),
            "elastic_restart_count": _elastic_restart_count(),
        },
    )
    # endregion
    assert torch.cuda.is_available()
    _dbg("before dist.init_process_group")
    # region agent log
    _agent_log(
        "H12",
        "train/pretrain_unified_navit.py:main",
        "before_init_process_group",
        {
            "backend": "nccl",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "env_rank": os.environ.get("RANK"),
            "env_local_rank": os.environ.get("LOCAL_RANK"),
            "env_world_size": os.environ.get("WORLD_SIZE"),
            "env_master_addr": os.environ.get("MASTER_ADDR"),
            "env_master_port": os.environ.get("MASTER_PORT"),
            "elastic_restart_count": _elastic_restart_count(),
        },
    )
    # endregion
    dist_init_timeout_sec = int(os.environ.get("DIST_INIT_TIMEOUT_SEC", "180"))
    watchdog_grace_sec = int(os.environ.get("DIST_INIT_WATCHDOG_GRACE_SEC", "30"))
    watchdog_stop_event = _start_init_watchdog(dist_init_timeout_sec + watchdog_grace_sec)
    try:
        dist.init_process_group("nccl", timeout=timedelta(seconds=dist_init_timeout_sec))
    except Exception as e:
        # region agent log
        _agent_log(
            "H12",
            "train/pretrain_unified_navit.py:main",
            "init_process_group_exception",
            {
                "backend": "nccl",
                "timeout_sec": dist_init_timeout_sec,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "env_rank": os.environ.get("RANK"),
                "env_local_rank": os.environ.get("LOCAL_RANK"),
                "env_world_size": os.environ.get("WORLD_SIZE"),
                "env_master_addr": os.environ.get("MASTER_ADDR"),
                "env_master_port": os.environ.get("MASTER_PORT"),
                "elastic_restart_count": _elastic_restart_count(),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        # endregion
        raise
    finally:
        watchdog_stop_event.set()
    # region agent log
    _agent_log(
        "H12",
        "train/pretrain_unified_navit.py:main",
        "after_init_process_group",
        {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "env_rank": os.environ.get("RANK"),
            "env_local_rank": os.environ.get("LOCAL_RANK"),
            "elastic_restart_count": _elastic_restart_count(),
        },
    )
    # endregion
    _dbg(f"dist.init_process_group done, rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    _dbg(f"cuda device set to {device}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    _dbg("arguments parsed")
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    use_wandb = False
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        try:
            wandb.init(
                project=training_args.wandb_project,
                id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
                name=training_args.wandb_name,
                resume=training_args.wandb_resume,
                mode="offline" if training_args.wandb_offline else "online",
                settings=wandb.Settings(init_timeout=120)
            )
            wandb.config.update(training_args)
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            use_wandb = True
            _WANDB_ACTIVE = True
        except Exception as e:
            # Keep training alive if W&B auth/network is unavailable.
            logger.warning(f"W&B init failed, continue without W&B: {e}")
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    _ACTIVE_LOGGER = logger
    _install_termination_handlers(logger)
    _sync_stage("post_logger_setup", logger)
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    _stage_enter("model_setup", logger)
    model_build_t0 = time()
    # region agent log
    _agent_log(
        "H11",
        "train/pretrain_unified_navit.py:model-build",
        "model construction start",
        {"rank": dist.get_rank(), "world_size": dist.get_world_size()},
    )
    # endregion
    if training_args.finetune_from_hf:
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "before load llm_config from json",
            {"rank": dist.get_rank(), "model_path": model_args.model_path},
        )
        # endregion
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "before load llm_config from pretrained",
            {"rank": dist.get_rank(), "llm_path": model_args.llm_path},
        )
        # endregion
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    # region agent log
    _agent_log(
        "H11",
        "train/pretrain_unified_navit.py:model-build",
        "before build language model",
        {"rank": dist.get_rank(), "finetune_from_hf": training_args.finetune_from_hf},
    )
    # endregion
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    # region agent log
    _agent_log(
        "H11",
        "train/pretrain_unified_navit.py:model-build",
        "after build language model",
        {"rank": dist.get_rank(), "elapsed_sec": round(time() - model_build_t0, 3)},
    )
    # endregion
    if training_args.copy_init_moe:
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "before language_model.init_moe",
            {"rank": dist.get_rank()},
        )
        # endregion
        language_model.init_moe()
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "after language_model.init_moe",
            {"rank": dist.get_rank(), "elapsed_sec": round(time() - model_build_t0, 3)},
        )
        # endregion

    if training_args.visual_und:  
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "before build vit model",
            {"rank": dist.get_rank(), "finetune_from_hf": training_args.finetune_from_hf},
        )
        # endregion
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "after build vit model",
            {"rank": dist.get_rank(), "elapsed_sec": round(time() - model_build_t0, 3)},
        )
        # endregion

    if training_args.visual_gen:
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "before load vae",
            {"rank": dist.get_rank(), "finetune_from_hf": training_args.finetune_from_hf},
        )
        # endregion
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )
        # region agent log
        _agent_log(
            "H11",
            "train/pretrain_unified_navit.py:model-build",
            "after load vae",
            {"rank": dist.get_rank(), "elapsed_sec": round(time() - model_build_t0, 3)},
        )
        # endregion

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )
    # region agent log
    _agent_log(
        "H11",
        "train/pretrain_unified_navit.py:model-build",
        "after build Bagel wrapper",
        {"rank": dist.get_rank(), "elapsed_sec": round(time() - model_build_t0, 3)},
    )
    # endregion

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    _dbg(f"rank {dist.get_rank()} start tokenizer setup")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    tracked_special_tokens = (
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<recaption>",
        "</recaption>",
    )
    tracked_special_token_ids = {}
    for token in tracked_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            tracked_special_token_ids[token] = token_id

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    _sync_stage("post_model_setup", logger, local_elapsed_sec=time() - model_build_t0, topk=16)

    # Setup packed dataloader
    _stage_enter("dataloader_setup", logger)
    dataloader_setup_t0 = time()
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    dataloader_kwargs = dict(
        dataset=train_dataset,
        batch_size=1,  # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=data_args.pin_memory,
        collate_fn=collate_wrapper(),
        drop_last=True,
        timeout=max(0, data_args.dataloader_timeout_sec),
    )
    if data_args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = data_args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = data_args.persistent_workers
        if data_args.multiprocessing_context:
            dataloader_kwargs["multiprocessing_context"] = data_args.multiprocessing_context
    train_loader = DataLoader(**dataloader_kwargs)
    _sync_stage(
        "post_dataloader_setup",
        logger,
        local_elapsed_sec=time() - dataloader_setup_t0,
        topk=16,
    )
    # region agent log
    _agent_log(
        "D1",
        "train/pretrain_unified_navit.py:step0",
        "after_post_dataloader_setup",
        {"rank": dist.get_rank(), "world_size": dist.get_world_size()},
    )
    # endregion

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    # region agent log
    _agent_log(
        "D2",
        "train/pretrain_unified_navit.py:step0",
        "before_train_loader_iter",
        {"rank": dist.get_rank()},
    )
    # endregion
    train_loader_iter = iter(train_loader)
    # region agent log
    _agent_log(
        "D2",
        "train/pretrain_unified_navit.py:step0",
        "after_train_loader_iter",
        {"rank": dist.get_rank()},
    )
    # endregion
    global _ACTIVE_TRAIN_LOADER_ITER
    _ACTIVE_TRAIN_LOADER_ITER = train_loader_iter
    # region agent log
    _agent_log(
        "D2",
        "train/pretrain_unified_navit.py:step0",
        "before_first_batch_next",
        {"rank": dist.get_rank()},
    )
    # endregion
    first_batch = next(train_loader_iter)
    # region agent log
    _agent_log(
        "D2",
        "train/pretrain_unified_navit.py:step0",
        "after_first_batch_next",
        {"rank": dist.get_rank()},
    )
    # endregion
    _dbg(f"rank {dist.get_rank()} first batch fetched from dataloader")
    for micro_step, data in enumerate(chain([first_batch], train_loader_iter)):
        if micro_step == 0:
            _dbg(f"rank {dist.get_rank()} entered training loop step0")
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        if micro_step == 0:
            # region agent log
            _agent_log(
                "D6",
                "train/pretrain_unified_navit.py:step0",
                "before_cuda_to_dict",
                {"rank": dist.get_rank(), "device": int(device)},
            )
            # endregion
        try:
            data = data.cuda(device).to_dict()
        except Exception as e:
            # region agent log
            _agent_log(
                "D6",
                "train/pretrain_unified_navit.py:step0",
                "cuda_to_dict_exception",
                {
                    "rank": dist.get_rank(),
                    "device": int(device),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            # endregion
            raise
        if micro_step == 0:
            # region agent log
            _agent_log(
                "D6",
                "train/pretrain_unified_navit.py:step0",
                "after_cuda_to_dict",
                {
                    "rank": dist.get_rank(),
                    "has_sequence_length": "sequence_length" in data,
                    "has_ce_loss_indexes": "ce_loss_indexes" in data,
                },
            )
            # endregion
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        tracked_special_token_stats = {}
        if (
            dist.get_rank() == 0
            and use_wandb
            and training_args.wandb_log_input_data
            and curr_step % training_args.wandb_log_input_every == 0
        ):
            wandb_input_log = build_wandb_input_log(
                data=data,
                data_indexes=data_indexes,
                tokenizer=tokenizer,
                max_preview_tokens=training_args.wandb_input_preview_tokens,
                log_images=training_args.wandb_log_input_images,
                max_logged_images=training_args.wandb_max_logged_images,
            )
            wandb.log(wandb_input_log, step=curr_step)
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        if micro_step == 0:
            # region agent log
            _agent_log(
                "D3",
                "train/pretrain_unified_navit.py:step0",
                "before_tokens_all_reduce",
                {"rank": dist.get_rank(), "tokens": float(tokens_tensor.item())},
            )
            # endregion
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        if micro_step == 0:
            # region agent log
            _agent_log(
                "D3",
                "train/pretrain_unified_navit.py:step0",
                "after_tokens_all_reduce",
                {"rank": dist.get_rank(), "tokens_sum": float(tokens_tensor.item())},
            )
            # endregion
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
        else:
            sample_square = torch.tensor(0.0, dtype=torch.float32, device=device)
        if micro_step == 0:
            # region agent log
            _agent_log(
                "D4",
                "train/pretrain_unified_navit.py:step0",
                "before_sample_square_all_reduce",
                {"rank": dist.get_rank(), "sample_square": float(sample_square.item())},
            )
            # endregion
        dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
        if micro_step == 0:
            # region agent log
            _agent_log(
                "D4",
                "train/pretrain_unified_navit.py:step0",
                "after_sample_square_all_reduce",
                {"rank": dist.get_rank(), "sample_square_sum": float(sample_square.item())},
            )
            # endregion
        seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                if micro_step == 0:
                    # region agent log
                    _agent_log(
                        "D5",
                        "train/pretrain_unified_navit.py:step0",
                        "before_fsdp_forward",
                        {"rank": dist.get_rank()},
                    )
                    # endregion
                loss_dict = fsdp_model(**data)
                if micro_step == 0:
                    # region agent log
                    _agent_log(
                        "D5",
                        "train/pretrain_unified_navit.py:step0",
                        "after_fsdp_forward",
                        {
                            "rank": dist.get_rank(),
                            "ce_is_none": loss_dict.get("ce") is None,
                            "mse_is_none": loss_dict.get("mse") is None,
                        },
                    )
                    # endregion
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            should_log_special_token_loss = curr_step % training_args.log_every == 0
            if should_log_special_token_loss and len(tracked_special_token_ids) > 0:
                packed_label_ids = data['packed_label_ids']
                ce_detached = ce.detach()
                for token, token_id in tracked_special_token_ids.items():
                    token_mask = packed_label_ids == token_id
                    token_count = token_mask.sum().to(torch.float32)
                    token_loss_sum = (
                        ce_detached[token_mask].sum()
                        if token_mask.any()
                        else torch.tensor(0.0, device=device)
                    )
                    dist.all_reduce(token_loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
                    tracked_special_token_stats[token] = {
                        "count": token_count.item(),
                        "loss": (
                            token_loss_sum / token_count.clamp_min(1.0)
                        ).item(),
                    }
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value
            for token, token_stats in tracked_special_token_stats.items():
                token_key = _wandb_token_metric_key(token)
                wandb_log[token_key] = token_stats["loss"]
                wandb_log[f"{token_key}_count"] = token_stats["count"]

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0 and use_wandb:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list
        )
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0 and use_wandb:
        wandb.finish()
        _WANDB_ACTIVE = False
    _cleanup_runtime(logger)


if __name__ == "__main__":
    _dbg("python entry reached (__main__)")
    # region agent log
    _agent_log(
        "H12",
        "train/pretrain_unified_navit.py:__main__",
        "__main___entered",
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "env_rank": os.environ.get("RANK"),
            "env_local_rank": os.environ.get("LOCAL_RANK"),
            "env_world_size": os.environ.get("WORLD_SIZE"),
            "elastic_restart_count": _elastic_restart_count(),
        },
    )
    # endregion
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        _dbg("[cleanup] KeyboardInterrupt received, exiting gracefully")
        exit_code = 130
    except Exception as e:
        err_rank = -1
        err_world = -1
        if dist.is_available() and dist.is_initialized():
            err_rank = dist.get_rank()
            err_world = dist.get_world_size()
        else:
            try:
                err_rank = int(os.environ.get("RANK", "-1"))
            except ValueError:
                err_rank = -1
            try:
                err_world = int(os.environ.get("WORLD_SIZE", "-1"))
            except ValueError:
                err_world = -1
        is_recoverable, reason = _classify_recoverable_exception(e)
        exit_code = 75 if is_recoverable else 1
        # region agent log
        _agent_log(
            "HERR",
            "train/pretrain_unified_navit.py:__main__",
            "unhandled_exception",
            {
                "rank": err_rank,
                "world_size": err_world,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "env_rank": os.environ.get("RANK"),
                "env_local_rank": os.environ.get("LOCAL_RANK"),
                "env_world_size": os.environ.get("WORLD_SIZE"),
                "elastic_restart_count": _elastic_restart_count(),
                "error_type": type(e).__name__,
                "error": str(e),
                "recoverable": bool(is_recoverable),
                "recoverable_reason": reason,
                "exit_code": exit_code,
                "traceback": traceback.format_exc(),
            },
        )
        # endregion
        _dbg(
            f"[error-policy] recoverable={is_recoverable} reason={reason} "
            f"exit_code={exit_code}"
        )
    finally:
        _cleanup_runtime()
    if exit_code != 0:
        sys.exit(exit_code)
