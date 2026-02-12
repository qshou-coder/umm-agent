# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .umm_sft_dataset import SftAgenticIterableDataset
from pathlib import Path


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'umm_sft': SftAgenticIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'umm_sft': {},
}


def _build_umm_sft_info(output_root):
    """
    Build umm_sft metainfo from output_root/category/{traj,intermediate,images}.
    """
    root = Path(output_root)
    if not root.exists():
        return {}

    dataset_meta = {}
    for category_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        traj_dir = category_dir / 'traj'
        reference_dir = category_dir / 'intermediate'
        generation_dir = category_dir / 'images'
        if not (traj_dir.is_dir() and reference_dir.is_dir() and generation_dir.is_dir()):
            continue

        num_samples = len(list(traj_dir.glob('*_trajectory.json')))
        if num_samples == 0:
            continue

        dataset_meta[category_dir.name] = {
            # Required by dataset_base.build_datasets.
            'data_dir': str(traj_dir),
            'reference_dir': str(reference_dir),
            'generation_dir': str(generation_dir),
            # For umm_sft, this can be a directory of *_trajectory.json files.
            'json_path': str(traj_dir),
            'num_total_samples': num_samples,
        }

    return dataset_meta


DATASET_INFO['umm_sft'].update(
    _build_umm_sft_info('/apdcephfs_zwfy2/share_303944931/shawncschen/umm_search_agent/output')
)