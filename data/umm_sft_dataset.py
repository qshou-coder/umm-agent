# Copyright 2026 Quanxin Shou
import json
import os
import glob
import traceback
from pathlib import Path
from PIL import Image
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

class SftAgenticIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, 
        json_path_list=None, reference_list=None, generation_list=None, num_used_data=None,
        data_dir_list=None,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=True, shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        if json_path_list is None:
            json_path_list = data_dir_list
        if json_path_list is None or reference_list is None or generation_list is None:
            raise ValueError(
                "umm_sft requires json_path_list/data_dir_list, reference_list, generation_list."
            )
        if num_used_data is None:
            num_used_data = [10 ** 9 for _ in json_path_list]

        self.data_paths = self.get_data_paths(
            json_path_list,
            reference_list,
            generation_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()
        
    def get_data_paths(
        self, json_path_list, reference_list, generation_list, 
        num_used_data, shuffle_lines, shuffle_seed,
    ):
        data_paths = []
        keys_to_keep = [
            "ip_index", "ip_name", "image_prompt", 
            "language", "country", "full_response"
        ]
        for json_path, ref_dir, gen_dir, num_data_point in zip(
            json_path_list, reference_list, generation_list, num_used_data
        ):
            items = []
            json_path_obj = Path(json_path)
            if json_path_obj.is_dir():
                json_files = sorted(json_path_obj.glob("*_trajectory.json"))
                if shuffle_lines:
                    self.rng.seed(shuffle_seed)
                    self.rng.shuffle(json_files)
                json_files = json_files[:num_data_point]
                for json_file in json_files:
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)
                    except Exception as e:
                        print(f"Error decoding JSON {json_file}: {e}")
                        continue
                    if isinstance(raw_data, dict):
                        items.append(raw_data)
                    elif isinstance(raw_data, list):
                        items.extend(raw_data)
            else:
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        raw_data = json.load(f)
                    except Exception as e:
                        print(f"Error decoding JSON {json_path}: {e}")
                        continue
                if isinstance(raw_data, dict):
                    raw_data = [raw_data]
                if shuffle_lines:
                    self.rng.seed(shuffle_seed)
                    self.rng.shuffle(raw_data)
                items = raw_data[:num_data_point]

            for item in items:
                standard_conversations = []
                filtered_item = {k: item[k] for k in keys_to_keep if k in item}
                
                if "full_response" in item:
                    response_turns = item["full_response"]
                    for turn in response_turns:
                        turn_id = turn.get("turn", "")
                        standard_conversations.append({
                            "role": "user",
                            "value": turn.get("input", "")
                        })
                        standard_conversations.append({
                            "role": "assistant",
                            "value": turn.get("response_text", "")
                        })
                        
                        if turn_id == len(response_turns) - 2:
                            standard_conversations.append({
                                "role": "ref_images",
                                "value": "<ref_image>\n<ref_image>"
                            })
                        elif turn_id == len(response_turns) - 1:
                            standard_conversations.append({
                                "role": "gen_images",
                                "value": "<gen_image>"
                            })
                
                filtered_item["conversations"] = standard_conversations
                os.makedirs(f"data/test_dataloader_outputs/{item['ip_index']}", exist_ok=True)
                with open(f"data/test_dataloader_outputs/{item['ip_index']}/filtered_item.json", "w", encoding="utf-8") as f:
                    json.dump(filtered_item, f, ensure_ascii=False, indent=4)
                    f.write("\n")
                data_paths.append((filtered_item, ref_dir, gen_dir))
        return data_paths
    
    def change_format(self, data, ref_images, gen_images):
        elements = []
        for conversation in data["conversations"]:
            role = conversation["role"]
            value = conversation["value"]
            
            if role == "user":
                elements.append({'type': 'text', 'has_loss': 0, 'text': value})
            elif role == "assistant":
                elements.append({'type': 'text', 'has_loss': 1, 'text': value})
            elif role == "ref_images":
                ref_tags = [v for v in value.strip().split('\n') if v.strip()]
                for idx, _ in enumerate(ref_tags):
                    if idx < len(ref_images):
                        elements.append({
                            'type': 'ref_image',
                            'has_loss': 0,
                            'image': ref_images[idx],
                        })
            elif role == "gen_images":
                gen_tags = [v for v in value.strip().split('\n') if v.strip()]
                for idx, _ in enumerate(gen_tags):
                    if idx < len(gen_images):
                        elements.append({
                            'type': 'gen_image',
                            'has_loss': 1,
                            'image': gen_images[idx],
                        })
        return elements
    
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.data_status[worker_id] + 1 if self.data_status is not None else 0
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.jfif'}
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )
        
        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_id, (data_item, ref_dir, gen_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    ip_index = data_item["ip_index"]
                    
                    ref_image_dir = os.path.join(ref_dir, ip_index)
                    if not os.path.exists(ref_image_dir):
                        continue
                    
                    ref_list = sorted([str(p) for p in Path(ref_image_dir).iterdir() 
                                        if p.suffix.lower() in valid_exts])[:2]
                    
                    possible_gen = glob.glob(os.path.join(gen_dir, f"{ip_index}*"))
                    gen_list = [f for f in possible_gen if os.path.splitext(f)[1].lower() in valid_exts][:1]

                    if len(ref_list) != 2 or len(gen_list) != 1:
                        continue
                    
                    try:
                        ref_images = [pil_img2rgb(Image.open(img)) for img in ref_list]
                        gen_images = [pil_img2rgb(Image.open(img)) for img in gen_list]
                    except Exception as e:
                        print(f"Image load error for {ip_index}: {e}")
                        continue

                    for idx, img in enumerate(ref_images):
                        img.save(f"data/test_dataloader_outputs/{ip_index}/ref_image_{idx}.jpg")
                    for img in gen_images:
                        img.save(f"data/test_dataloader_outputs/{ip_index}/gen_image.jpg")
                    
                    elements = self.change_format(data_item, ref_images, gen_images)
                    
                    image_tensor_list = []
                    text_ids_list = []
                    sequence_plan = []
                    num_tokens = 0

                    for item in elements:
                        if item['type'] == 'text':
                            text_ids = self.tokenizer.encode(item['text'])
                            if len(text_ids) > 0:
                                text_ids_list.append(text_ids)
                                num_tokens += len(text_ids)
                                sequence_plan.append({
                                    'type': 'text', 'enable_cfg': 0, 'loss': item['has_loss'],
                                    'special_token_loss': 0, 'special_token_label': None,
                                })
                        elif item['type'] == 'ref_image':
                            vit_image_tensor = self.vit_transform(item['image'])
                            image_tensor_list.append(vit_image_tensor)
                            num_tokens += (vit_image_tensor.shape[1] * vit_image_tensor.shape[2]) // self.vit_transform.stride ** 2
                            sequence_plan.append({
                                'type': 'vit_image', 'enable_cfg': 0, 'loss': item['has_loss'],
                                'special_token_loss': 0, 'special_token_label': None,
                            })
                        elif item['type'] == 'gen_image':
                            image_tensor = self.transform(item['image'])
                            image_tensor_list.append(image_tensor)
                            num_tokens += (image_tensor.shape[1] * image_tensor.shape[2]) // self.transform.stride ** 2
                            sequence_plan.append({
                                'type': 'vae_image', 'enable_cfg': 0, 'loss': item['has_loss'],
                                'special_token_loss': 0, 'special_token_label': None,
                            })
                            
                    if not any(it['loss'] for it in sequence_plan):
                        continue
                    
                    yield dict(
                        image_tensor_list=image_tensor_list,
                        text_ids_list=text_ids_list,
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": row_id,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                except Exception:
                    print(f"Error processing row {row_id} in {self.dataset_name}:")
                    traceback.print_exc()
                    continue
            
            row_start_id = 0
            print(f"{self.dataset_name} finished one epoch in rank-{self.local_rank}")