import os
os.environ['CUDA_VISIBLE_DEVICES']="6,7"
import sys

import pandas as pd
from torch.utils.data import DataLoader, random_split
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
import torch
import os
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor,Qwen2VLForConditionalGeneration
import torch
from SFRon import Mask_grad, Mask_Our

model_id="path_to_original_model"
model_path="path_to_vanilla_model"
model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
processor = AutoProcessor.from_pretrained(model_id)
# Additional processor configuration if necessary
processor.tokenizer.padding_side = "right"  # Ensure right padding
print(model)

mg=Mask_Our(model,1e-5)


forget_ratio=5

tofu_df=load_dataset("data/CLEAR/full+tofu",split="train")#tofu is the knowledge that we want to preserve
forget_df=load_dataset(f"data/CLEAR/forget{forget_ratio:02}",split="train")#forget is the dataset that we want to forget
retain_df=load_dataset(f"data/CLEAR/retain{100-forget_ratio}",split="train")#retain is the dataset that we want to preserve

multimodal_tofu_dataset = CLEAR_Dataset(data=tofu_df,mode=NONE_MODE)
multimodal_forget_dataset = CLEAR_Dataset(data=forget_df,mode=CAPTION_MODE)
multimodal_remain_dataset = CLEAR_Dataset(data=retain_df,mode=CAPTION_MODE)

language_preserve_dataset=torch.utils.data.ConcatDataset([multimodal_tofu_dataset, multimodal_remain_dataset])
vision_preserve_dataset=torch.utils.data.ConcatDataset([multimodal_remain_dataset])

ans_only=False
if ans_only:
    train_collate_function = train_collate_clear_ansonly
else:
    train_collate_function = train_collate_clear

forget_dataloader = DataLoader(
    multimodal_forget_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: train_collate_function(x, processor,"cuda", True)
)
vision_preserve_dataloader = DataLoader(
    vision_preserve_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: train_collate_function(x, processor,"cuda",  True)
)
language_preserve_dataloader = DataLoader(
    language_preserve_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: train_collate_function(x, processor,"cuda",  True)
)

root_dir=f"path_to_save_mask/forget{forget_ratio}"
os.makedirs(root_dir,exist_ok=True)
weight_mask,forget_grad,preserve_grad=mg.prepare_weight_saliency_mask(modules=["visual"], forget_loader=forget_dataloader, preserve_loader=vision_preserve_dataloader, threshold=1,save_path="")
res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
torch.save(res,f"{root_dir}/clear_vision_mask.pt")
weight_mask,forget_grad,preserve_grad=mg.prepare_weight_saliency_mask(modules=["model"], forget_loader=forget_dataloader, preserve_loader=language_preserve_dataloader, threshold=1,save_path="")
res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
torch.save(res,f"{root_dir}/clear_language_mask.pt")

weight_mask,forget_grad,preserve_grad=mg.get_weight_saliency_mask()
res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
torch.save(res,f"{root_dir}/clear_both_mask.pt")