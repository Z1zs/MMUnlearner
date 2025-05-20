import os
import sys

import pandas as pd
from torch.utils.data import DataLoader, random_split
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from data_process.MLLMU_process import K_TYPE, P_TYPE, MLLMU_manifold_Dataset,train_collate_fn_llava_new,MLLMU_text_Dataset
from data_process.CLEAR_process import train_collate_clear
from data_process.data_preprocess import  LLAVA_multimodal_Dataset
import torch
import os
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor,Qwen2VLForConditionalGeneration
import torch
from SFRon import Mask_Our, Mask_grad

model_id="path_to_original_model"
model_path="path_to_vanilla_model"
model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )

processor = AutoProcessor.from_pretrained(model_id)
# Additional processor configuration if necessary
processor.tokenizer.padding_side = "right"
print(model)

mg=Mask_Our(model,1e-5)


forget_ratio=5
forget_data_folder=f"/data/MLLMMU-Bench/forget_{forget_ratio}"
remain_data_folder=f"/data/MLLMMU-Bench/retain_{100-forget_ratio}"

# Define paths to the Parquet files for "forget" and "retain" datasets
forget_parquet_file = os.path.join(forget_data_folder, f"train-00000-of-00001.parquet")
remain_parquet_file = os.path.join(remain_data_folder, f"train-00000-of-00001.parquet")

# Load DataLoader
forget_df = pd.read_parquet(forget_parquet_file)
retain_df = pd.read_parquet(remain_parquet_file)
full_df=pd.read_parquet("/data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet")

multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)

multimodal_knowledge_dataset = MLLMU_text_Dataset(df=full_df)
multimodal_remain_dataset = LLAVA_multimodal_Dataset(df=retain_df)

language_preserve_dataset=torch.utils.data.ConcatDataset([multimodal_knowledge_dataset, multimodal_remain_dataset])
vision_preserve_dataset=torch.utils.data.ConcatDataset([multimodal_remain_dataset])


forget_dataloader = DataLoader(
    multimodal_forget_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: train_collate_clear(x, processor,model.device, True)
)
vision_preserve_dataloader = DataLoader(
    vision_preserve_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: train_collate_clear(x, processor,model.device, True)
)
language_preserve_dataloader = DataLoader(
    language_preserve_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: train_collate_clear(x, processor, model.device,True)
)

label="ours"
folder=f"path_to_save_mask/forget{forget_ratio}"
os.makedirs(folder,exist_ok=True)
weight_mask,forget_grad,preserve_grad=mg.prepare_weight_saliency_mask(modules=["vision_tower","multi_modal_projector"], forget_loader=forget_dataloader, preserve_loader=vision_preserve_dataloader, threshold=1,save_path="")
res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
torch.save(res,f"{folder}/mllmu_vision_mask_{label}.pt")
weight_mask,forget_grad,preserve_grad=mg.prepare_weight_saliency_mask(modules=["language_model"], forget_loader=forget_dataloader, preserve_loader=language_preserve_dataloader, threshold=1,save_path="")
res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
torch.save(res,f"{folder}/mllmu_language_mask_{label}.pt")

weight_mask,forget_grad,preserve_grad=mg.get_weight_saliency_mask()
res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
torch.save(res,f"{folder}/mllmu_both_mask_{label}.pt")