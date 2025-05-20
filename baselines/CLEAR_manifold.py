import os
import sys

import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
import torch
import os
import json
import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration,Qwen2VLForConditionalGeneration
import json
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from data_process.CLEAR_process import train_collate_clear
from accelerate import Accelerator
import torch
from datasets import load_dataset


def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    elif "qwen" in args.model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


######################### Accelerate Version #################################
def main(args):
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup
    forget_df=load_dataset(f"data/CLEAR/forget{args.forget_ratio:02}",split="train")#forget is the dataset that we want to forget
    retain_df=load_dataset(f"data/CLEAR/retain{100-args.forget_ratio}",split="train")#retain is the dataset that we want to preserve

    multimodal_forget_dataset = CLEAR_Dataset(data=forget_df,mode=CAPTION_MODE)
    multimodal_retain_dataset = CLEAR_Dataset(data=retain_df,mode=CAPTION_MODE)


    if args.ans_only:
        train_collate_function = train_collate_clear_ansonly
    else:
        train_collate_function = train_collate_clear

    if processor:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor,"cuda", True)
        )

        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor,"cuda", True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

        
    if args.grad_mask_path:
        module_set=set()
        grad_data=torch.load(args.grad_mask_path)
        grad_mask=grad_data['weight']
        layer_name_list=list(grad_mask.keys())
        for name in layer_name_list:
            if "proj" in name:
                continue
            elif "fc" in name:
                continue
            elif "linear" in name:
                continue
            elif "mlp" in name:
                continue
            elif "qkv" in name:
                continue
            else:
                grad_mask.pop(name)
        def calc_sparsity(tensor):
            num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)
            total_elements = tensor.numel()
            sparsity = num_zero_elements / total_elements
            return sparsity.item(), total_elements, num_zero_elements

        total_cnt=0
        w_cnt=0
        for k,v in grad_mask.items():
            try: 
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(v)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                module_set=module_set|set(k.split("."))
            except: 
                pass 
        print("Saliency mask generated!")
        print(f"Total sparsity among weight:{w_cnt/total_cnt*100}")

    print(f"Fine-tuned modules: {module_set}\n",)

    accelerator = Accelerator()
    if args.gradient_accumulation:
        raise ValueError("Gradient accumulation not supported.")
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader_forget) + len(train_dataloader_retain)) * args.num_epochs,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )
    len_retain,len_forget=len(train_dataloader_retain),len(train_dataloader_forget)
    n_iters=len(train_dataloader_retain)
    forget_freq=len(train_dataloader_retain)//len(train_dataloader_forget)
    print(f"Each epoch have {n_iters} iterations, with forget_freq={forget_freq}!")

    # Training loop with gradient ascent
    for epoch in range(args.num_epochs):
        model.train()
        train_data_forget=enumerate(train_dataloader_forget)
        train_data_retain=enumerate(train_dataloader_retain)# to iterate manually
        total_loss_forget = 0
        total_loss_retain = 0
        if args.gradient_accumulation:
            pass
        else:
            for iter in tqdm(range(0,n_iters)):
                if iter % forget_freq == 0: # update only in specific forget round
                    try:
                        _, forget_batch=next(train_data_forget)# avoid StopIteration Error
                        forget_outputs = model(**forget_batch)
                        print(iter, _, forget_batch.keys(),forget_outputs.keys())
                        loss_forget = -args.forget_alpha*forget_outputs.loss  # Gradient ascent
                        accelerator.backward(loss_forget)
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if args.grad_mask_path:
                            for name, p in model.named_parameters():
                                if p.grad is not None and name in grad_mask:
                                    # print(f"Grad of {name} masked")
                                    p.grad *= grad_mask[name].to(p.grad.device)
                        # loss_forget = forget_alpha * ori_loss_forget
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss_forget += loss_forget.item()
                    except Exception as e:
                        pass

                _, retain_batch=next(train_data_retain)
                retain_outputs = model(**retain_batch)
                loss_retain = retain_outputs.loss
                # print(retain_batch.keys(),retain_outputs.keys())
                accelerator.backward(loss_retain)
                optimizer.step()
                optimizer.zero_grad()
                total_loss_retain += loss_retain.item()

                print(f"Iteration {iter + 1} - Forget Loss: {-loss_forget.item()}")
                print(f"Iteration {iter + 1} - Retain Loss: {loss_retain.item()}")
        lr_scheduler.step()      
        print(f"Epoch {epoch + 1} - Forget Loss: {-total_loss_forget / len_forget}")
        print(f"Epoch {epoch + 1} - Retain Loss: {total_loss_retain / len_retain}")      

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--forget_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--grad_mask_path", type=str, default=None, help="Mask for gradient update")
    parser.add_argument("--ans_only", type=bool, default=False, help="Answer only training")
    parser.add_argument("--forget_alpha", type=float, default=1, help="forget loss weight")
    args = parser.parse_args()
    main(args)
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)