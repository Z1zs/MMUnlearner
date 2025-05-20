import sys
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset
import torch
import json
import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.data_preprocess import LLAVA_multimodal_Dataset, train_collate_fn_mllmu, train_collate_mllmu_ansonly,Vanilla_LLaVA_Dataset
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from data_process.MLLMU_process import train_collate_fn_llava_new
from accelerate import Accelerator
import torch

def find_all_linear_names(model):
    print(model)
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["embeddings","embed_tokens","patch_embed"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else ".".join(names[-2:]))

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    # if "qwen" in str(model).lower():
    #     lora_module_names.remove('proj')
    return list(lora_module_names)

# Example usage:
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
        print("getting peft model")
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)
    print(model)
    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)


    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)


    if processor:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_clear(x, processor, model.device,True)
        )

        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_clear(x, processor, model.device,True)
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

    train_dataloader_forget=enumerate(train_dataloader_forget)
    train_dataloader_retain=enumerate(train_dataloader_retain)# to iterate manually

    # Training loop with gradient ascent
    for epoch in range(args.num_epochs):
        model.train()
        total_loss_forget = 0
        total_loss_retain = 0
        if args.gradient_accumulation:
            pass
        else:
            for iter in tqdm(range(0,n_iters)):
                if iter % forget_freq == 0: # update only in specific forget round
                    try:
                        _, forget_batch=next(train_dataloader_forget)# avoid StopIteration Error
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

                _, retain_batch=next(train_dataloader_retain)
                retain_outputs = model(**retain_batch)
                loss_retain = retain_outputs.loss
                if loss_retain>1e+10:
                    torch.save(retain_batch,"Ill_batch.pt")
                    output = model.generate(**retain_batch, max_new_tokens=30, do_sample=False)
                    print(processor.decode(output[0], skip_special_tokens=True))
                    optimizer.zero_grad()
                    exit(-1)
                # print(retain_batch.keys(),retain_outputs.keys())
                accelerator.backward(loss_retain)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    if isinstance(model, PeftModel):
        unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--grad_mask_path", type=str, default=None, help="Mask for gradient update")
    parser.add_argument("--forget_alpha", type=float, default=1.0, help="Forget loss weight")
    args = parser.parse_args()
    main(args)
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)