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
from accelerate import Accelerator
import torch


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model','visual']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
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
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_id, 
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    elif "qwen" in args.model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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

    # dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    # print(f"Dataset size (profiles): {len(dataset)}")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    retain_df = pd.read_parquet(retain_parquet_file)

    multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)

    if processor:
        train_dataloader = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_mllmu_ansonly(x, processor, model.device, True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4  # Adjust based on memory
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        if args.gradient_accumulation:
            for step, batch in enumerate(progress_bar):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss / accumulation_steps
                    # loss = outputs.loss
                    accelerator.backward(loss)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))
            # lr_scheduler.step()
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")
        else:
            for batch in progress_bar:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

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
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    args = parser.parse_args()

    # Call main function
    main(args)
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)