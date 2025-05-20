import sys
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
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration,AutoTokenizer,Qwen2VLForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from accelerate import Accelerator
import torch
import torch.nn.functional as F

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
            args.vanilla_dir, 
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
            args.vanilla_dir, 
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

    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading Oracle LLAVA model...")
        oracle_model = LlavaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "llama" in args.model_id.lower():
        # Load LLAVA Next model and processor
        print("Loading Oracle llama model...")
        oracle_model = MllamaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "qwen" in args.model_id.lower():
        # Load LLAVA Next model and processor
        print("Loading Oracle qwen model...")
        oracle_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )


    print("Processor Tokenizer Length: ", len(processor.tokenizer)) #128257
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    # if args.models_id.startswith("meta-llama") == False:
    #     model.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    forget_df=load_dataset(f"{args.data_folder}/forget{args.forget_ratio.zfill(2)}",split="train")#retain is the dataset that we want to preserve
    multimodal_forget_dataset = CLEAR_Dataset(data=forget_df,mode=CAPTION_MODE)

    if args.ans_only:
        train_collate_function = train_collate_clear_ansonly
    else:
        train_collate_function = train_collate_clear

    if processor:
        train_dataloader = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor, "cuda",True)
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

    oracle_model, model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(oracle_model,
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        if args.gradient_accumulation:
            pass

        else:
            for batch in progress_bar:
                # Forward pass with the current model to get the loss
                outputs = model(**batch)
                current_loss = outputs.loss

                # Forward pass with the oracle model to get the oracle loss
                with torch.no_grad():
                    oracle_outputs = oracle_model(**batch)
                    oracle_loss = oracle_outputs.loss

                # Compute neg_log_ratios and NPO loss
                neg_log_ratios = current_loss - oracle_loss
                loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta

                # Backward pass and optimization
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

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
    parser.add_argument("--oracle_model_id", type=str, required=True, help="Oracle model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--forget_ratio", type=str, default="5", help="Directory to save the model")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--data_folder", type=str, default="./data", help="Data folder")
    parser.add_argument("--ans_only", type=bool, default=False, help="Only use answer")

    args = parser.parse_args()
    main(args)
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)
