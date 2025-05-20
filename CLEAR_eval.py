import os
import json
import random
from PIL import Image
from pydantic_core import validate_core_schema
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, MllamaForConditionalGeneration,Qwen2VLForConditionalGeneration
import pandas as pd
from io import BytesIO
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
from datasets import load_dataset,load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE
import re
random.seed(42)

def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score

def eval_classification(model, processor, data_path,with_options):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode, with_options={with_options} #########################################" )
    if "forget" in data_path:
        VQA_data=load_dataset(data_path,split="train")#forget is the dataset that we want to forget
    elif "retain" in data_path:
        VQA_data=load_from_disk(data_path)
    else:
        ValueError("Data path should contain forget or retain")
    print(VQA_data)
    correct_count,VQA_num = 0,0
    for idx,VQA_sample in enumerate(VQA_data):
        image=VQA_sample.get("image",None)
        question = VQA_sample.get("question", "What is the name of the person in the image?")
        answer = VQA_sample.get("name", "")
        options=VQA_sample.get("perturbed_names",[])
        options.insert(random.randint(0, len(options)), answer)
        if with_options:
            prompt, correct_answer = formulate_prompt_with_options(question, options, answer)
        else:
            prompt = question
            correct_answer=answer
        conversation = [
            {"role": "user","content": [{"type": "image"},{"type": "text", "text": prompt},],},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        
        with torch.no_grad():
            VQA_outputs = model.generate(**inputs,max_new_tokens=50, do_sample=False)
        
        out_wo_prompt = VQA_outputs[ : , inputs.input_ids.shape[-1] : ]
        generated_text=processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
        print("Generated text is : \n","**************\n",generated_text,"\n**************")

        if not with_options: # answer in response is okay
            if answer in assistant_response.lower():
                print("Correct Answer!")
                correct_count+=1
            else:
                print(f"Wrong Answer! ${assistant_response}$ doesn't include ${answer}$")
        else: # string matching
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() else None
            if predicted_answer==correct_answer:
                print("Correct Answer!")
                correct_count+=1
            else:
                print(f"Wrong Answer! ${predicted_answer}$ != ${correct_answer}$. {answer}")
        print("##################################")
        VQA_num+=1
    
    print(f"VQA Correct Count: {correct_count}/{VQA_num}")
    print(f"VQA Accuracy: {correct_count/VQA_num}")
    print("################################## Classification Task Ends ##############################################")
    return {"VQA Accuracy": correct_count/VQA_num}


def eval_generation(model, processor, data_path,output_folder,mode):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode #########################################" )
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    df=load_dataset(data_path,split="train")#forget is the dataset that we want to forget
    VQA_data = CLEAR_Dataset(df, mode=CAPTION_MODE)
    QA_data=CLEAR_Dataset(df, mode=NONE_MODE)

    results={"Generation_Questions":[]}
    avg_scores = {}

    total_bleu_VQA,total_rouge1_VQA,total_rouge2_VQA,total_rougeL_VQA,total_VQA_num=0,0,0,0,0
    for i, VQA_sample in enumerate(VQA_data):
        image,question,answer=VQA_sample["image"],VQA_sample["question"],VQA_sample["answer"]
        conversation = [
            {"role": "user","content": [{"type": "image"},{"type": "text", "text": question},],},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        with torch.no_grad():
            VQA_outputs = model.generate(**inputs,max_new_tokens=50, do_sample=False)
        out_wo_prompt = VQA_outputs[ : , inputs.input_ids.shape[-1] : ]
        generated_text=processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response=generated_text
        print("Generated text is : \n","**************\n",generated_text,"\n**************")
        results["Generation_Questions"].append({
                "idx":i,
                "type":"VQA",
                "image": str(image),
                "question": question,
                "generated_answer": assistant_response,
                "ground_truth": answer
        })
        # Calculate ROUGE and BLEU scores
        bleu_score = compute_bleu(answer, assistant_response)
        rouge_scores = rouge_scorer_obj.score(answer, assistant_response)
        total_bleu_VQA+= bleu_score
        total_rouge1_VQA += rouge_scores['rouge1'].fmeasure
        total_rouge2_VQA += rouge_scores['rouge2'].fmeasure
        total_rougeL_VQA += rouge_scores['rougeL'].fmeasure
        total_VQA_num += 1
    
    total_bleu_QA,total_rouge1_QA,total_rouge2_QA,total_rougeL_QA,total_QA_num=0,0,0,0,0
    for i, QA_sample in enumerate(QA_data):
        image,question,answer=QA_sample["image"],QA_sample["question"],QA_sample["answer"]
        conversation = [
            {"role": "user","content": [{"type": "text", "text": question},],},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        with torch.no_grad():
            QA_outputs = model.generate(**inputs,max_new_tokens=50, do_sample=False)
        out_wo_prompt = VQA_outputs[ : , inputs.input_ids.shape[-1] : ]
        generated_text=processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        print("Generated text is : \n","**************\n",generated_text,"\n**************")
        results["Generation_Questions"].append({
                "idx":i,
                "type":"QA",
                "image": str(image),
                "question": question,
                "generated_answer": assistant_response,
                "ground_truth": answer
        })
        # Calculate ROUGE and BLEU scores
        bleu_score = compute_bleu(answer, assistant_response)
        rouge_scores = rouge_scorer_obj.score(answer, assistant_response)
        total_bleu_QA+= bleu_score
        total_rouge1_QA += rouge_scores['rouge1'].fmeasure
        total_rouge2_QA += rouge_scores['rouge2'].fmeasure
        total_rougeL_QA += rouge_scores['rougeL'].fmeasure
        total_QA_num += 1
    
    avg_scores.update({
        "Average ROUGE-1 (VQA)": total_rouge1_VQA / total_VQA_num,
        "Average ROUGE-2 (VQA)": total_rouge2_VQA / total_VQA_num,
        "Average ROUGE-L (VQA)": total_rougeL_VQA / total_VQA_num,
        "Average BLEU (VQA)": total_bleu_VQA / total_VQA_num
    })
    avg_scores.update({
        "Average ROUGE-1 (QA)": total_rouge1_QA / total_QA_num,
        "Average ROUGE-2 (QA)": total_rouge2_QA / total_QA_num,
        "Average ROUGE-L (QA)": total_rougeL_QA / total_QA_num,
        "Average BLEU (QA)": total_bleu_QA / total_QA_num
    })

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(f'{output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("################################## Classification Task Ends ##############################################")
    return avg_scores


def formulate_prompt_with_options(question, options, answer):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (list): The options for the question (e.g., ["Option A", "Option B"]).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{chr(ord('A')+i)}. {value}" for i,value in enumerate(options)])
    gt=chr(ord('A')+options.index(answer))
    prompt = f"{question}\n{options_str}\n"
    return prompt, gt

def eval_classification_real(model, processor, data_path):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode #########################################" )
    df=load_dataset(data_path,split="train")
    correct_count,VQA_num = 0,0
    for i, sample in enumerate(df):
        question = sample.get("question", "What is the name of the person in the image?")
        answer = sample.get("answer", "")
        options=sample.get("options",[])
        options.insert(random.randint(0, len(options)), answer)
        image=sample.get("image",None)
        prompt, correct_answer = formulate_prompt_with_options(question, options, answer)
        conversation = [
            {"role": "user","content": [{"type": "image"},{"type": "text", "text": prompt},],},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
        generated_text=processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
        print("Generated text is : \n","**************\n",generated_text,"\n**************")
        predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() else None
        if predicted_answer==correct_answer:
            print("Correct Answer!")
            correct_count+=1
        else:
            print(f"Wrong Answer! ${assistant_response}$ != ${correct_answer}$. {answer}")
        VQA_num+=1
        print("##################################")
    
    print(f"VQA Correct Count: {correct_count}/{VQA_num}")
    print(f"VQA Accuracy: {correct_count/VQA_num}")
    print("################################## Classification Task Ends ##############################################")
    return {"VQA Accuracy": correct_count/VQA_num}



def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--forget_cls_folder', type=str, required=True, help='Path to the forget cls folder.')
    parser.add_argument('--forget_gen_folder', type=str, required=True, help='Path to the forget gen folder.')
    parser.add_argument('--retain_gen_folder', type=str, required=True, help='Path to the retain cls folder.')
    parser.add_argument('--retain_cls_folder', type=str, required=True, help='Path to the retain gen folder.')
    parser.add_argument('--realface_folder', type=str, required=True, help='Path to real person folder.')
    parser.add_argument('--realworld_folder', type=str, required=True, help='Path to real world folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output file.')
    parser.add_argument('--eval_list', type=str, required=True, help='Spilts waited to eval')
    parser.add_argument('--shot_num', type=str, required=True, help='Shot nums for ICL')
    return parser.parse_args()

def main():
    args = parse_arguments()
    global few_shots_num
    if "zero" in args.shot_num.lower():
        few_shots_num=0
    else:
        few_shots_num=1

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()

    if "llava" in args.model_id.lower():
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    elif "llama" in args.model_id.lower():
        print("Loading LLAMA model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    elif "qwen" in args.model_id.lower():
        print("Loading Qwen model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.cache_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )


    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    results_data={}
    if "forget" in args.eval_list:
        print("### Evaluating Forget Set ###")
        wo_flag= "perturbed" in args.forget_cls_folder.lower()
        forget_classification_result = eval_classification(model=model, processor=processor, data_path=f"{args.data_folder}/{args.forget_cls_folder}",with_options=wo_flag)

        forget_generation_result = eval_generation(model=model, processor=processor, data_path=f"{args.data_folder}/{args.forget_gen_folder}",output_folder=args.output_folder,mode="forget")
        print("Forget Set Results:")
        print(forget_classification_result)
        print(forget_generation_result)
        results_data["Forget Set Results"]={
            "classification": forget_classification_result,
            "generation": forget_generation_result
        }

    if "retain" in args.eval_list:
        print("### Evaluating Retain Shared Set ###")
        wo_flag= "perturbed" in args.retain_cls_folder.lower()
        print(f"{args.data_folder}/{args.retain_cls_folder}")
        retain_classification_result = eval_classification(model=model, processor=processor, data_path=f"{args.data_folder}/{args.retain_cls_folder}",with_options=wo_flag)

        retain_generation_result = eval_generation(model=model, processor=processor, data_path=f"{args.data_folder}/{args.retain_gen_folder}",output_folder=args.output_folder,mode="retain")
        print("Retain Set Results:")
        print(retain_classification_result)
        print(retain_generation_result)
        results_data["Retain Set Results"]= {
            "classification": retain_classification_result,
            "generation": retain_generation_result
        }
    
    if "realface" in args.eval_list:
        print("### Evaluating Real Face Set ###")

        realface_classification_result = eval_classification_real(model=model, processor=processor, data_path=f"{args.data_folder}/{args.realface_folder}")

        print("Real Face Results:")
        print(realface_classification_result)
        results_data['Real Face Results']={
            "classification": realface_classification_result
        }
    
    if "realworld" in args.eval_list:
        print("### Evaluating Real World Set ###")

        realworld_classification_result = eval_classification_real(model=model, processor=processor, data_path=f"{args.data_folder}/{args.realworld_folder}")

        print("Real World Results:")
        print(realworld_classification_result)
        results_data['Real World Results']={
            "classification": realworld_classification_result
        }

    output_file = f'{args.output_folder}/{args.output_file}_final_evaluation_results.json'

    os.makedirs(args.output_folder, exist_ok=True)
    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")

    with open(f'{args.output_folder}/{args.output_file}_evalconfig.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()