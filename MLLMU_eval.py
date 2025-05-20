import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration
import pandas as pd
from io import BytesIO
# from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration, MllamaProcessor, MllamaForConditionalGeneration
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

few_shots_num=0

def load_and_combine_parquet_files(directory):
    # Get all Parquet files in the directory
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

    # Read and concatenate all Parquet files
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df

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

def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt

def evaluate_classification(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    """
    Evaluate classification task with/without few-shot samples based on the mode.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        few_shot_parquet_file: Path to the Parquet file containing few-shot examples.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Mode that controls how few-shot samples are handled ('forget', 'retain_share', 'test', or others). Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################" )

    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the main Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    # Randomly select few-shot examples based on the model
    selected_ids = random.sample(id_list, few_shots_num)

    print(f"Selected few-shot IDs: {selected_ids}")

    few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
    few_shot_images = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}  # Dictionary to track few-shot question indices

    # Load few-shot examples based on selected_ids from the few-shot Parquet file
    few_shot_df = pd.read_parquet(few_shot_parquet_file)
    few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]
    for _, row in few_shot_samples.iterrows():
        classification_questions = row["Classification_Task"]
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Track the indices of few-shot questions for each ID
        few_shot_question_indices[row["ID"]] = {
            "image_textual": [],
            "pure_text": []
        }

        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            few_shot_image_prompts.append({
                "Question": question_data["Question"],
                "Options": question_data["Options"],
                "Correct Answer": question_data["Correct_Answer"]
            })
            few_shot_images.append(image)
            few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

        for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
            few_shot_text_prompts.append({
                "Question": question_data["Question"],
                "Options": question_data["Options"],
                "Correct Answer": question_data["Correct_Answer"]
            })
            few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

    print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
    print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        classification_questions = row["Classification_Task"]

        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each image-textual question
        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                continue  # Skip few-shot question

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            # Prepare few-shot prompt if applicable
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for i, few_shot_image in enumerate(few_shot_images):
                    few_shot_question = few_shot_image_prompts[i]["Question"]
                    few_shot_options = few_shot_image_prompts[i]["Options"]
                    few_shot_answer = few_shot_image_prompts[i]["Correct Answer"]
                    few_shot_prompt += (
                        f"USER: <image>\n"
                        f"Question: {few_shot_question}\n"
                        f"A: {few_shot_options['A']}\n"
                        f"B: {few_shot_options['B']}\n"
                        f"C: {few_shot_options['C']}\n"
                        f"D: {few_shot_options['D']}\n"
                        f"Correct Answer: {few_shot_answer}\n"
                    )

            # Model specific logic for generating answers
            if "llama" in args.model_id.lower():
                prompt = (f"{few_shot_prompt}"
                      f"USER: <|image|><|begin_of_text|>\n{question_with_options}\n"
                      f"Just give ONE letter representing the answer directly.\nASSISTANT:")
                inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            elif "llava" in args.model_id.lower():
                prompt = (f"{few_shot_prompt}"
                      f"USER: <image>\n{question_with_options}\n"
                      f"Just give ONE letter representing the answer directly.\nASSISTANT:")
                inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            elif "qwen" in args.model_id.lower():
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{few_shot_prompt}\n{question_with_options}\nJust give ONE letter representing the answer directly."},
                                # {"type": "text", "text": f"{few_shot_prompt}\n{question_with_options}\n"},
                            ],
                        }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=[*few_shot_images, image],
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    # outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=10)
            out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
            generated_text=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
            assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
            print("Prompt is : ","**************",prompt,"**************")
            print("Generated text is : ","**************",assistant_response,"**************")
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            print("Predicted answer is : ","**************",predicted_answer,"**************")
            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
            total_image_textual_questions += 1

        # Process Pure_Text_Questions
        for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                continue  # Skip few-shot question

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for few_shot in few_shot_text_prompts:
                    few_shot_question = few_shot["Question"]
                    few_shot_options = few_shot["Options"]
                    few_shot_answer = few_shot["Correct Answer"]
                    few_shot_prompt += (
                        f"USER:\n"
                        f"Question: {few_shot_question}\n"
                        f"A: {few_shot_options['A']}\n"
                        f"B: {few_shot_options['B']}\n"
                        f"C: {few_shot_options['C']}\n"
                        f"D: {few_shot_options['D']}\n"
                        f"Correct Answer: {few_shot_answer}\n"
                    )

            prompt = (
                f"{few_shot_prompt}USER:\n{question_with_options}\n"
                f"Just give ONE letter representing the answer directly.\nASSISTANT:"
            )


            # Model specific logic
            if "llama" in args.model_id.lower():
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            elif "llava" in args.model_id.lower():
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            elif "qwen" in args.model_id.lower():
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
            generated_text=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
            assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
            assistant_response+=" "
            print("Prompt is : ","**************",prompt,"**************")
            print("Generated text is : ","**************",assistant_response,"**************")
            predicted_answer = assistant_response[0].upper()
            print("Predicted answer is : ","**************",predicted_answer,"**************")
            if predicted_answer == correct_answer:
                total_pure_text_correct += 1
            total_pure_text_questions += 1

            print("Generate Text: ", generated_text)
            print("Model Answer: ", predicted_answer)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", predicted_answer == correct_answer)
            print("\n")

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy
    }


# def evaluate_fill_in_the_blank(json_files, image_folder, processor, tokenizer, model, args, id_list_file=None, mode="default"):
def evaluate_fill_in_the_blank(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    """
    Evaluate classification task with/without few-shot samples based on the mode.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        few_shot_parquet_file: Path to the Parquet file containing few-shot examples.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Mode that controls how few-shot samples are handled ('forget', 'retain_share', or others). Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print(
        "################################## Fill-in-the-blank Task Starts ##############################################")

    print(f"Evaluating {mode} Mode")
    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    # Randomly select few-shot examples based on the model
    selected_ids = random.sample(id_list, few_shots_num)

    print(f"Selected few-shot IDs: {selected_ids}")

    few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
    few_shot_images = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}  # Dictionary to track few-shot question indices

    # Load few-shot examples based on selected_ids from the few-shot Parquet file
    few_shot_df = pd.read_parquet(few_shot_parquet_file)
    few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]
    for _, row in few_shot_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Track the indices of few-shot questions for each ID
        few_shot_question_indices[row["ID"]] = {
            "image_textual": [],
            "pure_text": []
        }

        for idx, question_data in enumerate(fill_in_the_blank_questions):
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]
            question_type = question_data["Type"]

            # Add few-shot examples for both question types
            if question_type == "Image_Textual":
                # Prepare the prompt
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                few_shot_image_prompts.append({
                    "Question": question,
                    "Correct Answer": ground_truth
                })
                few_shot_images.append(image)
                # Record the index of this question as few-shot
                few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

            elif question_type == "Pure_Text":
                # Prepare the prompt
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                few_shot_text_prompts.append({
                    "Question": question,
                    "Correct Answer": ground_truth
                })
                # Record the index of this question as few-shot
                few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

    print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
    print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

    # Load evaluation samples
    # Load the test set with multiple Parquet files if mode is "test"
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]

        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each question in Mask_Task and skip if it's a few-shot question
        for idx, question_entry in enumerate(fill_in_the_blank_questions):
            question = question_entry["Question"]
            ground_truth = question_entry["Ground_Truth"]
            question_type = question_entry["Type"]
            question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."

            # Skip if this question was used for few-shot learning
            if row["ID"] in few_shot_question_indices:
                if question_type == "Image_Textual" and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                    continue  # Skip this image-textual question
                elif question_type == "Pure_Text" and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                    continue  # Skip this pure-text question

            # Combine few-shot examples with the current prompt (only if mode requires it)
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test", "retain_celebrity"]:
                if question_type == "Image_Textual":
                    for i, few_shot_image in enumerate(few_shot_images):
                        few_shot_prompt += (f"USER:<image>\n{few_shot_image_prompts[i]['Question']}\n"
                                            f"Correct Answer: {few_shot_image_prompts[i]['Correct Answer']}\n")
                elif question_type == "Pure_Text":
                    for i, few_shot_text in enumerate(few_shot_text_prompts):
                        few_shot_prompt += (f"USER:\n{few_shot_text['Question']}\n"
                                            f"Correct Answer: {few_shot_text['Correct Answer']}\n")

            # Model specific logic
            if "llama" in args.model_id.lower():
                prompt = (f"{few_shot_prompt}USER: "
                      f"<|image|><|begin_of_text|>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                      f"{few_shot_prompt}USER:\n{question}\nASSISTANT:")
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            elif "llava" in args.model_id.lower():
                prompt = (f"{few_shot_prompt}USER: "
                      f"<image>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                      f"{few_shot_prompt}USER:\n{question}\nASSISTANT:")
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            elif "qwen" in args.model_id.lower():
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{few_shot_prompt}\n{question}" if question_type == "Image_Textual" else question},
                            ],
                        }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
            assistant_response=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            print("\n")
            # Evaluate if the generated answer contains the correct ground truth
            if question_type == "Image_Textual":
                if ground_truth.lower() in assistant_response.lower():
                    total_image_textual_correct += 1
                total_image_textual_questions += 1
            elif question_type == "Pure_Text":
                if ground_truth.lower() in assistant_response.lower():
                    total_pure_text_correct += 1
                total_pure_text_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "image_textual_accuracy": image_textual_accuracy,
        "pure_text_accuracy": pure_text_accuracy
    }

def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None):
    """
    Evaluate the generation tasks using the ROUGE and BLEU scores.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        processor: The processor for handling text and images (e.g., from Hugging Face).
        tokenizer: The tokenizer for decoding model outputs.
        model: The model for answering the generation questions.
        args: Arguments object containing model ID and other configurations.
        file_name: Name of the file to save the evaluation results.
        mode: Mode to control which evaluation setup to use. Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary containing average ROUGE and BLEU scores for Image_Textual and Pure_Text questions.
    """
    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    # Load the ID list from the forget Parquet file for filtering if mode is "test"
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # Load all IDs from the Parquet file if no filtering is needed
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Loop through each person's data in the evaluation samples
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        generation_questions = row["Generation_Task"]

        # Randomly select one image if in test mode and multiple images are available
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Process each generation question
        for question_data in generation_questions:
            question_type = question_data["Type"]
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                if "llava" in args.model_id.lower():
                    prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.\nASSISTANT: "
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif "qwen" in args.model_id.lower():
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH."},
                            ],
                        }
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif "llama" in args.model_id.lower():
                    prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                else:
                    raise ValueError("Model ID not supported")
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
                predicted_answer=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)

            else:  # Pure_Text case

                if "llama" in args.model_id.lower():
                    prompt = f"<|begin_of_text|>### Question: {question}\n### Answer:"
                    inputs = processor(text=prompt, return_tensors="pt").to("cuda")
                else:
                    prompt = f"USER: {question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH.\nASSISTANT:"
                    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
                predicted_answer=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)

            # Print debug information
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                # Accumulate scores for Image_Textual questions
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1
            else:
                # Accumulate scores for Pure_Text questions
                total_bleu_text += bleu_score
                total_rouge1_text += rouge_scores['rouge1'].fmeasure
                total_rouge2_text += rouge_scores['rouge2'].fmeasure
                total_rougeL_text += rouge_scores['rougeL'].fmeasure
                total_pure_text_questions += 1

    # Save the results to a JSON file
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{mode}{args.forget_ratio}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate and print average ROUGE and BLEU scores
    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions
        })

    if total_pure_text_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
            "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
            "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
            "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions
        })

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_split_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--few_shot_data', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--celebrity_data', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--forget_ratio', type=int, default=5, help='Path to real person image folder.')
    parser.add_argument('--pretrain', type=bool, default=False, help="load pretrain model")
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
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)
    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    # real_paraquet_file = os.path.join(args.celebrity_data, f"train-00000-of-00001.parquet")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()
    if args.pretrain:
        if "llava" in args.model_id.lower():
            print("Loading LLAVA Pretrained model...")
            # Load LLAVA model and processor
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                # torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif "llama" in args.model_id.lower():
            print("Loading idefics2 Pretrained model...")
            model = MllamaForConditionalGeneration.from_pretrained(
                "HuggingFaceM4/idefics2-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
        elif "qwen" in args.model_id.lower():
            print("Loading Qwen Pretrained model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_id, 
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                local_files_only=True,
                attn_implementation="flash_attention_2",
            )
    else:
        if "llava" in args.model_id.lower():
            print("Loading LLAVA Vanilla model...")
            model = LlavaForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        elif "llama" in args.model_id.lower():
            print("Loading idefics2 Vanilla model...")
            model = MllamaForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        elif "qwen" in args.model_id.lower():
            print("Loading Qwen Pretrained model...")
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

        forget_classification_result = evaluate_classification(parquet_file=forget_parquet_file,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget")
        
        forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=forget_parquet_file,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget")

        forget_generation_result = evaluate_generation(parquet_file=forget_parquet_file,
                                                            processor=processor,
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            args=args,
                                                            mode="forget")
        print("Forget Set Results:")
        print(forget_classification_result)
        print(forget_generation_result)
        print(forget_fill_in_the_blank_result)
        results_data["Forget Set Results"]={
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result
        }
    if "test" in args.eval_list:
        print("### Evaluating Test Set ###")
        test_classification_result = evaluate_classification(parquet_file=args.test_data,
                                                                    few_shot_parquet_file=args.few_shot_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="test",
                                                                    forget_parquet_file=forget_parquet_file)

        test_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.test_data,
                                                                    few_shot_parquet_file=args.few_shot_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="test",
                                                                    forget_parquet_file=forget_parquet_file)

        test_generation_result = evaluate_generation(parquet_file=args.test_data,
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    args=args,
                                                    mode="test",
                                                    forget_parquet_file=forget_parquet_file)
        print("Test Set Results:")
        print(test_fill_in_the_blank_result)
        print(test_classification_result)
        print(test_generation_result)
        results_data["Test Set Results"]= {
            "fill_in_the_blank": test_fill_in_the_blank_result,
            "classification": test_classification_result,
            "generation": test_generation_result,
        }
    if "retain" in args.eval_list:
        print("### Evaluating Retain Shared Set ###")
        retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=retain_parquet_file,
                                                                    few_shot_parquet_file=args.few_shot_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="retain_shared")

        retain_classification_result = evaluate_classification(parquet_file=retain_parquet_file,
                                                            few_shot_parquet_file=args.few_shot_data,
                                                            processor=processor,
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            args=args,
                                                            mode="retain_shared")

        retain_generation_result = evaluate_generation(parquet_file=retain_parquet_file,
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    args=args,
                                                    mode="retain_shared")
        print("Retain Set (shared dataset) Results:")
        print( retain_fill_in_the_blank_result)
        print(retain_classification_result)
        print(retain_generation_result)
        results_data["Retain Set (shared dataset) Results"]= {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result
        }

    if "real" in args.eval_list:
        print("### Evaluating Real Celebrity Set ###")
        real_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.celebrity_data,
                                                                    few_shot_parquet_file=args.few_shot_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="retain_celebrity")

        real_classification_result = evaluate_classification(parquet_file=args.celebrity_data,
                                                            few_shot_parquet_file=args.few_shot_data,
                                                            processor=processor,
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            args=args,
                                                            mode="retain_celebrity")

        real_generation_result = evaluate_generation(parquet_file=args.celebrity_data,
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    args=args,
                                                    mode="retain_celebrity")

        print("Retain Set (real person) Results:")
        print(real_fill_in_the_blank_result)
        print(real_classification_result)
        print(real_generation_result)
        results_data['Retain Set (real person) Results']={
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result
        }

    output_file = f'{args.output_folder}/{args.output_file}_{args.forget_ratio}_final_evaluation_results.json'

    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


