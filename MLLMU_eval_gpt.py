import sys
sys.path.append(('../'))
sys.path.append(('../../'))
import os
import openai
from openai import OpenAI
import json
import base64
import requests
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
import json
import requests
import openai
from openai import OpenAI

def evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="generation"):
    # Custom prompt for factuality evaluation
    prompt = f"""
        You will be provided with two types of questions: generation questions and description questions.
        For each, you will evaluate the **factuality** of the "generated_answer" or "generated_description" 
        against the "ground_truth" or "ground_truth_description" respectively. Your task is to assess how well 
        the generated response aligns with the factual content of the ground truth and assign a **factuality score** 
        from 1 to 10 based on the following criteria:

        1. **Factuality (core importance)**:
        - **10-9:** The generated response is fully factually correct and has the same meaning as the ground truth, even if phrased differently.
        - **8-7:** The response is mostly correct but may be missing minor details or contain slightly less important deviations.
        - **6-5:** The response is partially correct but has a noticeable factual error or significant missing information.
        - **4-3:** The response has major factual errors or lacks crucial elements of the ground truth.
        - **2-1:** The response is nonsensical, completely incorrect, or irrelevant.

        2. **Relevance and Detail**:
        - More detail does not always improve the score; added details should be factually relevant.
        - If the generated response contains excessive or irrelevant details (e.g., adding personal information when only appearance is requested), lower the score accordingly.

        ### Task Type: {task_type.capitalize()}
        - **Image ID**: {image_id}
        - **Question**: {question}
        - **Generated Answer**: {generated_answer}
        - **Ground Truth**: {ground_truth}

        Please evaluate the factuality of the generated response based on the rubric above, and return a score (1-10) along with a short justification.
        Example Output:
        {{
            "Factuality Score": [Insert score from 1-10],
            "Justification": "[Optional] Provide a brief justification explaining why the factuality score was assigned."
        }}
    """

    # Call the OpenAI API to evaluate factuality
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at evaluating the factuality of responses."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 700,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    evaluation_result = response.json()['choices'][0]['message']['content']

    print(evaluation_result)
    return evaluation_result

def process_generation_questions(generation_questions, output_data, generation_scores):
    # Loop through all the generation questions
    for question_data in generation_questions:
        image_id = question_data.get("image_id")
        question = question_data.get("question")
        generated_answer = question_data.get("generated_answer")
        ground_truth = question_data.get("ground_truth")

        # Evaluate factuality for generation questions
        evaluation = evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="generation")

        # Extract factuality score and justification from the evaluation result
        factuality_score, justification = extract_factuality_score_and_justification(evaluation)

        if factuality_score is not None:
            generation_scores.append(factuality_score)

        # Append the results to the output data
        output_data.append({
            "Task Type": "Generation",
            "Image_ID": image_id,
            "Question": question,
            "Factuality Score": factuality_score,
            "Justification": justification
        })

def process_description_questions(description_questions, output_data, description_scores):
    # Loop through all the description questions
    for description_data in description_questions:
        image_id = description_data.get("image_id")
        question = description_data.get("description_question")
        generated_answer = description_data.get("generated_description")
        ground_truth = description_data.get("ground_truth_description")

        # Evaluate factuality for description questions
        evaluation = evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="description")

        # Extract factuality score and justification from the evaluation result
        factuality_score, justification = extract_factuality_score_and_justification(evaluation)

        if factuality_score is not None:
            description_scores.append(factuality_score)

        # Append the results to the output data
        output_data.append({
            "Task Type": "Description",
            "Image_ID": image_id,
            "Question": question,
            "Factuality Score": factuality_score,
            "Justification": justification
        })

def extract_factuality_score_and_justification(evaluation_result):
    # Extract score and justification from the evaluation result
    try:
        # Find the line that contains "Factuality Score"
        score_line = [line for line in evaluation_result.split('\n') if "Factuality Score" in line][0]
        # Extract the score and remove commas
        score = score_line.split(':')[-1].strip().replace(',', '')

        # Find the line that contains "Justification"
        justification_line = [line for line in evaluation_result.split('\n') if "Justification" in line][0]
        justification = justification_line.split(':', 1)[-1].strip()

        return int(score), justification
    except Exception as e:
        print(f"Error extracting factuality score and justification: {e}")
        return None, None

def evaluate_factuality_from_json(json_file_path, output_folder):
    # Read the input JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Prepare the output data list
    output_data = []
    generation_scores = []
    description_scores = []

    # Process Generation Questions
    generation_questions = data.get("Generation_Questions", [])
    process_generation_questions(generation_questions, output_data, generation_scores)

    # Process Description Questions
    description_questions = data.get("Description_Questions", [])
    process_description_questions(description_questions, output_data, description_scores)

    # Calculate average scores
    avg_generation_score = sum(generation_scores) / len(generation_scores) if generation_scores else 0
    avg_description_score = sum(description_scores) / len(description_scores) if description_scores else 0

    # Append average scores to the output data
    output_data.append({
        "Average Generation Factuality Score": avg_generation_score,
        "Average Description Factuality Score": avg_description_score
    })

    # Define output filename with the new name
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_factuality_score.json")

    # Save the results to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as output_f:
        json.dump(output_data, output_f, indent=4)

    print(f"Factuality evaluation results saved to: {output_file}")

def count_evaluated_folders(input_folder, output_folder):
    total_folders = 0
    evaluated_folders = 0

    # Iterate through all subdirectories in the input folder
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            total_folders += 1

            # Get all JSON files in the directory
            json_files = [f for f in os.listdir(subdir_path) if f.endswith(".json")]

            # Check if all the relevant files in this folder have been processed
            all_processed = True
            for filename in json_files:
                if filename.startswith(("forget", "retain_celebrity", "retain_shared", "test")) and "_factuality_score" not in filename:
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(output_folder, f"{base_name}_factuality_score.json")

                    if not os.path.exists(output_file):
                        all_processed = False
                        break

            # If all files are processed, count this folder as evaluated
            if all_processed:
                evaluated_folders += 1

    # Report the number of evaluated folders
    print(f"{evaluated_folders}/{total_folders} folders evaluated.")
def process_all_files_in_folder(input_folder, output_folder):
    # List all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file starts with the specified prefixes and ends with ".json"
        # Also, skip files that already contain "_factuality_score" to avoid reprocessing output files
        if (filename.startswith(("forget", "retain_celebrity", "retain_shared", "test")) and
            filename.endswith(".json") and
            "_factuality_score" not in filename):  # Skip processed files

            json_file_path = os.path.join(input_folder, filename)

            # Check if the output file for this JSON file already exists
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"{base_name}_factuality_score.json")

            if os.path.exists(output_file):
                print(f"Skipping {json_file_path}, already evaluated.")
                continue

            print(f"Processing file: {json_file_path}")
            evaluate_factuality_from_json(json_file_path, output_folder)

def process_all_folders_in_eval_result(root_folder):
    # Iterate through all subdirectories in the root eval_result folder
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            print(f"Processing folder: {subdir_path}")
            process_all_files_in_folder(subdir_path, subdir_path)  # Use the same folder for input and output

def run_evaluation(input_folder):
    # Check if the folder contains JSON files directly
    contains_json_files = any(
        filename.startswith(("forget", "retain_celebrity", "retain_shared", "test")) and filename.endswith(".json")
        for filename in os.listdir(input_folder)
    )

    if contains_json_files:
        # If the folder contains JSON files directly, process the folder
        print(f"Processing a single folder: {input_folder}")
        process_all_files_in_folder(input_folder, input_folder)
    else:
        # If the folder does not contain JSON files directly, process all subdirectories
        print(f"Processing nested folders under: {input_folder}")
        process_all_folders_in_eval_result(input_folder)

input_folder = "../eval_result"
count_evaluated_folders(input_folder, input_folder)
run_evaluation(input_folder)

