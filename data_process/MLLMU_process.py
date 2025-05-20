import pandas as pd
import copy
import json
from typing import Any, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

K_TYPE="Knowledge"
P_TYPE="Perception"

class MLLMU_manifold_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame, full_set: pd.DataFrame, mainfold_type: str = None,  target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.full_set=full_set
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        self.mainfold_type = mainfold_type
        self.question_set = defaultdict(int)
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def get_bio(self,pid):
        pdf = self.full_set[self.full_set['ID']==str(pid)].to_dict()
        try:
            bio_str=list(pdf['biography'].values())[0]
            bio = json.loads(bio_str)  # Using json.loads to parse JSON safely
        except json.JSONDecodeError as e:
            print(f"Error decoding biography at index {pid}: {e}")
        return bio
        
    
    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # Extract the bytes from the 'image' dictionary
            image_data = row['image'].get('bytes')  # Access the image bytes

            # Convert the image bytes to a PIL Image
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            # Safely load metadata as JSON
            try:
                metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue
            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                person_ID = qa_pair.get("ID", "")
                if person_ID:
                    bio=self.get_bio(person_ID)
                else:
                    # print("No ID found: ",qa_pair)
                    continue

                if self.mainfold_type!=P_TYPE:
                    pass
                elif "name" in question:#retain appearance
                    question=bio['appearance_detail']['question']
                    answer=bio['appearance_detail']['answer']
                elif "gender" in question:#retain gender
                    pass
                else:
                    question=""
                if question and answer and person_ID:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer,
                        "person_ID": person_ID,
                        "biography":bio,
                    })
        # print(flattened_data)
        return flattened_data
    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]
        
        name=sample['biography']['Name']
        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if self.mainfold_type is None:
            pass
        elif self.mainfold_type==P_TYPE:
            pass #process in flatten
        elif self.mainfold_type==K_TYPE:
            pron_list=["this person","the person","this individual","the individual"]
            if any(p in question for p in pron_list):
                image=None
                question=question.replace("this person",name).replace("the person",name).replace("this individual",name).replace("the individual",name)
            else:
                pass

        print("Mainfold type: ",self.mainfold_type, " ; Image shape: ",image," ; Question: ",question," ; Anwer: ",answer)
        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        # print(question,tokenized_question)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)
        # print(answer,tokenized_answer)
        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }

class MLLMU_text_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()
        
    
    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            classification_questions = row["Classification_Task"]
            for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
                question = question_data["Question"]
                options = question_data["Options"]
                correct_answer = question_data["Correct_Answer"]
                ground_truth=options[correct_answer ]
                flattened_data.append({
                        "image": None,
                        "question": question,
                        "answer": ground_truth,
                })
        return flattened_data
    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]
        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        # print(question,tokenized_question,image)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)
        # print(answer,tokenized_answer)
        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer,
            "print_flg":True
        }



def train_collate_fn_llava_new(examples, processor, train_flag):
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')

        if image is None:
            user_content=[
                {"type": "text", "text": question}
            ]
        else:
            user_content=[
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            images.append(image)
        # Construct prompt with question and answer
        messages = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        if "print_flg" in example:
            print(text,image)
    if len(texts) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")

    # Process the batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    # print(batch["input_ids"],batch["pixel_values"].shape)
    if train_flag:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]