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
from torch.utils.data import DataLoader

class LLAVA_multimodal_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

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

                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer
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

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }
        


class LLAVA_customized_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

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
                if "is" not in question:
                    continue
                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer
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

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


    
class Vanilla_LLaVA_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

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

                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer
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

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


def train_collate_fn_llava(examples, processor, train_flag):
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        images.append(image)

        # Construct prompt with question and answer
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        texts.append(prompt)

    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")

    # Process the batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    if train_flag:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]

def train_collate_fn_mllmu(examples, processor, device,train_flag):
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
        images=images if len(images)>0 else None,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.image_token] = -100
    batch["labels"] = labels
    # print(batch["input_ids"],batch["pixel_values"].shape)
    if train_flag:
        batch={k:v.to(device) for k,v in batch.items()}
        return batch
    else:
        return (v.to(device) for v in batch.values())


def train_collate_mllmu_ansonly(examples, processor, device, train_flag):
    images = []
    texts = []
    answer_ids=[]
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

        answer_token = processor.tokenizer(answer, return_tensors="pt")
        answer_token=answer_token['input_ids'][0][1:]
        # print(answer_token)
        answer_text = processor.decode(answer_token, skip_special_tokens=False)
        # print(answer_text)

        answer_ids.append(answer_token)
        if "print_flg" in example:
            print(text,image)
    if len(texts) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")

    # Process the batch
    batch = processor(
        text=texts,
        images=images if len(images)>0 else None,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    for label,answer_id in zip(labels,answer_ids):
        res=False
        for idx in range(len(label) - len(answer_id) + 1):
            if torch.equal(label[idx: idx + len(answer_id)],answer_id):
                res = True
                # element other than answer_id should be masked
                label[:idx] = -100
                label[idx + len(answer_id):] = -100
                break
        if not res:
            ValueError("Answer not found in the input_ids")

    # labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    # unmasked_ids=labels.clone()
    # unmasked_ids[unmasked_ids==-100]=processor.tokenizer.pad_token_id
    # unmasked_text = processor.decode(batch['input_ids'][0], skip_special_tokens=False)
    # masked_text = processor.decode(unmasked_ids[0], skip_special_tokens=False)
    # print("unmasked_text",unmasked_text)
    # print("masked_text",masked_text)
    # exit(0)
    # print(batch["input_ids"],batch["pixel_values"].shape)
    if train_flag:
        batch={k:v for k,v in batch.items()}
        return batch
    else:
        return (v for v in batch.values())
