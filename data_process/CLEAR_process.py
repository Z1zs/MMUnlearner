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
import random
random.seed(42)

IMAGE_CAPTION_QUESTIONS = [
    "What can you see in this picture?",
    "Tell me about the content of this image",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]

CAPTION_MODE="CAP"
RECOGNITION_MODE="REC"
NONE_MODE="NONE"

class CLEAR_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, data, mode=CAPTION_MODE):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.data = data
        self.mode=mode
        self.dataset=self.process_dataset()

    def process_dataset(self):
        dataset=[]
        for sample in self.data:

            # Get the image and resize it if necessary
            image = sample.get("image",None)

            # Get the question and answer
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            caption=sample.get("caption","")
            name=sample.get("name","")
            if image and self.mode==NONE_MODE:#pure text mode
                continue
            if question and len(question)>0:#Pure QA mode
                row= {
                    "image": image,
                    "question": question,
                    "answer": answer
                }
            
            elif self.mode==CAPTION_MODE:#VQA caption mode
                row= {
                    "image": image,
                    "question": random.choice(IMAGE_CAPTION_QUESTIONS),
                    "answer": caption
                }

            elif self.mode==RECOGNITION_MODE:#VQA recognition mode
                row= {
                    "image": image,
                    "question": "The name of the person on the image is ",
                    "answer": name
                }

            else:
                ValueError(f"Invalid sample: {sample}!")
            
            dataset.append(row)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        return self.dataset[idx]
        

def train_collate_clear_w_example(examples, processor,device, train_flag):
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
    batch["labels"] = labels
    # print(batch["input_ids"],batch["pixel_values"].shape)
    if train_flag:
        batch={k:v.to(device) for k,v in batch.items()}
        return batch,examples
    else:
        return (v.to(device) for v in batch.values()),examples


def train_collate_clear(examples, processor,device, train_flag):
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
    batch["labels"] = labels
    # print(batch["input_ids"],batch["pixel_values"].shape)
    if train_flag:
        batch={k:v.to(device) for k,v in batch.items()}
        return batch
    else:
        return (v.to(device) for v in batch.values())


def train_collate_clear_ansonly(examples, processor,device, train_flag):
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
        all_special_ids = torch.Tensor(processor.tokenizer.all_special_ids)
        answer_mask=torch.isin(answer_token['input_ids'][0], all_special_ids,invert=True)
        
        answer_token=answer_token['input_ids'][0][answer_mask]
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
    # print("*****************")
    # print("unmasked_text",unmasked_text)
    # print("masked_text",masked_text)
    # print("*****************")
    # exit(0)
    # print(batch["input_ids"],batch["pixel_values"].shape)
    if train_flag:
        batch={k:v.to(device) for k,v in batch.items()}
        return batch
    else:
        return (v.to(device) for v in batch.values())