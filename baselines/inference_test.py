from transformers import LlavaForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration

import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
model_path="models/Llama-3.2-11B-Vision-Instruct/Llama-3.2-11B-Vision-Instruct"
model_id = "models/llava-1.5-7b-hf"
model = MllamaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
).to("cuda")
print(model_path)
processor = AutoProcessor.from_pretrained(model_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))