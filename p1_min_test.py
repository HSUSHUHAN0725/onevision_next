import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "lmms-lab/llava-onevision-qwen2-7b-ov"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    load_in_8bit=True
)

# 用更小的圖（224 -> 128）
image = Image.new("RGB", (128, 128), color="white")

prompt = (
    "<|im_start|>user\n"
    "<image>\n"
    "What is this?\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to("cuda", torch.float16) for k, v in inputs.items()}

print("Generating...")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=None, top_p=None, top_k=None)

print(processor.batch_decode(out, skip_special_tokens=True)[0])
