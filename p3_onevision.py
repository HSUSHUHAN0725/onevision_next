import os
import re
import json
import csv
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, UnidentifiedImageError

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf"
)

DATASET_ROOT = "/home/betty/datasets/locount_class_samples/locount_class_samples"

OUTPUT_DIR = "/home/betty/onevision_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# load label descriptions
# -----------------------------
JSON_PATH = JSON_PATH = "/mnt/c/Users/User/onevision_next/label_descriptions.json"
with open(JSON_PATH) as f:
    LABEL_DESCRIPTIONS = json.load(f)

CATEGORIES = list(LABEL_DESCRIPTIONS.keys())


# -----------------------------
# load embedding model
# -----------------------------
print("Loading embedding model...")
retriever = SentenceTransformer("all-MiniLM-L6-v2")

label_texts = [
    f"{k}: {v}" for k, v in LABEL_DESCRIPTIONS.items()
]

label_embeddings = retriever.encode(label_texts)


# -----------------------------
# load LLaVA
# -----------------------------
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model (4bit)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)


# -----------------------------
# helper
# -----------------------------
def generate(prompt, image):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=128
    )

    text = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return text


# -----------------------------
# visual description
# -----------------------------
def get_visual_description(image):

    prompt = (
        "Describe the visual appearance of the product in the image. "
        "Focus on shape, material, packaging, and color. "
        "Do not output category names."
    )

    text = generate(prompt, image)

    if "assistant" in text:
        text = text.split("assistant")[-1]

    return text.strip()


# -----------------------------
# retrieval
# -----------------------------
def retrieve_labels(description, topk=5):

    query = retriever.encode([description])

    scores = cosine_similarity(query, label_embeddings)[0]

    top_ids = np.argsort(scores)[::-1][:topk]

    return [CATEGORIES[i] for i in top_ids]


# -----------------------------
# choose final label
# -----------------------------
def choose_label(image, candidates):

    label_list = "\n".join([f"- {c}" for c in candidates])

    prompt = f"""
You are a product classifier.

Choose the best label from the list.

Return ONLY the label text.

Labels:
{label_list}
"""

    text = generate(prompt, image)

    if "assistant" in text:
        text = text.split("assistant")[-1]

    text = text.strip()

    return text


# -----------------------------
# evaluation
# -----------------------------
total = 0
correct = 0

wrong_cases = []

for gt_label in os.listdir(DATASET_ROOT):

    folder = os.path.join(DATASET_ROOT, gt_label)

    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):

        # 只處理 PNG
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder, img_name)

        # 安全開圖（避免壞圖 crash）
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print("Skip bad image:", img_path)
            continue

        # Step1 visual description
        desc = get_visual_description(image)

        # Step2 retrieval
        candidates = retrieve_labels(desc, 5)

        # Step3 final decision
        pred = choose_label(image, candidates)

        total += 1

        if pred == gt_label:
            correct += 1
        else:
            wrong_cases.append([
                img_path,
                gt_label,
                pred,
                desc
            ])

        if total % 20 == 0:
            print(f"[{total}] acc={correct/total:.4f}")


# -----------------------------
# save results
# -----------------------------
acc = correct / total

print("\n===== RESULT =====")
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", acc)


with open(
    f"{OUTPUT_DIR}/wrong_cases_retrieval.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "img_path",
        "gt_label",
        "pred_label",
        "visual_description"
    ])

    writer.writerows(wrong_cases)