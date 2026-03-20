import os
import re
import csv
import json
from collections import defaultdict

import torch
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)

# ======================
# Config
# ======================
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf"
)
DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    "/home/betty/datasets/locount_class_samples/locount_class_samples"
)
COARSE_JSON_PATH = os.environ.get(
    "COARSE_JSON_PATH",
    "/mnt/c/Users/User/onevision_next/coarse_descriptions.json"
)

with open(COARSE_JSON_PATH, "r", encoding="utf-8") as f:
    COARSE_DESCRIPTIONS = json.load(f)
OUT_DIR = os.environ.get("OUT_DIR", "/home/betty/onevision_outputs")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
PER_CLASS_LIMIT = int(os.environ.get("PER_CLASS_LIMIT", "0"))  # 0 = no limit

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# Coarse labels
# ======================
COARSE_LABELS = [
    "FoodDrink",
    "Appliance",
    "PersonalCare",
    "ClothingWear",
    "BabyProduct",
    "Kitchenware",
    "HouseholdOther",
    "SportsToy",
]

UNKNOWN_LABEL = "Unknown"

# ======================
# Fine -> Coarse mapping
# 依你目前合併後/原始類別整理
# ======================
COARSE_MAP = {
    # Food / Drink
    "milk powder": "FoodDrink",
    "Biscuits": "FoodDrink",
    "Cake": "FoodDrink",
    "Can": "FoodDrink",
    "Carbonated drinks": "FoodDrink",
    "Chewing gum": "FoodDrink",
    "Chocolates": "FoodDrink",
    "Cocktail": "FoodDrink",
    "Coffee": "FoodDrink",
    "Cooking wine": "FoodDrink",
    "Dairy": "FoodDrink",
    "Dried beans": "FoodDrink",
    "Dried fish": "FoodDrink",
    "Dried meat": "FoodDrink",
    "Fish tofu": "FoodDrink",
    "Flour": "FoodDrink",
    "Ginger Tea": "FoodDrink",
    "Guozhen": "FoodDrink",
    "Herbal tea": "FoodDrink",
    "Hot strips": "FoodDrink",
    "Ice cream": "FoodDrink",
    "Instant noodles": "FoodDrink",
    "Liquor and Spirits": "FoodDrink",
    "Lotus root flour": "FoodDrink",
    "Mixed congee": "FoodDrink",
    "Noodle": "FoodDrink",
    "Oats": "FoodDrink",
    "Pasta": "FoodDrink",
    "Pie": "FoodDrink",
    "Potato chips": "FoodDrink",
    "Quick-frozen dumplings": "FoodDrink",
    "Quick-frozen Tangyuan": "FoodDrink",
    "Quick-frozen Wonton": "FoodDrink",
    "Red wine": "FoodDrink",
    "Rice": "FoodDrink",
 
    "Sauce": "FoodDrink",
    "Sesame paste": "FoodDrink",
    "Sour Plum Soup": "FoodDrink",
    "Soy sauce": "FoodDrink",
    "Soymilk": "FoodDrink",
    "Tea": "FoodDrink",
    "Tea beverage": "FoodDrink",
    "Vinegar": "FoodDrink",
    "Walnut powder": "FoodDrink",

    # Appliance
    "Air conditioner": "Appliance",
    "Air conditioning fan": "Appliance",
    "Desk lamp": "Appliance",
    "Electric fan": "Appliance",
    "Electric frying pan": "Appliance",
    "Electric Hot pot": "Appliance",
    "Electric iron": "Appliance",
    "Electric kettle": "Appliance",
    "Electric steaming pan": "Appliance",
    "Electromagnetic furnace": "Appliance",
    "Hair drier": "Appliance",
    "Juicer": "Appliance",
    "Microwave Oven": "Appliance",
    "Refrigerator": "Appliance",
    "Rice cooker": "Appliance",
    "Television": "Appliance",
    "Washing machine": "Appliance",

    # Personal care / beauty / hygiene
    "Band aid": "PersonalCare",
    "Bath lotion": "PersonalCare",
    "Care Kit": "PersonalCare",
    "Cotton swab": "PersonalCare",
    "Emulsion": "PersonalCare",
    "Facial Cleanser": "PersonalCare",
    "Facial mask": "PersonalCare",
    "Hair conditioner": "PersonalCare",
    "Hair dye": "PersonalCare",
    "Hair gel": "PersonalCare",
    "Makeup tools": "PersonalCare",
    "Mouth wash": "PersonalCare",
    "Razor": "PersonalCare",
    "Shampoo": "PersonalCare",
    "Skin care set": "PersonalCare",
    "Soap": "PersonalCare",
    "Tampon": "PersonalCare",
    "Toothbrush": "PersonalCare",
    "Toothpaste": "PersonalCare",

    # Clothing / wearables
    "Diapers": "ClothingWear",
    "Hat": "ClothingWear",
    "Jacket": "ClothingWear",
    "Lingerie": "ClothingWear",
    "shoes": "ClothingWear",
    "socks": "ClothingWear",
    "Trousers": "ClothingWear",
    "underwear": "ClothingWear",

    # Baby products
    "Baby carriage": "BabyProduct",
    "Baby crib": "BabyProduct",
    "Baby Furniture": "BabyProduct",
    "Baby handkerchiefs": "BabyProduct",
    "Baby tableware": "BabyProduct",
    "Baby washing and nursing supplie": "BabyProduct",
    "Toys": "BabyProduct",

    # Kitchenware / containers / tableware
    "Basin": "Kitchenware",
    "Bowl": "Kitchenware",
    "Chopping block": "Kitchenware",
    "Chopsticks": "Kitchenware",
    "Dinner plate": "Kitchenware",
    "Disposable cups": "Kitchenware",
    "Food box": "Kitchenware",
    "Forks": "Kitchenware",
    "Fresh-keeping film": "Kitchenware",
    "Knives": "Kitchenware",
    "Mug": "Kitchenware",
    "Pot shovel": "Kitchenware",
    "Soup ladle": "Kitchenware",
    "Spoon": "Kitchenware",
    "Sports cup": "Kitchenware",
    "Storage bottle": "Kitchenware",
    "Storage box": "Kitchenware",
    "Thermos bottle": "Kitchenware",
    "Disposable cups": "Kitchenware",
    "Dinner plate": "Kitchenware",
    "Sports cup": "Kitchenware",

    # Household / storage / other
    "Coat hanger": "HouseholdOther",
    "Comb": "HouseholdOther",
    "Cutter": "HouseholdOther",
    "Disposable bag": "HouseholdOther",
    "Draw bar box": "HouseholdOther",
    "Knapsack": "HouseholdOther",
    "Notebook": "HouseholdOther",
    "Pen": "HouseholdOther",
    "Pencil case": "HouseholdOther",
    "Socket": "HouseholdOther",
    "Stool": "HouseholdOther",
    "Trash": "HouseholdOther",
    "Bedding set": "HouseholdOther",

    # Sports / toy / ball
    "Badminton": "SportsToy",
    "Basketball": "SportsToy",
    "Football": "SportsToy",
    "Rubber ball": "SportsToy",
    "Skate": "SportsToy",
}

# ======================
# Prompt
# ======================
def build_coarse_prompt():
    label_lines = []
    for label in COARSE_LABELS:
        desc = COARSE_DESCRIPTIONS.get(label, "")
        label_lines.append(f"- {label}: {desc}")

    label_list = "\n".join(label_lines + [f"- {UNKNOWN_LABEL}: use this only if the image is too unclear to determine."])

    return (
        "You are a retail product coarse classifier.\n"
        "Given a product image, choose EXACTLY ONE coarse label from the list.\n\n"
        "Use visible evidence such as:\n"
        "- object type\n"
        "- shape\n"
        "- material\n"
        "- package type\n"
        "- likely use or function\n\n"
        "Return ONLY one line in this format:\n"
        "FINAL_LABEL: <one label>\n\n"
        "Coarse label descriptions:\n"
        f"{label_list}\n"
    )

PROMPT = build_coarse_prompt()

# ======================
# Helpers
# ======================
def generate(model, processor, image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    full_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def clean_prediction_text(text: str) -> str:
    text = text.strip()

    if "assistant" in text:
        text = text.split("assistant")[-1].strip()

    m = re.search(r"FINAL_LABEL\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    text = re.sub(r"^[-•\*\s]+", "", text).strip()
    text = re.sub(r"[\.。,，;；:：]+$", "", text).strip()
    text = text.split("\n")[0].strip()

    return text

def normalize_coarse_label(pred: str) -> str:
    pred = clean_prediction_text(pred)

    if pred in COARSE_LABELS or pred == UNKNOWN_LABEL:
        return pred

    lower_map = {x.lower(): x for x in COARSE_LABELS}
    if pred.lower() in lower_map:
        return lower_map[pred.lower()]
    if pred.lower() == UNKNOWN_LABEL.lower():
        return UNKNOWN_LABEL

    for x in sorted(COARSE_LABELS, key=len, reverse=True):
        if x.lower() in pred.lower():
            return x

    return UNKNOWN_LABEL

def fine_to_coarse(gt_label: str) -> str:
    return COARSE_MAP.get(gt_label, "UNMAPPED")

# ======================
# Load model
# ======================
print("MODEL_ID:", MODEL_ID)
print("DATASET_ROOT:", DATASET_ROOT)
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)

print("Loading model (4-bit NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    local_files_only=True,
)
model.eval()

# ======================
# Evaluate
# ======================
total = 0
correct = 0
parse_fail = 0
unmapped_count = 0

wrong_rows = []
confusion = defaultdict(lambda: defaultdict(int))

for gt_label in sorted(os.listdir(DATASET_ROOT)):
    folder = os.path.join(DATASET_ROOT, gt_label)

    if not os.path.isdir(folder):
        continue

    gt_coarse = fine_to_coarse(gt_label)
    if gt_coarse == "UNMAPPED":
        unmapped_count += 1
        print("UNMAPPED GT:", gt_label)
        continue

    count_this_class = 0

    for img_name in sorted(os.listdir(folder)):
        if "Zone.Identifier" in img_name:
            continue
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        img_path = os.path.join(folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print("Skip bad image:", img_path)
            continue

        raw_pred = generate(model, processor, image, PROMPT)
        pred_coarse = normalize_coarse_label(raw_pred)

        total += 1
        count_this_class += 1

        if pred_coarse == UNKNOWN_LABEL:
            parse_fail += 1

        confusion[gt_coarse][pred_coarse] += 1

        if pred_coarse == gt_coarse:
            correct += 1
        else:
            wrong_rows.append([
                img_path,
                gt_label,
                gt_coarse,
                pred_coarse,
                raw_pred[:300].replace("\n", "\\n")
            ])

        if total % 20 == 0:
            print(f"[{total}] acc={correct/total:.4f} parse_fail={parse_fail}")

        if PER_CLASS_LIMIT > 0 and count_this_class >= PER_CLASS_LIMIT:
            break

acc = correct / total if total > 0 else 0.0

# ======================
# Save outputs
# ======================
wrong_path = os.path.join(OUT_DIR, "wrong_cases_coarse.csv")
with open(wrong_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["img_path", "gt_label_fine", "gt_label_coarse", "pred_label_coarse", "raw_output_head"])
    writer.writerows(wrong_rows)

conf_path = os.path.join(OUT_DIR, "confusion_coarse.csv")
with open(conf_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["GT_coarse", "Pred_coarse", "Count"])
    for gt in sorted(confusion.keys()):
        for pred, cnt in sorted(confusion[gt].items(), key=lambda x: -x[1]):
            writer.writerow([gt, pred, cnt])

summary_path = os.path.join(OUT_DIR, "result_summary_coarse.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== RESULT (COARSE) =====\n")
    f.write(f"Total      : {total}\n")
    f.write(f"Correct    : {correct}\n")
    f.write(f"Accuracy   : {acc:.4f}\n")
    f.write(f"Parse fail : {parse_fail}\n")
    f.write(f"Unmapped GT folders skipped : {unmapped_count}\n")
    f.write(f"wrong_cases_coarse.csv -> {wrong_path}\n")
    f.write(f"confusion_coarse.csv -> {conf_path}\n")

print("\n===== RESULT (COARSE) =====")
print("Total      :", total)
print("Correct    :", correct)
print("Accuracy   :", f"{acc:.4f}")
print("Parse fail :", parse_fail)
print("Unmapped GT folders skipped :", unmapped_count)
print("Saved wrong cases ->", wrong_path)
print("Saved confusion ->", conf_path)
print("Saved summary ->", summary_path)