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

OUT_DIR = os.environ.get("OUT_DIR", "/home/betty/onevision_outputs")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "48"))
PER_CLASS_LIMIT = int(os.environ.get("PER_CLASS_LIMIT", "0"))  # 0 = no limit

os.makedirs(OUT_DIR, exist_ok=True)

UNKNOWN_LABEL = "Unknown"

# ======================
# Fine -> Coarse mapping
# 已補 4 個 unmapped:
# Bedding set, Disposable cups, Dinner plate, Sports cup
# ======================
COARSE_MAP = {
    # ======================
    # Food / Drink
    # ======================
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

    # ======================
    # Appliance
    # ======================
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

    # ======================
    # Personal care / beauty / hygiene
    # ======================
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

    # ======================
    # Wearables
    # ======================
    "Diapers": "Wearables",
    "Hat": "Wearables",
    "Jacket": "Wearables",
    "Lingerie": "Wearables",
    "Shoes": "Wearables",
    "Socks": "Wearables",
    "Trousers": "Wearables",
    "Underwear": "Wearables",

    # ======================
    # Baby + Toy
    # ======================
    "Baby carriage": "BabyAndToy",
    "Baby crib": "BabyAndToy",
    "Baby Furniture": "BabyAndToy",
    "Baby handkerchiefs": "BabyAndToy",
    "Baby tableware": "BabyAndToy",
    "Baby washing and nursing supplie": "BabyAndToy",
    "Toys": "BabyAndToy",
    "Badminton": "BabyAndToy",
    "Basketball": "BabyAndToy",
    "Football": "BabyAndToy",
    "Rubber ball": "BabyAndToy",
    "Skate": "BabyAndToy",

    # ======================
    # Household + Kitchen
    # ======================
    "Basin": "HouseholdKitchen",
    "Bedding set": "HouseholdKitchen",
    "Bowl": "HouseholdKitchen",
    "Chopping block": "HouseholdKitchen",
    "Chopsticks": "HouseholdKitchen",
    "Coat hanger": "HouseholdKitchen",
    "Comb": "HouseholdKitchen",
    "Cutter": "HouseholdKitchen",
    "Dinner plate": "HouseholdKitchen",
    "Disposable bag": "HouseholdKitchen",
    "Disposable cups": "HouseholdKitchen",
    "Draw bar box": "HouseholdKitchen",
    "Food box": "HouseholdKitchen",
    "Forks": "HouseholdKitchen",
    "Fresh-keeping film": "HouseholdKitchen",
    "Knapsack": "HouseholdKitchen",
    "Knives": "HouseholdKitchen",
    "Mug": "HouseholdKitchen",
    "Notebook": "HouseholdKitchen",
    "Pen": "HouseholdKitchen",
    "Pencil case": "HouseholdKitchen",
    "Pot shovel": "HouseholdKitchen",
    "Socket": "HouseholdKitchen",
    "Soup ladle": "HouseholdKitchen",
    "Spoon": "HouseholdKitchen",
    "Sports cup": "HouseholdKitchen",
    "Stool": "HouseholdKitchen",
    "Storage bottle": "HouseholdKitchen",
    "Storage box": "HouseholdKitchen",
    "Thermos bottle": "HouseholdKitchen",
    "Trash": "HouseholdKitchen",
}

COARSE_LABELS = sorted(set(COARSE_MAP.values()))

# ======================
# Build reverse map: coarse -> fine labels
# ======================
FINE_BY_COARSE = defaultdict(list)
for fine_label, coarse_label in COARSE_MAP.items():
    FINE_BY_COARSE[coarse_label].append(fine_label)

for coarse_label in FINE_BY_COARSE:
    FINE_BY_COARSE[coarse_label] = sorted(FINE_BY_COARSE[coarse_label])

# ======================
# Prompt builders
# ======================
def build_coarse_prompt():
    label_list = "\n".join([f"- {x}" for x in COARSE_LABELS])

    return (
        "You are a retail product coarse classifier.\n"
        "Given a product image, choose EXACTLY ONE coarse label from the list.\n\n"

        "Compare categories using visible evidence only:\n"
        "- object type\n"
        "- package form\n"
        "- material\n"
        "- size and structure\n"
        "- likely function\n\n"

        "Key distinctions:\n"
        "- FoodDrink: edible food, snacks, beverages, frozen food, condiments, powders.\n"
        "- Appliance: powered electrical devices or household machines.\n"
        "- PersonalCare: hygiene, skincare, beauty, haircare, oral care, medical care.\n"
        "- Wearables: wearable items such as clothes, shoes, hats, socks, underwear, diapers.\n"
        "- BabyAndToy: baby-specific products, stroller, crib, baby care items, baby toys, sports/play items.\n"
        "- HouseholdKitchen: bowls, plates, utensils, containers, bottles, bags, storage items, stationery, household utility goods.\n\n"

        "Important exclusions:\n"
        "- If it is edible or drinkable, choose FoodDrink.\n"
        "- If it is a powered device or machine, choose Appliance.\n"
        "- If it is a hygiene, skincare, beauty, oral care, or medical-use item, choose PersonalCare.\n"
        "- If it is a wearable fabric or footwear item, choose Wearables.\n"
        "- If it is mainly for babies or play/sports use, choose BabyAndToy.\n"
        "- If it is a bowl, cup, utensil, bottle, container, bag, storage item, stationery, hanger, socket, or household utility item, choose HouseholdKitchen.\n\n"

        "Do not explain.\n"
        "Return ONLY one line in this format:\n"
        "FINAL_LABEL: <one label>\n\n"

        "Label list:\n"
        f"{label_list}\n"
    )

def build_fine_prompt(coarse_label: str):
    fine_labels = FINE_BY_COARSE[coarse_label]
    label_list = "\n".join([f"- {x}" for x in fine_labels])

    return (
        f"You are a retail product fine classifier.\n"
        f"The product has already been classified into the coarse category: {coarse_label}.\n\n"

        f"Your task is to compare the candidate fine labels using only visible evidence.\n"
        f"Focus on:\n"
        f"- package type\n"
        f"- shape and structure\n"
        f"- material\n"
        f"- container form\n"
        f"- likely function or usage clue\n\n"

        f"Important:\n"
        f"- Choose EXACTLY ONE fine label from the list.\n"
        f"- Do not explain.\n"
        f"- Do not output candidates.\n"
        f"- Do not output anything except the final answer.\n\n"

        f"Return ONLY one line in this format:\n"
        f"FINAL_LABEL: <one label>\n\n"

        f"Fine label list:\n"
        f"{label_list}\n"
    )

COARSE_PROMPT = build_coarse_prompt()

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

def normalize_label(pred: str, valid_labels):
    pred = clean_prediction_text(pred)

    if pred in valid_labels or pred == UNKNOWN_LABEL:
        return pred

    lower_map = {x.lower(): x for x in valid_labels}
    if pred.lower() in lower_map:
        return lower_map[pred.lower()]
    if pred.lower() == UNKNOWN_LABEL.lower():
        return UNKNOWN_LABEL

    for x in sorted(valid_labels, key=len, reverse=True):
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
correct_fine = 0
correct_coarse = 0
parse_fail_coarse = 0
parse_fail_fine = 0
unmapped_count = 0

wrong_rows = []
confusion_fine = defaultdict(lambda: defaultdict(int))
confusion_coarse = defaultdict(lambda: defaultdict(int))

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

        if not img_name.lower().endswith((".png",".jpg",".jpeg",".webp")):
            continue

        img_path = os.path.join(folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print("Skip bad image:", img_path)
            continue

        total += 1

        # -------- Stage 1: coarse --------
        raw_coarse = generate(model, processor, image, COARSE_PROMPT)
        pred_coarse = normalize_label(raw_coarse, COARSE_LABELS)

        if pred_coarse == UNKNOWN_LABEL:
            parse_fail_coarse += 1

        confusion_coarse[gt_coarse][pred_coarse] += 1

        if pred_coarse == gt_coarse:
            correct_coarse += 1

        # -------- Stage 2: fine --------
        # 用模型預測到的 coarse group 做下一階段
        # 若 coarse fail，就 fallback 用 GT coarse，避免整個 list 壞掉
        stage2_group = pred_coarse if pred_coarse in FINE_BY_COARSE else gt_coarse

        fine_prompt = build_fine_prompt(stage2_group)
        candidate_fine_labels = FINE_BY_COARSE[stage2_group]

        raw_fine = generate(model, processor, image, fine_prompt)
        pred_fine = normalize_label(raw_fine, candidate_fine_labels)

        if pred_fine == UNKNOWN_LABEL:
            parse_fail_fine += 1

        confusion_fine[gt_label][pred_fine] += 1

        if pred_fine == gt_label:
            correct_fine += 1
        else:
            wrong_rows.append([
                img_path,
                gt_label,
                gt_coarse,
                pred_coarse,
                pred_fine,
                raw_coarse[:200].replace("\n", "\\n"),
                raw_fine[:200].replace("\n", "\\n")
            ])

        if total % 20 == 0:
            print(
                f"[{total}] coarse_acc={correct_coarse/total:.4f} "
                f"fine_acc={correct_fine/total:.4f} "
                f"coarse_fail={parse_fail_coarse} fine_fail={parse_fail_fine}"
            )

        if PER_CLASS_LIMIT > 0 and count_this_class >= PER_CLASS_LIMIT:
            break

        count_this_class += 1

coarse_acc = correct_coarse / total if total > 0 else 0.0
fine_acc = correct_fine / total if total > 0 else 0.0

# ======================
# Save outputs
# ======================
wrong_path = os.path.join(OUT_DIR, "wrong_cases_hierarchical_6class.csv")
with open(wrong_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "img_path",
        "gt_label_fine",
        "gt_label_coarse",
        "pred_label_coarse",
        "pred_label_fine",
        "raw_coarse_head",
        "raw_fine_head",
    ])
    writer.writerows(wrong_rows)

conf_coarse_path = os.path.join(OUT_DIR, "confusion_hierarchical_coarse_6class.csv")
with open(conf_coarse_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["GT_coarse", "Pred_coarse", "Count"])
    for gt in sorted(confusion_coarse.keys()):
        for pred, cnt in sorted(confusion_coarse[gt].items(), key=lambda x: -x[1]):
            writer.writerow([gt, pred, cnt])

conf_fine_path = os.path.join(OUT_DIR, "confusion_hierarchical_fine.csv")
with open(conf_fine_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["GT_fine", "Pred_fine", "Count"])
    for gt in sorted(confusion_fine.keys()):
        for pred, cnt in sorted(confusion_fine[gt].items(), key=lambda x: -x[1]):
            writer.writerow([gt, pred, cnt])

summary_path = os.path.join(OUT_DIR, "result_summary_hierarchical_6class.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== RESULT (HIERARCHICAL) =====\n")
    f.write(f"Total                 : {total}\n")
    f.write(f"Correct coarse        : {correct_coarse}\n")
    f.write(f"Correct fine          : {correct_fine}\n")
    f.write(f"Coarse accuracy       : {coarse_acc:.4f}\n")
    f.write(f"Fine accuracy         : {fine_acc:.4f}\n")
    f.write(f"Parse fail coarse     : {parse_fail_coarse}\n")
    f.write(f"Parse fail fine       : {parse_fail_fine}\n")
    f.write(f"Unmapped GT skipped   : {unmapped_count}\n")
    f.write(f"Coarse labels         : {COARSE_LABELS}\n")

print("\n===== RESULT (HIERARCHICAL) =====")
print("Total               :", total)
print("Correct coarse      :", correct_coarse)
print("Correct fine        :", correct_fine)
print("Coarse accuracy     :", f"{coarse_acc:.4f}")
print("Fine accuracy       :", f"{fine_acc:.4f}")
print("Parse fail coarse   :", parse_fail_coarse)
print("Parse fail fine     :", parse_fail_fine)
print("Unmapped GT skipped :", unmapped_count)
print("Saved wrong cases ->", wrong_path)
print("Saved coarse confusion ->", conf_coarse_path)
print("Saved fine confusion ->", conf_fine_path)
print("Saved summary ->", summary_path)