import os
import re
import csv
import json
import argparse
from collections import defaultdict
from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# 1. Terminal 參數與 Config
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0, help="每個類別跑幾張")
parser.add_argument("--total_limit", type=int, default=0, help="總共跑幾張就停止")
args = parser.parse_args()

MODEL_ID = os.environ.get("MODEL_ID", "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf")
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/home/betty/datasets/locount_class_samples/locount_class_samples")
OUT_DIR = os.environ.get("OUT_DIR", "/home/betty/onevision_outputs")
COARSE_JSON_PATH = os.environ.get("COARSE_JSON_PATH", "/mnt/c/Users/User/onevision_next/coarse_descriptions.json")

MAX_NEW_TOKENS_DESC = 96
MAX_NEW_TOKENS_FINAL = 48
PER_CLASS_LIMIT = args.limit
TOTAL_LIMIT = args.total_limit
UNKNOWN_LABEL = "Unknown"

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# Fine -> Coarse mapping (8-class version)
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
    "Rise": "FoodDrink",
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

    # PersonalCare
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

    # ClothingWear
    "Diapers": "ClothingWear",
    "Hat": "ClothingWear",
    "Jacket": "ClothingWear",
    "Lingerie": "ClothingWear",
    "Shoes": "ClothingWear",
    "Socks": "ClothingWear",
    "Trousers": "ClothingWear",
    "Underwear": "ClothingWear",

    # BabyProduct
    "Baby carriage": "BabyProduct",
    "Baby crib": "BabyProduct",
    "Baby Furniture": "BabyProduct",
    "Baby handkerchiefs": "BabyProduct",
    "Baby tableware": "BabyProduct",
    "Baby washing and nursing supplie": "BabyProduct",
    "Toys": "BabyProduct",

    # Kitchenware
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

    # HouseholdOther
    "Bedding set": "HouseholdOther",
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

    # Sports
    "Badminton": "Sports",
    "Basketball": "Sports",
    "Football": "Sports",
    "Rubber ball": "Sports",
    "Skate": "Sports",
}

ALL_FINE_LABELS = list(COARSE_MAP.keys())

# ======================
# 3. 初始化模型與細類別索引
# ======================
print("Loading Models & Building Fine-grained Index...")
retriever = SentenceTransformer("BAAI/bge-small-en-v1.5")
fine_embeddings = retriever.encode(ALL_FINE_LABELS, normalize_embeddings=True)
tokenized_fine_corpus = [label.lower().split() for label in ALL_FINE_LABELS]
bm25_fine = BM25Okapi(tokenized_fine_corpus)

processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.float16, local_files_only=True,
)
model.eval()

# ======================
# 4. 功能函數
# ======================
def web_search(query):
    try:
        # 清理一下 query，只取前 50 個字，避免過長導致搜尋出錯
        clean_q = re.sub(r'\d+\.', '', query).replace('\n', ' ').strip()
        with DDGS() as ddgs:
            # 加上 "product type" 關鍵字幫助精確搜尋
            results = [r['body'] for r in ddgs.text(f"{clean_q[:50]} product type", max_results=2)]
            return " ".join(results)
    except Exception as e:
        # 如果搜尋失敗，回傳空字串，確保程式不會中斷
        return ""
        
def generate(image, prompt, max_new_tokens):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    full_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=full_prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def clean_prediction_text(text: str) -> str:
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    m = re.search(r"FINAL_LABEL\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    return text.split("\n")[0].strip().strip(".*-• ")

def build_structured_attr_prompt():
    return (
        "Extract only visible product attributes. Return short phrases only.\n"
        "Include: 1. object type, 2. package type, 3. material, 4. shape, 5. powered/non-powered, "
        "6. wearable, 7. edible/drinkable, 8. baby-related, 9. function.\n"
        "Do NOT output category names.\n"
    )

def retrieve_top_fine_labels(structured_desc, k=3):
    # 這裡不加搜尋結果，保持檢索的純淨度，避免正確答案被擠出 Top-3
    query = structured_desc.lower()
    q_vec = retriever.encode([query], normalize_embeddings=True)
    v_scores = cosine_similarity(q_vec, fine_embeddings)[0]
    
    t_query = query.split()
    b_scores = bm25_fine.get_scores(t_query)
    max_b = np.max(b_scores)
    b_scores = b_scores / max_b if max_b > 0 else b_scores
    
    final_scores = 0.6 * v_scores + 0.4 * b_scores
    top_ids = np.argsort(final_scores)[::-1][:k]
    return [ALL_FINE_LABELS[i] for i in top_ids]

def build_fine_selection_prompt(candidates):
    cand_str = "\n".join([f"- {c}" for c in candidates])
    return (
        "You are a retail product expert. Based on the image and its attributes, "
        "which of the following labels best describes the product?\n\n"
        f"{cand_str}\n\n"
        "Return ONLY the label name from the list above.\n"
        "FINAL_LABEL: <label>\n"
    )

# ======================
# 5. 主評估迴圈 (修正字串比對與映射)
# ======================
# ======================
# 5. 主評估迴圈 (整合大類、小類、Top-3 統計)
# ======================
total = 0
correct_coarse = 0  # 粗類別對 (原本的 Acc)
correct_fine = 0    # 細類別對 (新增)
top3_hits = 0       # 檢索命中 (小類名在前三)

wrong_rows = []
ATTR_PROMPT = build_structured_attr_prompt()

print(f"Starting: Total Limit {TOTAL_LIMIT}, Per Class {PER_CLASS_LIMIT}")

for gt_label_name in sorted(os.listdir(DATASET_ROOT)):
    folder = os.path.join(DATASET_ROOT, gt_label_name)
    if not os.path.isdir(folder): continue
    
    gt_coarse = COARSE_MAP.get(gt_label_name, "UNMAPPED")
    if gt_coarse == "UNMAPPED": continue

    class_count = 0
    for img_name in sorted(os.listdir(folder)):
        if "Zone.Identifier" in img_name or not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        img_path = os.path.join(folder, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except: continue

        # --- 開始處理流程 ---
        total += 1
        if TOTAL_LIMIT > 0 and total > TOTAL_LIMIT: break

        # Step 1: 屬性提取
        raw_attr = generate(image, ATTR_PROMPT, MAX_NEW_TOKENS_DESC)
        attr_desc = raw_attr.split("assistant")[-1].strip() if "assistant" in raw_attr else raw_attr.strip()

        # Step 2: 檢索 Top 3
        top3_fine = retrieve_top_fine_labels(attr_desc, k=3)
        
        # [統計] 檢索命中率 (正確的小類名有沒有在 Top3 裡)
        gt_fine_clean = gt_label_name.strip().lower()
        top3_fine_lower = [t.strip().lower() for t in top3_fine]
        if gt_fine_clean in top3_fine_lower:
            top3_hits += 1

        # Step 3: 知識補充與 VLM 決策
        search_knowledge = ""
        if any(k in attr_desc.lower() for k in ["brand", "logo", "text", "label"]):
            search_knowledge = web_search(attr_desc)

        cand_str = "\n".join([f"- {c}" for c in top3_fine])
        fine_prompt = (
            "You are a retail product expert. Identify the product from the list.\n"
            f"Attributes: {attr_desc}\n"
            f"Search Info: {search_knowledge}\n\n"
            "Pick ONE label from below:\n"
            f"{cand_str}\n\n"
            "FINAL_LABEL: <label>"
        )

        raw_pred_fine = generate(image, fine_prompt, MAX_NEW_TOKENS_FINAL)
        pred_fine_clean = clean_prediction_text(raw_pred_fine).strip()
        
        # Step 4: 映射回粗類別
        pred_coarse = UNKNOWN_LABEL
        for fl in ALL_FINE_LABELS:
            if fl.strip().lower() == pred_fine_clean.lower():
                pred_coarse = COARSE_MAP[fl]
                break

        # [統計] 細類別正確率
        if pred_fine_clean.lower() == gt_fine_clean:
            correct_fine += 1

        # [統計] 粗類別正確率 (你原本的指標)
        if pred_coarse == gt_coarse:
            correct_coarse += 1
        else:
            # 只有粗類別錯了才存入 csv，方便分析最大的錯誤
            wrong_rows.append([
                img_path, gt_label_name, gt_coarse, pred_coarse, 
                pred_fine_clean, " | ".join(top3_fine), attr_desc.replace("\n", " ")
            ])

        # 實時進度印出
        if total % 10 == 0:
            print(f"[{total}] Coarse_Acc: {correct_coarse/total:.4f} | "
                  f"Fine_Acc: {correct_fine/total:.4f} | "
                  f"Top3_Hit: {top3_hits/total:.4f}")

        class_count += 1
        if PER_CLASS_LIMIT > 0 and class_count >= PER_CLASS_LIMIT: break
    if TOTAL_LIMIT > 0 and total > TOTAL_LIMIT: break

# ======================
# 6. 儲存結果與最終統計
# ======================
final_coarse_acc = correct_coarse / total if total > 0 else 0
final_fine_acc = correct_fine / total if total > 0 else 0
final_top3_hit = top3_hits / total if total > 0 else 0

wrong_path = os.path.join(OUT_DIR, "wrong_cases_fine_rag.csv")
with open(wrong_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["img_path", "gt_fine", "gt_coarse", "pred_coarse", "pred_fine_vlm", "top3_candidates", "attributes"])
    writer.writerows(wrong_rows)

print(f"\n===== FINAL RESULT =====")
print(f"Total Images      : {total}")
print(f"Coarse Accuracy   : {final_coarse_acc:.4f} ")
print(f"Fine Accuracy     : {final_fine_acc:.4f} ")
print(f"Top-3 Hit Rate    : {final_top3_hit:.4f} ")
print(f"Wrong cases saved to: {wrong_path}")