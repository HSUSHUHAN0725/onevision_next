# p2_onevision.py
import os
import re
import csv
import time
from collections import Counter, defaultdict

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)

# ======================
# Config
# ======================
MODEL_ID = os.environ.get("MODEL_ID", "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf")
DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    "/home/betty/datasets/locount_class_samples/locount_class_samples"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/betty/onevision_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 投票次數（建議 3 或 5）
K_VOTES = int(os.environ.get("K_VOTES", "3"))

# generation sampling 參數（可微調）
TEMP = float(os.environ.get("TEMP", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))

# ======================
# Your labels
# ======================
CATEGORIES = [
    "Adult Diapers",
    "Adult hat",
    "Adult milk powder",
    "Adult shoes",
    "Adult socks",
    "Air conditioner",
    "Air conditioning fan",
    "Baby carriage",
    "Baby crib",
    "Baby diapers",
    "Baby Furniture",
    "Baby handkerchiefs",
    "Baby milk powder",
    "Baby slippers",
    "Baby tableware",
    "Baby Toys",
    "Baby washing and nursing supplie",

    "Badminton",
    "Band aid",
    "Basin",
    "Basketball",
    "Bath lotion",
    "Bedding set",
    "Biscuits",
    "Bowl",
    "Cake",
    "Can",
    "Carbonated drinks",
    "Care Kit",
    "Chewing gum",
    "Children hats",
    "Children shoes",
    "Children Socks",
    "Children Toys",
    "Children underwear",
    "Chocolates",
    "Chopping block",
    "Chopsticks",
    "Coat hanger",
    "Cocktail",
    "Coffee",
    "Comb",
    "Cooking wine",
    "Cotton swab",
    "Cutter",
    "Dairy",
    "Desk lamp",
    "Dinner plate",
    "Disposable bag",
    "Disposable cups",
    "Draw bar box",
    "Dried beans",
    "Dried fish",
    "Dried meat",
    "Electric fan",
    "Electric frying pan",
    "Electric Hot pot",
    "Electric iron",
    "Electric kettle",
    "Electric steaming pan",
    "Electromagnetic furnace",
    "Emulsion",
    "Facial Cleanser",
    "Facial mask",
    "Fish tofu",
    "Flour",
    "Food box",
    "Football",
    "Forks",
    "Fresh-keeping film",
    "Ginger Tea",
    "Guozhen",
    "Hair conditioner",
    "Hair drier",
    "Hair dye",
    "Hair gel",
    "Herbal tea",
    "Hot strips",
    "Ice cream",
    "Instant noodles",
    "Jacket",
    "Juicer",
    "Knapsack",
    "Knives",
    "Lingerie",
    "Liquor and Spirits",
    "Lotus root flour",
    "Makeup tools",
    "Men underwear",
    "Microwave Oven",
    "Mixed congee",
    "Mouth wash",
    "Mug",
    "Noodle",
    "Notebook",
    "Oats",
    "Pasta",
    "Pen",
    "Pencil case",
    "Pie",
    "Pot shovel",
    "Potato chips",

    "Quick-frozen dumplings",
    "Quick-frozen Tangyuan",
    "Quick-frozen Wonton",

    "Razor",
    "Red wine",
    "Refrigerator",
    "Rice cooker",
    "Rice",
    "Rubber ball",
    "Sauce",
    "Sesame paste",
    "Shampoo",
    "Skate",
    "Skin care set",
    "Soap",
    "Socket",
    "Soup ladle",
    "Sour Plum Soup",
    "Soy sauce",
    "Soybean Milk machine",
    "Soymilk",
    "Spoon",
    "Sports cup",
    "Stool",
    "Storage bottle",
    "Storage box",
    "Tampon",
    "Tea",
    "Tea beverage",
    "Television",
    "Thermos bottle",
    "Toothbrush",
    "Toothpaste",
    "Trash",
    "Trousers",
    "Vinegar",
    "Walnut powder",
    "Washing machine",
]

UNKNOWN_LABEL = "Unknown"

# ======================
# Merge mapping (你要合併的)
# key = 新標籤名, value = 原本標籤列表
# ======================
MERGE_GROUPS = {
    # 你剛剛貼的合併例子：
    "Diapers": ["Adult Diapers", "Baby diapers"],
    "Milk powder": ["Adult milk powder", "Baby milk powder"],
    "Shoes": ["Adult shoes", "Baby slippers", "Children shoes"],
    "Socks": ["Adult socks", "Children Socks"],
    "Hats": ["Adult hat", "Children hats"],
    "Toys": ["Baby Toys", "Children Toys"],
    "Underwear": ["Children underwear", "Men underwear"],
    "Juicer": ["Soybean Milk machine", "Juicer"],
}

# 由 MERGE_GROUPS 建立「原標籤 -> 新標籤」的映射（包含沒合併的維持原樣）
ORIG2MERGED = {}
for new_lbl, orig_list in MERGE_GROUPS.items():
    for o in orig_list:
        ORIG2MERGED[o] = new_lbl


def to_merged(label: str) -> str:
    """把 label 轉成合併後的 label（若不在合併表就原樣）"""
    return ORIG2MERGED.get(label, label)


# ======================
# Prompt / parsing
# ======================
def build_prompt():
    labels = "\n".join([f"- {c}" for c in CATEGORIES] + [f"- {UNKNOWN_LABEL}"])
    return (
        "You are a classifier for retail product categories.\n"
        "Given the image, choose EXACTLY ONE label from the list below.\n"
        "Rules:\n"
        "1) Output MUST be exactly one label string from the list.\n"
        "2) No extra words, no punctuation, no explanation.\n"
        f"3) If none match or unsure, output '{UNKNOWN_LABEL}'.\n\n"
        "Label list:\n"
        f"{labels}\n"
    )


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[\s"\'`]+|[\s"\'`\.!,;:]+$', "", s)
    return s


def pick_label(model_text: str) -> str:
    raw = model_text.strip()

    # 有些輸出會包含 user/assistant 字樣，抓最後段
    if "assistant" in raw:
        raw = raw.split("assistant")[-1].strip()

    cand = normalize(raw)

    # 完全匹配
    if cand in CATEGORIES or cand == UNKNOWN_LABEL:
        return cand

    # 忽略大小寫匹配
    lower_map = {c.lower(): c for c in CATEGORIES}
    if cand.lower() in lower_map:
        return lower_map[cand.lower()]
    if cand.lower() == UNKNOWN_LABEL.lower():
        return UNKNOWN_LABEL

    # 句子中包含某 label（先找長的）
    text_lower = raw.lower()
    for c in sorted(CATEGORIES, key=len, reverse=True):
        if c.lower() in text_lower:
            return c

    return UNKNOWN_LABEL


def majority_vote(labels):
    """
    多次輸出後投票：
    - 多數決
    - 同票：選「第一個出現」的（穩定可重現）
    """
    cnt = Counter(labels)
    top = cnt.most_common()
    if not top:
        return UNKNOWN_LABEL

    best_n = top[0][1]
    candidates = [k for k, v in top if v == best_n]

    if len(candidates) == 1:
        return candidates[0]

    # tie-break：第一個出現者
    for x in labels:
        if x in candidates:
            return x
    return candidates[0]


# ======================
# Dataset scanning
# ======================
def iter_images(dataset_root):
    # 假設結構：root/GT_LABEL/*.jpg
    for gt in sorted(os.listdir(dataset_root)):
        gt_dir = os.path.join(dataset_root, gt)
        if not os.path.isdir(gt_dir):
            continue
        for fn in sorted(os.listdir(gt_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                yield os.path.join(gt_dir, fn), gt


# ======================
# Main
# ======================
def main():
    print("MODEL_ID:", MODEL_ID)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("K_VOTES:", K_VOTES)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    prompt_text = build_prompt()

    total = 0
    correct = 0
    parse_fail = 0

    # 統計：GT(merged) -> Pred(merged) -> count
    confusion = defaultdict(int)

    wrong_rows = []
    t0 = time.time()

    for img_path, gt_label in iter_images(DATASET_ROOT):
        total += 1

        gt_m = to_merged(gt_label)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 讀不到圖：當 parse fail
            parse_fail += 1
            wrong_rows.append([img_path, gt_label, "READ_FAIL", str(e)[:200]])
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]

        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=img, text=chat_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        preds = []
        raw_heads = []

        with torch.inference_mode():
            for _ in range(K_VOTES):
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,              # ✅ 讓每次投票有差異
                    temperature=TEMP,
                    top_p=TOP_P,
                )
                txt = processor.decode(out[0], skip_special_tokens=True)
                raw_heads.append(txt[:200].replace("\n", "\\n"))
                pred = pick_label(txt)
                preds.append(pred)

        # 如果 K 次全都 Unknown，視為 parse_fail（你也可改成不算）
        if all(p == UNKNOWN_LABEL for p in preds):
            parse_fail += 1

        final_pred = majority_vote(preds)

        pred_m = to_merged(final_pred)

        confusion[(gt_m, pred_m)] += 1

        if pred_m == gt_m:
            correct += 1
        else:
            wrong_rows.append([img_path, gt_label, final_pred, raw_heads[0]])

        if total % 20 == 0:
            acc = correct / total
            print(f"[{total}] acc={acc:.4f} (parse_fail={parse_fail})")

    acc = correct / total if total else 0.0

    # ===== save outputs =====
    wrong_path = os.path.join(OUTPUT_DIR, "wrong_cases.csv")
    confusion_path = os.path.join(OUTPUT_DIR, "confusion_merged.csv")
    stats_path = os.path.join(OUTPUT_DIR, "result_summary.txt")

    with open(wrong_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "gt_label", "pred_label", "raw_output_head"])
        w.writerows(wrong_rows)

    # confusion csv: GT, Pred, Count
    with open(confusion_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["GT", "Pred", "Count"])
        for (gtm, pm), c in sorted(confusion.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            w.writerow([gtm, pm, c])

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("===== RESULT (MERGED, VOTING) =====\n")
        f.write(f"Total      : {total}\n")
        f.write(f"Correct    : {correct}\n")
        f.write(f"Parse fail : {parse_fail}\n")
        f.write(f"Accuracy   : {acc:.4f}\n")
        f.write(f"wrong_cases.csv -> {wrong_path}\n")
        f.write(f"confusion_merged.csv -> {confusion_path}\n")
        f.write(f"Time(s)    : {time.time()-t0:.1f}\n")

    print("===== RESULT (MERGED, VOTING) =====")
    print("Total      :", total)
    print("Correct    :", correct)
    print("Parse fail :", parse_fail)
    print("Accuracy   :", f"{acc:.4f}")
    print("Saved wrong cases ->", wrong_path)
    print("Saved confusion ->", confusion_path)
    print("Saved summary ->", stats_path)


if __name__ == "__main__":
    main()