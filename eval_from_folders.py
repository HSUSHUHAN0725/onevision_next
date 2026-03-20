import os
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration

# ========= 1) 路徑 & 模型 =========
ROOT_DIR = "/home/betty/datasets/locount_class_samples/locount_class_samples"
MODEL_DIR = "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf"   # 你 snapshot_download 存的本機資料夾
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 2) 你的 140 類 =========
CATEGORIES = [
    "Adult Diapers","Adult hat","Adult milk powder","Adult shoes","Adult socks",
    "Air conditioner","Air conditioning fan","Baby carriage","Baby crib","Baby diapers",
    "Baby Furniture","Baby handkerchiefs","Baby milk powder","Baby slippers","Baby tableware",
    "Baby Toys","Baby washing and nursing supplie","Badminton","Band aid","Basin",
    "Basketball","Bath lotion","Bedding set","Biscuits","Bowl","Cake","Can",
    "Carbonated drinks","Care Kit","Chewing gum","Children hats","Children shoes",
    "Children Socks","Children Toys","Children underwear","Chocolates","Chopping block",
    "Chopsticks","Coat hanger","Cocktail","Coffee","Comb","Cooking wine","Cotton swab",
    "Cutter","Dairy","Desk lamp","Dinner plate","Disposable bag","Disposable cups",
    "Draw bar box","Dried beans","Dried fish","Dried meat","Electric fan",
    "Electric frying pan","Electric Hot pot","Electric iron","Electric kettle",
    "Electric steaming pan","Electromagnetic furnace","Emulsion","Facial Cleanser",
    "Facial mask","Fish tofu","Flour","Food box","Football","Forks","Fresh-keeping film",
    "Ginger Tea","Guozhen","Hair conditioner","Hair drier","Hair dye","Hair gel",
    "Herbal tea","Hot strips","Ice cream","Instant noodles","Jacket","Juicer","Knapsack",
    "Knives","Lingerie","Liquor and Spirits","Lotus root flour","Makeup tools",
    "Men underwear","Microwave Oven","Mixed congee","Mouth wash","Mug","Noodle",
    "Notebook","Oats","Pasta","Pen","Pencil case","Pie","Pot shovel","Potato chips",
    "Quick-frozen dumplings","Quick-frozen Tangyuan","Quick-frozen Wonton","Razor",
    "Red wine","Refrigerator","Rice cooker","Rice","Rubber ball","Sauce","Sesame paste",
    "Shampoo","Skate","Skin care set","Soap","Socket","Soup ladle","Sour Plum Soup",
    "Soy sauce","Soybean Milk machine","Soymilk","Spoon","Sports cup","Stool",
    "Storage bottle","Storage box","Tampon","Tea","Tea beverage","Television",
    "Thermos bottle","Toothbrush","Toothpaste","Trash","Trousers","Vinegar",
    "Walnut powder","Washing machine"
]
CAT_SET = set(CATEGORIES)

# ========= 3) 讓模型「一定只回 140 類其中一個」的 prompt =========
def build_prompt():
    # 你也可以把規則寫更嚴格，例如 "回答必須完全等於某個類別字串"
    return (
        "You are a classifier. Choose exactly ONE label from the list below.\n"
        "Return ONLY one label from the list.\n\n"
        "Put the final answer on the FIRST LINE only."
        "No extra words, no punctuation.
        "Labels:\n- " + "\n- ".join(CATEGORIES)
    )

# ========= 4) 解析模型輸出，盡量映射成 140 類 =========
def normalize_pred(text: str):
    t = text.strip()

    # 常見：模型會把 prompt/role 也吐出來，先粗暴去掉
    t = re.sub(r"(?is).*assistant\s*", "", t).strip()
    t = re.sub(r"(?is).*user\s*", "", t).strip()

    # 如果它回一整句，我們就找「哪個類別字串出現在輸出裡」
    # (最穩的方法：從長到短比對，避免 "Tea" 被 "Tea beverage" 誤撞)
    for label in sorted(CATEGORIES, key=len, reverse=True):
        if label.lower() in t.lower():
            return label

    # 有些會加引號
    t2 = t.strip("\"'` ").strip()
    if t2 in CAT_SET:
        return t2

    return None  # 找不到就算 unknown/parse_fail

# ========= 5) 載入模型（4bit NF4） =========
def load_model():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

    if getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print("Loading model (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        device_map="auto",  # 讓 accelerate 自己放到 GPU
        quantization_config=bnb,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    return processor, model

# ========= 6) 對單張圖片推論 =========
@torch.inference_mode()
def predict_one(processor, model, img_path: str):
    img = Image.open(img_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": build_prompt()},
        ],
    }]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def iter_images(root_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    root = Path(root_dir)

    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        gt_label = cls_dir.name
        for img in cls_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in exts:
                yield gt_label, str(img)

def main():
    processor, model = load_model()

    total = 0
    correct = 0
    parse_fail = 0

    wrong_rows = []

    for gt_label, img_path in iter_images(ROOT_DIR):
        if gt_label not in CAT_SET:
            # 如果資料夾名字不是 140 類其中之一，先跳過避免污染結果
            continue

        total += 1
        raw = predict_one(processor, model, img_path)
        pred = text.splitlines()[0].strip()

        if pred is None:
            parse_fail += 1
            wrong_rows.append([img_path, gt_label, "", raw.replace("\n", "\\n")[:300]])
            continue

        if pred == gt_label:
            correct += 1
        else:
            wrong_rows.append([img_path, gt_label, pred, raw.replace("\n", "\\n")[:300]])

        if total % 20 == 0:
            acc = correct / total
            print(f"[{total}] acc={acc:.4f} (parse_fail={parse_fail})")

    acc = correct / total if total else 0.0
    print("\n===== RESULT =====")
    print(f"Total      : {total}")
    print(f"Correct    : {correct}")
    print(f"Parse fail : {parse_fail}")
    print(f"Accuracy   : {acc:.4f}")

    # 輸出錯誤案例
    out_csv = "wrong_cases.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "gt_label", "pred_label", "raw_output_head"])
        w.writerows(wrong_rows)
    print(f"Saved wrong cases -> {out_csv}")

if __name__ == "__main__":
    main()
