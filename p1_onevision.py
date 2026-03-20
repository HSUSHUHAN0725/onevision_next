import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration

# 你本機下載好的模型資料夾（建議用這個，避免又去下載）
# 例如你現在的：./llava_onevision_qwen2_7b_ov_chat_hf
MODEL_ID = os.environ.get("MODEL_ID", "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf")


IMG_PATH = os.environ.get("IMG_PATH", "/mnt/c/Users/User/Pictures/batch1/0cb7cba6af.jpg")

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
    "Rise",
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


def build_prompt():
    labels = "\n".join([f"- {c}" for c in CATEGORIES] + [f"- {UNKNOWN_LABEL}"])
    return (
        "You are a classifier for retail product categories.\n"
        "Given the image, choose EXACTLY ONE label from the list below.\n"
        "Rules:\n"
        "1) Output MUST be exactly one label string from the list.\n"
        "2) No extra words, no punctuation, no explanation.\n"
        "3) If none match or unsure, output 'Unknown'.\n\n"
        "Label list:\n"
        f"{labels}\n"
    )


def normalize(s: str) -> str:
    s = s.strip()
    # 去掉可能的引號/句點等
    s = re.sub(r'^[\s"\'`]+|[\s"\'`\.!,;:]+$', "", s)
    return s


def pick_label(model_text: str) -> str:
    raw = model_text.strip()

    # 有些輸出會像：
    # user ... assistant ... <答案>
    # 我們抓最後一段
    if "assistant" in raw:
        raw = raw.split("assistant")[-1].strip()

    cand = normalize(raw)

    # 1) 直接完全匹配
    if cand in CATEGORIES or cand == UNKNOWN_LABEL:
        return cand

    # 2) 忽略大小寫匹配
    lower_map = {c.lower(): c for c in CATEGORIES}
    if cand.lower() in lower_map:
        return lower_map[cand.lower()]
    if cand.lower() == UNKNOWN_LABEL.lower():
        return UNKNOWN_LABEL

    # 3) 如果模型回了一句話：嘗試在句子中找出 label（含空白的 label 也能抓）
    text_lower = raw.lower()
    for c in sorted(CATEGORIES, key=len, reverse=True):  # 先找長的避免撞短字
        if c.lower() in text_lower:
            return c

    # 都找不到就 Unknown
    print("[WARN] Could not match a label. Raw output was:\n", model_text)
    return UNKNOWN_LABEL


def main():
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

    img = Image.open(IMG_PATH).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    decoded = processor.decode(out[0], skip_special_tokens=True)
    label = pick_label(decoded)
    print(label)


if __name__ == "__main__":
    main()
