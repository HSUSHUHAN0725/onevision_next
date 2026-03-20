import os

DATASET_ROOT = "/home/betty/datasets/locount_class_samples/locount_class_samples"

# 這裡貼你現在的 COARSE_MAP
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
    "underwear": "Wearables",

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

def normalize(x):
    return x.strip()

normalized_map = {normalize(k): v for k, v in COARSE_MAP.items()}

unmapped = []

for folder in sorted(os.listdir(DATASET_ROOT)):
    path = os.path.join(DATASET_ROOT, folder)

    if not os.path.isdir(path):
        continue

    if normalize(folder) not in normalized_map:
        unmapped.append(folder)

print("\nUNMAPPED GT folders:\n")

for x in unmapped:
    print(repr(x))

print("\nTotal unmapped:", len(unmapped))