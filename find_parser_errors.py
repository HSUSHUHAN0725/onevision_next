import pandas as pd

# 讀取你的錯誤檔
df = pd.read_csv("/home/betty/onevision_outputs/wrong_cases_retrieval.csv")

rows = []

for i, row in df.iterrows():

    gt = str(row["gt_label"]).strip()
    pred = str(row["pred_label"]).strip()

    # 如果 pred 是 "- label"
    if pred.startswith("-"):

        clean_pred = pred.lstrip("-").strip()

        # 如果去掉 "-" 後其實是正確
        if clean_pred == gt:

            rows.append({
                "excel_row": i + 2,   # Excel列號 (因為第1列是header)
                "img_path": row["img_path"],
                "gt_label": gt,
                "pred_label": pred,
                "fixed_pred": clean_pred
            })

result = pd.DataFrame(rows)

print("可修正的筆數:", len(result))

# 存檔
result.to_csv("parser_fixable_rows.csv", index=False)

print("已輸出 parser_fixable_rows.csv")