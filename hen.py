import pandas as pd
from rec import recommend   # ✅ your function

# ✅ Load Test-Set sheet
df_test = pd.read_excel("Gen_AI Dataset.xlsx", sheet_name="Test-Set")

output_rows = []

# ✅ Loop over all queries
for q in df_test["Query"]:
    preds = recommend(q, top_k=7)   # max 10 allowed
    for p in preds:
        output_rows.append({
            "Query": q,
            "Assessment_url": p["url"]
        })

# ✅ Convert → DataFrame
df_out = pd.DataFrame(output_rows)

# ✅ Save CSV
df_out.to_csv("predictions.csv", index=False)

print("✅ predictions.csv successfully created!")
