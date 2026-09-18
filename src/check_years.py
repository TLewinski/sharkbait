import pandas as pd

df = pd.read_csv("data/clean_attacks.csv")

df["year"] = pd.to_numeric(df["year"], errors="coerce")

print("Earliest attack year:", int(df["year"].min()))
print("Latest attack year:", int(df["year"].max()))

print("\nAttacks by year:")
print(df["year"].value_counts().sort_index())