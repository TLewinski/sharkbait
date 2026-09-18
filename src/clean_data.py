import pandas as pd


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/global_shark_attacks.csv")

print("Original dataset:")
print(df.shape)


# =========================
# 2. CONVERT DATE
# =========================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# =========================
# 3. CREATE TIME FEATURES
# =========================

df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek


# Convert shark attack time into an hour

df["hour"] = pd.to_numeric(
    df["time"]
    .astype(str)
    .str.extract(r"(\d{1,2})")[0],
    errors="coerce"
)


# =========================
# 4. CLEAN TEXT COLUMNS
# =========================

text_columns = [
    "type",
    "country",
    "area",
    "location",
    "activity",
    "sex",
    "species"
]

for column in text_columns:
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# =========================
# 5. CREATE FATALITY TARGET
# =========================

df["fatal"] = (
    df["fatal_y_n"]
    .str.upper()
    .map({
        "Y": 1,
        "N": 0
    })
)


# =========================
# 6. REMOVE RECORDS
#    WITHOUT A DATE
# =========================

df = df.dropna(
    subset=["date"]
)


# =========================
# 7. SAVE CLEAN DATA
# =========================

df.to_csv(
    "data/clean_attacks.csv",
    index=False
)


# =========================
# 8. DISPLAY RESULTS
# =========================

print("\nCleaned dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print("data/clean_attacks.csv")