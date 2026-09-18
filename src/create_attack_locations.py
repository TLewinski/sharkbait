import pandas as pd

# Load files
attacks = pd.read_csv("data/clean_attacks.csv")
coordinates = pd.read_csv("data/location_coordinates.csv")

# Make sure the columns match
for df in [attacks, coordinates]:
    for column in ["country", "area", "location"]:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

# Merge attacks with coordinates
merged = attacks.merge(
    coordinates[
        [
            "country",
            "area",
            "location",
            "latitude",
            "longitude"
        ]
    ],
    on=["country", "area", "location"],
    how="left"
)

# Convert year
merged["year"] = pd.to_numeric(
    merged["year"],
    errors="coerce"
)

# Keep attacks from years NOAA currently has
merged = merged[
    (merged["year"] >= 2011) &
    (merged["year"] <= 2017)
]

# Remove attacks without coordinates
merged = merged.dropna(
    subset=["latitude", "longitude"]
)

# Save
merged.to_csv(
    "data/attack_locations.csv",
    index=False
)

print("Created attack_locations.csv")
print("Rows:", len(merged))
print("Columns:", merged.columns.tolist())
print()
print(merged[
    [
        "date",
        "country",
        "area",
        "location",
        "latitude",
        "longitude"
    ]
].head(10))