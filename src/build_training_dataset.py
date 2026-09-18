import pandas as pd

# Load datasets
attacks = pd.read_csv("data/noaa_matched_attacks.csv")
negatives = pd.read_csv("data/negative_samples.csv")

print("Attack samples:", len(attacks))
print("Negative samples:", len(negatives))

# Add target label
attacks["attack"] = 1

# Create day from attack date
attacks["date"] = pd.to_datetime(
    attacks["date"],
    errors="coerce"
)

attacks["day"] = attacks["date"].dt.day

# Rename attack coordinates
attacks = attacks.rename(
    columns={
        "attack_latitude": "latitude",
        "attack_longitude": "longitude"
    }
)

# Features
features = [
    "year",
    "month",
    "day",
    "hour",
    "latitude",
    "longitude",
    "sea_surface_temp",
    "wind_speed",
    "wave_height",
    "wave_period",
    "swell_height",
    "swell_period",
    "attack"
]

# Keep required columns
attacks = attacks[features]
negatives = negatives[features]

# Combine datasets
df = pd.concat(
    [attacks, negatives],
    ignore_index=True
)

print()
print("Combined samples:", len(df))

# Environmental features
environmental_features = [
    "sea_surface_temp",
    "wind_speed",
    "wave_height",
    "wave_period",
    "swell_height",
    "swell_period"
]

# Convert all model columns to numeric
for column in features:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Fill missing environmental values
# using the median of the available data
for column in environmental_features:
    median_value = df[column].median()

    df[column] = df[column].fillna(
        median_value
    )

# Fill missing time/location values
numeric_columns = [
    "year",
    "month",
    "day",
    "hour",
    "latitude",
    "longitude"
]

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
df.to_csv(
    "data/training_data.csv",
    index=False
)

print()
print("Training dataset created!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Class distribution:")
print(df["attack"].value_counts())

print()
print("Missing values:")
print(df.isnull().sum())

print()
print("Saved:")
print("data/training_data.csv")