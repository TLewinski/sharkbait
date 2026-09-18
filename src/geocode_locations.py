import pandas as pd
import time
import os
from geopy.geocoders import ArcGIS

INPUT_FILE = "data/clean_attacks.csv"
OUTPUT_FILE = "data/location_coordinates.csv"

df = pd.read_csv(INPUT_FILE)

for column in ["country", "area", "location"]:
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

unique_locations = (
    df[["country", "area", "location"]]
    .drop_duplicates()
)

print("Unique location records:", len(unique_locations))

# Create or load results file
if os.path.exists(OUTPUT_FILE):
    results_df = pd.read_csv(OUTPUT_FILE)
else:
    results_df = pd.DataFrame(columns=[
        "country",
        "area",
        "location",
        "latitude",
        "longitude",
        "match_type"
    ])

processed = set(
    zip(
        results_df["country"],
        results_df["area"],
        results_df["location"]
    )
)

successful_count = results_df["latitude"].notna().sum()

print("Already processed:", len(processed))
print("Already successful:", successful_count)

geolocator = ArcGIS(timeout=10)

new_results = []

for index, (_, row) in enumerate(unique_locations.iterrows()):

    key = (
        row["country"],
        row["area"],
        row["location"]
    )

    if key in processed:
        continue

    location = row["location"]
    area = row["area"]
    country = row["country"]

    if location == "Unknown":
        continue

    searches = [
        f"{location}, {area}, {country}",
        f"{location}, {country}"
    ]

    found = False

    print(f"\nProcessing {index + 1}/{len(unique_locations)}")
    print("Location:", location)

    for search in searches:

        try:
            print("Trying:", search)

            result = geolocator.geocode(search)

            if result:

                print(
                    "FOUND:",
                    result.latitude,
                    result.longitude
                )

                new_results.append({
                    "country": country,
                    "area": area,
                    "location": location,
                    "latitude": result.latitude,
                    "longitude": result.longitude,
                    "match_type": "location"
                })

                successful_count += 1
                found = True
                break

        except Exception as error:
            print("Error:", error)

    if not found:

        print("FAILED:", location)

        new_results.append({
            "country": country,
            "area": area,
            "location": location,
            "latitude": None,
            "longitude": None,
            "match_type": "failed"
        })

    processed.add(key)

    # Save every 25 locations
    if len(new_results) >= 25:

        new_df = pd.DataFrame(new_results)

        results_df = pd.concat(
            [results_df, new_df],
            ignore_index=True
        )

        results_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        new_results = []

        print("\nPROGRESS SAVED")
        print("Successful:", successful_count)

if new_results:

    new_df = pd.DataFrame(new_results)

    results_df = pd.concat(
        [results_df, new_df],
        ignore_index=True
    )

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================")
print("GEOCODING COMPLETE")
print("==========================")
print("Total processed:", len(results_df))
print(
    "Successful:",
    results_df["latitude"].notna().sum()
)
print(
    "Failed:",
    results_df["latitude"].isna().sum()
)
print("Saved:", OUTPUT_FILE)