from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "sharkbait-508602"
DATASET_ID = "sharkbait"
TABLE_ID = "attack_locations"

FILE = "data/attack_locations.csv"

# Connect to BigQuery
client = bigquery.Client(project=PROJECT_ID)

# Load CSV
df = pd.read_csv(FILE)

print("Loaded:", len(df), "attack records")

# Keep only the columns we need for matching
df = df[
    [
        "date",
        "year",
        "month",
        "day_of_week",
        "hour",
        "country",
        "area",
        "location",
        "latitude",
        "longitude",
        "activity",
        "fatal"
    ]
]

# Make sure date is actually a date
df["date"] = pd.to_datetime(df["date"]).dt.date

# BigQuery table
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
    autodetect=True
)

print("Uploading to BigQuery...")

job = client.load_table_from_dataframe(
    df,
    table_ref,
    job_config=job_config
)

job.result()

table = client.get_table(table_ref)

print()
print("Upload successful!")
print("Table:", table_ref)
print("Rows:", table.num_rows)
print("Columns:", len(table.schema))