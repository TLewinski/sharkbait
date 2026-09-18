from google.cloud import bigquery

PROJECT_ID = "sharkbait-508602"
DATASET_ID = "sharkbait"

client = bigquery.Client(project=PROJECT_ID)

dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

dataset = bigquery.Dataset(dataset_ref)

# Use the US multi-region
dataset.location = "US"

try:
    client.create_dataset(dataset)
    print("Dataset created successfully!")
    print("Dataset:", dataset_ref)

except Exception as error:
    if "Already Exists" in str(error):
        print("Dataset already exists.")
    else:
        print("Error creating dataset:")
        print(error)