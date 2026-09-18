from google.cloud import bigquery

client = bigquery.Client(project="sharkbait-508602")

query = """
SELECT
    year,
    month,
    day,
    latitude,
    longitude
FROM
    `bigquery-public-data.noaa_icoads.icoads_core_2011`
WHERE
    latitude IS NOT NULL
    AND longitude IS NOT NULL
LIMIT 5
"""

results = client.query(query).result()

print("BigQuery connection successful!")
print()

for row in results:
    print(row)