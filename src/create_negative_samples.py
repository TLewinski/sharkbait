from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "sharkbait-508602"

client = bigquery.Client(project=PROJECT_ID)

query = """
WITH attacks AS (

    SELECT
        ROW_NUMBER() OVER (
            ORDER BY date, latitude, longitude, location
        ) AS attack_id,

        date,
        year,
        month,
        EXTRACT(DAY FROM date) AS day,
        latitude,
        longitude

    FROM
        `sharkbait-508602.sharkbait.attack_locations`

),

noaa AS (

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2011`

    UNION ALL

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2012`

    UNION ALL

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2013`

    UNION ALL

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2014`

    UNION ALL

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2015`

    UNION ALL

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2016`

    UNION ALL

    SELECT
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        sea_surface_temp,
        wind_speed,
        wave_height,
        wave_period,
        swell_height,
        swell_period

    FROM
        `bigquery-public-data.noaa_icoads.icoads_core_2017`

),

candidate_negatives AS (

    SELECT

        attacks.attack_id,

        noaa.year,
        noaa.month,
        noaa.day,
        noaa.hour,

        noaa.latitude,
        noaa.longitude,

        noaa.sea_surface_temp,
        noaa.wind_speed,
        noaa.wave_height,
        noaa.wave_period,
        noaa.swell_height,
        noaa.swell_period,

        ST_DISTANCE(
            ST_GEOGPOINT(
                attacks.longitude,
                attacks.latitude
            ),
            ST_GEOGPOINT(
                noaa.longitude,
                noaa.latitude
            )
        ) / 1000 AS distance_km

    FROM attacks

    JOIN noaa

    ON
        attacks.year = noaa.year
        AND attacks.month = noaa.month
        AND attacks.day = noaa.day

    WHERE

        noaa.latitude IS NOT NULL
        AND noaa.longitude IS NOT NULL

        AND noaa.sea_surface_temp IS NOT NULL

        -- Rough geographic filter
        AND noaa.latitude BETWEEN
            attacks.latitude - 1
            AND attacks.latitude + 1

        AND noaa.longitude BETWEEN
            attacks.longitude - 1
            AND attacks.longitude + 1

        -- Negative must be close enough to be comparable
        AND ST_DWITHIN(
            ST_GEOGPOINT(
                attacks.longitude,
                attacks.latitude
            ),
            ST_GEOGPOINT(
                noaa.longitude,
                noaa.latitude
            ),
            100000
        )

        -- Keep negatives at least 25 km from the attack
        AND NOT ST_DWITHIN(
            ST_GEOGPOINT(
                attacks.longitude,
                attacks.latitude
            ),
            ST_GEOGPOINT(
                noaa.longitude,
                noaa.latitude
            ),
            25000
        )

),

selected_negatives AS (

    SELECT *

    FROM candidate_negatives

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY attack_id
        ORDER BY distance_km
    ) = 1

)

SELECT
    year,
    month,
    day,
    hour,
    latitude,
    longitude,
    sea_surface_temp,
    wind_speed,
    wave_height,
    wave_period,
    swell_height,
    swell_period

FROM selected_negatives
"""

print("Querying NOAA...")
print("Creating matched non-attack observations...")
print()

results = client.query(query).result()

rows = [dict(row) for row in results]

df = pd.DataFrame(rows)

df["attack"] = 0

df.to_csv(
    "data/negative_samples.csv",
    index=False
)

print("Negative samples created:", len(df))

print()
print("Negative samples by year:")
print(df["year"].value_counts().sort_index())

print()
print("Saved:")
print("data/negative_samples.csv")