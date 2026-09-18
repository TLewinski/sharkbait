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
        hour,

        country,
        area,
        location,
        activity,
        fatal,

        latitude AS attack_latitude,
        longitude AS attack_longitude

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

matches AS (

    SELECT

        attacks.attack_id,

        attacks.date,
        attacks.year,
        attacks.month,
        attacks.hour,

        attacks.country,
        attacks.area,
        attacks.location,
        attacks.activity,
        attacks.fatal,

        attacks.attack_latitude,
        attacks.attack_longitude,

        noaa.latitude AS noaa_latitude,
        noaa.longitude AS noaa_longitude,

        noaa.sea_surface_temp,
        noaa.wind_speed,
        noaa.wave_height,
        noaa.wave_period,
        noaa.swell_height,
        noaa.swell_period,

        ST_DISTANCE(
            ST_GEOGPOINT(
                attacks.attack_longitude,
                attacks.attack_latitude
            ),
            ST_GEOGPOINT(
                noaa.longitude,
                noaa.latitude
            )
        ) / 1000 AS distance_km

    FROM
        attacks

    JOIN
        noaa

    ON
        attacks.year = noaa.year
        AND attacks.month = noaa.month
        AND attacks.day = noaa.day

    WHERE

        noaa.latitude IS NOT NULL
        AND noaa.longitude IS NOT NULL

        AND noaa.latitude BETWEEN
            attacks.attack_latitude - 1
            AND attacks.attack_latitude + 1

        AND noaa.longitude BETWEEN
            attacks.attack_longitude - 1
            AND attacks.attack_longitude + 1

        AND ST_DWITHIN(
            ST_GEOGPOINT(
                attacks.attack_longitude,
                attacks.attack_latitude
            ),
            ST_GEOGPOINT(
                noaa.longitude,
                noaa.latitude
            ),
            100000
        )

)

SELECT *

FROM matches

QUALIFY ROW_NUMBER() OVER (

    PARTITION BY attack_id

    ORDER BY distance_km

) = 1
"""

print("Running NOAA matching query...")
print("Matching 858 attacks against NOAA observations...")
print()

results = client.query(query).result()

rows = [dict(row) for row in results]

print("Matching complete!")
print("Matched attacks:", len(rows))

if rows:

    print()
    print("First 10 matches:")

    for row in rows[:10]:

        print(
            row["date"],
            "|",
            row["location"],
            "|",
            round(row["distance_km"], 2),
            "km"
        )

df = pd.DataFrame(rows)

df.to_csv(
    "data/noaa_matched_attacks.csv",
    index=False
)

print()
print("Saved:")
print("data/noaa_matched_attacks.csv")