import pandas as pd
import joblib
import requests
import math
import xml.etree.ElementTree as ET
from datetime import datetime


# =========================
# SETTINGS
# =========================

MODEL_PATH = "models/sharkbait_time_model.pkl"

ACTIVE_STATIONS_URL = (
    "https://www.ndbc.noaa.gov/activestations.xml"
)

REALTIME_URL = (
    "https://www.ndbc.noaa.gov/data/realtime2/"
)

MAX_STATIONS_TO_CHECK = 10


# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# DISTANCE
# =========================

def distance_km(lat1, lon1, lat2, lon2):

    radius = 6371

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    difference_lat = math.radians(lat2 - lat1)
    difference_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(difference_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(difference_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius * c


# =========================
# GET NOAA STATIONS
# =========================

def get_stations():

    response = requests.get(
        ACTIVE_STATIONS_URL,
        timeout=20
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    stations = []

    for station in root.findall("station"):

        station_id = station.get("id")
        latitude = station.get("lat")
        longitude = station.get("lon")

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError):
            continue

        stations.append({
            "station": station_id,
            "latitude": latitude,
            "longitude": longitude
        })

    return pd.DataFrame(stations)


# =========================
# GET STATION DATA
# =========================

def get_station_data(station_id):

    url = (
        f"{REALTIME_URL}"
        f"{station_id}.txt"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

    except requests.RequestException:

        return None

    if response.status_code != 200:

        return None

    lines = response.text.splitlines()

    header = None

    for line in lines:

        if line.startswith("#YY"):

            header = line[1:].split()
            break

    if header is None:

        return None

    data = []

    for line in lines:

        if line.startswith("#"):
            continue

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != len(header):
            continue

        data.append(parts)

    if not data:

        return None

    df = pd.DataFrame(
        data,
        columns=header
    )

    numeric_columns = [
        "WSPD",
        "WVHT",
        "DPD",
        "WTMP"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df.iloc[0]


# =========================
# FIND BEST NOAA STATION
# =========================

def find_best_station(
    latitude,
    longitude,
    stations
):

    stations = stations.copy()

    stations["distance_km"] = stations.apply(
        lambda row: distance_km(
            latitude,
            longitude,
            row["latitude"],
            row["longitude"]
        ),
        axis=1
    )

    stations = stations.sort_values(
        "distance_km"
    )

    print()
    print(
        f"Checking nearest {MAX_STATIONS_TO_CHECK} "
        f"NOAA stations..."
    )

    best_station = None
    best_observation = None

    for _, station in stations.head(
        MAX_STATIONS_TO_CHECK
    ).iterrows():

        station_id = station["station"]

        print(
            f"Checking {station_id} "
            f"({station['distance_km']:.1f} km)..."
        )

        observation = get_station_data(
            station_id
        )

        if observation is None:
            continue

        # Count useful marine measurements
        useful_values = 0

        for column in [
            "WTMP",
            "WSPD",
            "WVHT",
            "DPD"
        ]:

            if (
                column in observation
                and pd.notna(observation[column])
            ):

                useful_values += 1

        # Prefer stations with more usable data.
        # Distance breaks ties.
        if best_station is None:

            best_station = station
            best_observation = observation
            best_useful_values = useful_values

        elif useful_values > best_useful_values:

            best_station = station
            best_observation = observation
            best_useful_values = useful_values

        elif (
            useful_values == best_useful_values
            and station["distance_km"]
            < best_station["distance_km"]
        ):

            best_station = station
            best_observation = observation
            best_useful_values = useful_values

    return best_station, best_observation


# =========================
# SAFE VALUE
# =========================

def get_value(
    observation,
    column
):

    if column not in observation:

        return None

    value = observation[column]

    if pd.isna(value):

        return None

    return float(value)


# =========================
# MAIN
# =========================

