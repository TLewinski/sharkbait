import pandas as pd
import requests
import math
import xml.etree.ElementTree as ET


# =========================
# NOAA SETTINGS
# =========================

ACTIVE_STATIONS_URL = (
    "https://www.ndbc.noaa.gov/activestations.xml"
)

REALTIME_URL = (
    "https://www.ndbc.noaa.gov/data/realtime2/"
)


# =========================
# CALCULATE DISTANCE
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
# LOAD ACTIVE NOAA STATIONS
# =========================

def get_stations():

    print("Loading NOAA stations...")

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

        if not station_id:
            continue

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
# FIND NEAREST STATION
# =========================

def find_nearest_station(
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

    return stations.sort_values(
        "distance_km"
    ).iloc[0]


# =========================
# GET REAL-TIME DATA
# =========================

def get_station_data(station_id):

    url = (
        f"{REALTIME_URL}"
        f"{station_id}.txt"
    )

    print(
        f"Downloading NOAA data for "
        f"station {station_id}..."
    )

    response = requests.get(
        url,
        timeout=20
    )

    if response.status_code != 200:
        return None

    lines = response.text.splitlines()

    # Find the actual column header
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

    # Convert measurements to numbers
    numeric_columns = [
        "WSPD",
        "WVHT",
        "DPD",
        "APD",
        "WTMP"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Most recent observation
    latest = df.iloc[0]

    return latest


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("==============================")
    print("   SHARKBAIT REAL-TIME DATA")
    print("==============================")
    print()

    latitude = float(
        input("Latitude: ")
    )

    longitude = float(
        input("Longitude: ")
    )

    print()

    stations = get_stations()

    print(
        f"Found {len(stations)} active NOAA stations."
    )

    if stations.empty:

        print(
            "ERROR: No NOAA stations were found."
        )

        raise SystemExit

    nearest = find_nearest_station(
        latitude,
        longitude,
        stations
    )

    station_id = nearest["station"]

    print()
    print("==============================")
    print("NEAREST NOAA STATION")
    print("==============================")
    print()

    print(
        f"Station: {station_id}"
    )

    print(
        f"Station latitude: "
        f"{nearest['latitude']:.4f}"
    )

    print(
        f"Station longitude: "
        f"{nearest['longitude']:.4f}"
    )

    print(
        f"Distance: "
        f"{nearest['distance_km']:.1f} km"
    )

    print()

    observation = get_station_data(
        station_id
    )

    if observation is None:

        print(
            "No real-time observation "
            "was available for this station."
        )

        raise SystemExit

    print("==============================")
    print("CURRENT MARINE CONDITIONS")
    print("==============================")
    print()

    print(
        f"Water Temperature: "
        f"{observation.get('WTMP', 'N/A')} °C"
    )

    print(
        f"Wind Speed: "
        f"{observation.get('WSPD', 'N/A')} m/s"
    )

    print(
        f"Wave Height: "
        f"{observation.get('WVHT', 'N/A')} m"
    )

    print(
        f"Wave Period: "
        f"{observation.get('DPD', 'N/A')} s"
    )

    print()

    print("==============================")
    print("REAL-TIME NOAA DATA RECEIVED")
    print("==============================")