import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Use Streamlit Cloud secret when deployed
try:
    import streamlit as st

    if "OPENWEATHER_API_KEY" in st.secrets:
        API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except Exception:
    pass


def get_pollution_data(city):
    # Step 1: Get latitude and longitude
    geo_url = (
        "https://api.openweathermap.org/geo/1.0/direct"
        f"?q={city}&limit=1&appid={API_KEY}"
    )

    geo_res = requests.get(geo_url).json()

    if not isinstance(geo_res, list):
        raise RuntimeError(f"OpenWeather error: {geo_res}")

    if not geo_res:
        raise RuntimeError(f"City not found: {city}")

    lat = geo_res[0]["lat"]
    lon = geo_res[0]["lon"]

    # Step 2: Get pollution data
    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={API_KEY}"
    )

    data = requests.get(url).json()

    if "list" not in data:
        raise RuntimeError(f"Pollution API error: {data}")

    comp = data["list"][0]["components"]

    pm25 = comp["pm2_5"]
    pm10 = comp["pm10"]
    no2 = comp["no2"]
    co = comp["co"]
    o3 = comp["o3"]
    so2 = comp.get("so2", 10)

    return [pm25, pm10, no2, so2, co, o3]