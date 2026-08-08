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
    if not city or not str(city).strip():
        raise ValueError("City name is required")

    city_name = str(city).strip()

    if not API_KEY:
        raise RuntimeError("OpenWeather API key is not configured")

    try:
        geo_url = (
            "https://api.openweathermap.org/geo/1.0/direct"
            f"?q={city_name}&limit=1&appid={API_KEY}"
        )
        geo_res = requests.get(geo_url, timeout=10).json()

        if not isinstance(geo_res, list):
            raise RuntimeError(f"OpenWeather error: {geo_res}")

        if not geo_res:
            raise RuntimeError(f"City not found: {city_name}")

        lat = geo_res[0]["lat"]
        lon = geo_res[0]["lon"]

        url = (
            "https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}&lon={lon}&appid={API_KEY}"
        )
        data = requests.get(url, timeout=10).json()

        if "list" not in data:
            raise RuntimeError(f"Pollution API error: {data}")

        comp = data["list"][0]["components"]

        pm25 = comp.get("pm2_5")
        pm10 = comp.get("pm10")
        no2 = comp.get("no2")
        co = comp.get("co")
        o3 = comp.get("o3")
        so2 = comp.get("so2")

        return {"pm25": pm25, "pm10": pm10, "no2": no2, "so2": so2, "co": co, "o3": o3}
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to reach the pollution API: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to read pollution data: {exc}") from exc