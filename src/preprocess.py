from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.predict import calculate_current_aqi

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_AQI_PATH = ROOT_DIR / "data" / "AQI.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "master_aqi_dataset.csv"
FEATURE_COLUMNS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

POLLUTANT_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "OZONE": "o3",
}


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _normalize_pollutant_name(value: object) -> str:
    if pd.isna(value):
        return ""
    name = str(value).strip().upper()
    return POLLUTANT_MAP.get(name, "")


def _pick_measurement(row: pd.Series) -> float | None:
    for column in ["pollutant_avg", "pollutant_min", "pollutant_max"]:
        value = row.get(column)
        if pd.notna(value):
            return float(value)
    return None


def build_master_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    raw_aqi = pd.read_csv(RAW_AQI_PATH)
    raw_aqi = raw_aqi.copy()
    raw_aqi["last_update"] = _parse_datetime(raw_aqi["last_update"])
    raw_aqi["city"] = raw_aqi["city"].astype(str).str.strip()
    raw_aqi["state"] = raw_aqi["state"].astype(str).str.strip()
    raw_aqi["station"] = raw_aqi["station"].astype(str).str.strip()

    cutoff = pd.Timestamp.now()
    valid_rows = raw_aqi.loc[raw_aqi["last_update"].notna() & (raw_aqi["last_update"] <= cutoff)].copy()

    if valid_rows.empty:
        raise ValueError("No valid AQI.csv records remain after applying the current-date filter")

    records = []
    for _, row in valid_rows.iterrows():
        pollutant_name = _normalize_pollutant_name(row.get("pollutant_id"))
        if not pollutant_name:
            continue
        measurement = _pick_measurement(row)
        if measurement is None:
            continue

        records.append(
            {
                "city": row.get("city"),
                "state": row.get("state"),
                "station": row.get("station"),
                "last_update": row.get("last_update"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "pollutant_key": pollutant_name,
                "measurement_value": measurement,
            }
        )

    if not records:
        raise ValueError("No valid pollutant rows were available in AQI.csv")

    long_df = pd.DataFrame(records)
    long_df = long_df.drop_duplicates(
        subset=["city", "state", "station", "last_update", "pollutant_key"],
        keep="first",
    )

    wide_df = (
        long_df.pivot_table(
            index=["city", "state", "station", "last_update", "latitude", "longitude"],
            columns="pollutant_key",
            values="measurement_value",
            aggfunc="first",
        )
        .reset_index()
    )

    wide_df["last_update"] = pd.to_datetime(wide_df["last_update"], errors="coerce")
    wide_df = wide_df.sort_values("last_update", kind="mergesort").reset_index(drop=True)

    for column in FEATURE_COLUMNS:
        if column not in wide_df.columns:
            wide_df[column] = pd.NA

    wide_df = wide_df.drop_duplicates(subset=["city", "state", "station", "last_update"], keep="first")
    wide_df = wide_df.reset_index(drop=True)
    wide_df = wide_df[["city", "state", "station", "last_update", "latitude", "longitude", *FEATURE_COLUMNS]]

    aqi_rows = []
    for _, row in wide_df.iterrows():
        pollutant_values = {col: row[col] for col in FEATURE_COLUMNS}
        aqi, category, _ = calculate_current_aqi(pollutant_values)
        if aqi is None:
            continue

        processed_row = row.to_dict()
        processed_row["aqi"] = aqi
        processed_row["aqi_category"] = category
        aqi_rows.append(processed_row)

    if not aqi_rows:
        raise ValueError("No valid AQI targets could be derived from AQI.csv using CPCB methodology")

    processed_df = pd.DataFrame(aqi_rows)
    processed_df = processed_df.sort_values("last_update", kind="mergesort").reset_index(drop=True)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    return processed_df


def load_processed_dataset() -> pd.DataFrame:
    df = build_master_dataset()
    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
    return df.sort_values("last_update", kind="mergesort").reset_index(drop=True)


def normalize_city_name(city: str) -> str:
    if city is None:
        return ""
    normalized = str(city).strip()
    normalized = " ".join(normalized.split())
    return normalized


def _normalize_city_for_search(city: str) -> str:
    return normalize_city_name(city).lower()


def get_available_cities() -> list[str]:
    df = load_processed_dataset()
    cities = df["city"].dropna().astype(str).str.strip()
    return sorted(cities.unique().tolist())


def get_city_matches(city: str) -> pd.DataFrame:
    city_name = _normalize_city_for_search(city)
    if not city_name:
        return pd.DataFrame(columns=["city"] + FEATURE_COLUMNS)

    df = load_processed_dataset()
    matches = df[df["city"].astype(str).str.strip().str.lower() == city_name].copy()
    return matches


def get_city_latest_record(city: str) -> Optional[dict]:
    matches = get_city_matches(city)
    if matches.empty:
        return None

    latest = matches.sort_values("last_update", ascending=False).iloc[0]
    return latest.to_dict()


def get_city_latest_valid_record(city: str) -> Optional[dict]:
    matches = get_city_matches(city)
    if matches.empty:
        return None

    ordered = matches.sort_values("last_update", ascending=False)
    for _, row in ordered.iterrows():
        pollutant_values = {col: row[col] for col in FEATURE_COLUMNS}
        aqi, _, category = calculate_current_aqi(pollutant_values)
        if aqi is not None:
            processed_row = row.to_dict()
            processed_row["aqi"] = aqi
            processed_row["aqi_category"] = category
            return processed_row

    return None


def get_city_latest_complete_record(city: str) -> Optional[dict]:
    matches = get_city_matches(city)
    if matches.empty:
        return None

    ordered = matches.sort_values("last_update", ascending=False)
    for _, row in ordered.iterrows():
        if all(pd.notna(row[column]) for column in FEATURE_COLUMNS):
            return row.to_dict()

    return ordered.iloc[0].to_dict()


def build_feature_row_from_record(record: dict) -> list[float]:
    return [record.get("pm25"), record.get("pm10"), record.get("no2"), record.get("so2"), record.get("co"), record.get("o3")]
