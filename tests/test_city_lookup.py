import pandas as pd
import pytest

from src.model import train_model
from src.preprocess import build_master_dataset, get_available_cities, get_city_latest_valid_record, load_processed_dataset


def test_build_master_dataset_uses_only_valid_aqi_source_records():
    df = build_master_dataset(force_rebuild=True)

    assert not df.empty
    assert {"city", "state", "station", "last_update", "latitude", "longitude", "pm25", "pm10", "no2", "so2", "co", "o3", "aqi", "aqi_category"}.issubset(df.columns)
    assert df["aqi"].notna().any()
    assert df["aqi_category"].notna().any()

    cutoff = pd.Timestamp.now()
    assert (df["last_update"] <= cutoff).all()


def test_city_lookup_returns_latest_valid_record_for_supported_city():
    cities = get_available_cities()
    assert "Delhi" in cities

    record = get_city_latest_valid_record("Delhi")
    assert record is not None
    assert record["city"] == "Delhi"
    assert record["last_update"] is not None
    assert record["last_update"] <= pd.Timestamp.now()
    assert pd.notna(record["aqi"])
    assert record["aqi"] >= 0


def test_model_training_returns_best_model_and_metrics():
    model, metrics = train_model(load_processed_dataset())

    assert model is not None
    assert isinstance(metrics, tuple)
    assert len(metrics) == 4
    assert metrics[0] in {"RandomForest", "GradientBoosting"}
    assert metrics[1] >= 0.0
    assert metrics[2] >= 0.0
