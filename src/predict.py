import pandas as pd

POLLUTANT_BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 350.0, 401, 450),
        (350.1, 500.0, 451, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 520.0, 401, 450),
        (520.1, 600.0, 451, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 524.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2100.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 54.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 500),
    ],
}

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderately Polluted"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

FEATURE_COLUMNS = ["pm25", "pm10", "no2", "so2", "co", "o3"]


def _find_breakpoint(pollutant: str, concentration: float):
    ranges = POLLUTANT_BREAKPOINTS.get(pollutant)
    if ranges is None or concentration is None or pd.isna(concentration):
        return None

    for c_lo, c_hi, i_lo, i_hi in ranges:
        if c_lo <= concentration <= c_hi:
            return c_lo, c_hi, i_lo, i_hi

    last_range = ranges[-1]
    if concentration > last_range[1]:
        return last_range

    return None


def _calculate_sub_index(pollutant: str, concentration: float) -> float | None:
    if concentration is None or pd.isna(concentration):
        return None

    bp = _find_breakpoint(pollutant, float(concentration))
    if bp is None:
        return None

    c_lo, c_hi, i_lo, i_hi = bp
    pollutant_aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (float(concentration) - c_lo) + i_lo
    return min(max(pollutant_aqi, 0.0), 500.0)


def get_aqi_category(aqi: float) -> str:
    if aqi is None or pd.isna(aqi):
        return "Insufficient data"

    for low, high, category in AQI_CATEGORIES:
        if low <= aqi <= high:
            return category
    return "Severe"


def get_health_advisory(aqi: float) -> str:
    category = get_aqi_category(aqi)
    return {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
        "Satisfactory": "Air quality is acceptable. Sensitive people should take care.",
        "Moderately Polluted": "People with respiratory issues should limit prolonged outdoor exposure.",
        "Poor": "Children and elderly should avoid outdoor exertion.",
        "Very Poor": "Avoid outdoor activities and use air purifiers if possible.",
        "Severe": "Stay indoors and seek medical advice if you experience symptoms.",
    }.get(category, "Insufficient data to provide health guidance.")


def calculate_current_aqi(values):
    if values is None:
        return None, "Insufficient data", "Cannot calculate current AQI."

    if isinstance(values, (list, tuple)):
        values = dict(zip(FEATURE_COLUMNS, values))

    pollutant_values = {p: values.get(p) for p in FEATURE_COLUMNS}
    valid_values = {p: float(v) for p, v in pollutant_values.items() if v is not None and not pd.isna(v)}

    if len(valid_values) < 3:
        return None, "Insufficient data", "Cannot calculate current AQI because fewer than three valid pollutant measurements are available."

    if all(p not in valid_values for p in ["pm25", "pm10"]):
        return None, "Insufficient data", "Cannot calculate current AQI without at least one PM2.5 or PM10 measurement."

    sub_indices = [
        _calculate_sub_index(pollutant, concentration)
        for pollutant, concentration in valid_values.items()
    ]
    sub_indices = [value for value in sub_indices if value is not None]

    if not sub_indices:
        return None, "Insufficient data", "No supported pollutant sub-indices could be calculated."

    aqi = float(max(sub_indices))
    category = get_aqi_category(aqi)
    advice = get_health_advisory(aqi)
    return aqi, category, advice


def predict_aqi(model, values):
    if model is None:
        return None, "N/A", "Model unavailable"

    if isinstance(values, dict):
        values = [values.get(col) for col in FEATURE_COLUMNS]

    sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    aqi = float(model.predict(sample)[0])
    category = get_aqi_category(aqi)
    advice = get_health_advisory(aqi)
    return aqi, category, advice