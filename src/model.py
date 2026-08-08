import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2


def train_model(df):
    required_columns = ["pm25", "pm10", "no2", "so2", "co", "o3", "last_update"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError("Training data must contain pm25, pm10, no2, so2, co, o3, and last_update columns")

    if "aqi" not in df.columns:
        raise ValueError("AQI target is missing from the source data; model training cannot proceed")

    training_df = df[["pm25", "pm10", "no2", "so2", "co", "o3", "aqi", "last_update"]].copy()
    training_df = training_df.dropna(subset=["aqi"]).copy()
    if training_df.empty:
        raise ValueError("Training data is empty after removing rows without an AQI target")

    training_df = training_df.sort_values("last_update").reset_index(drop=True)
    X = training_df[["pm25", "pm10", "no2", "so2", "co", "o3"]]
    y = training_df["aqi"]

    split_index = int(len(training_df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) < 2 or len(X_test) < 1:
        raise ValueError("Not enough rows to create a reliable time-aware split")

    candidates = {
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
        "GradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }

    best_model = None
    best_mae = None
    best_metrics = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        mae, rmse, r2 = _evaluate_model(model, X_test, y_test)
        if best_mae is None or mae < best_mae:
            best_model = model
            best_mae = mae
            best_metrics = (name, mae, rmse, r2)

    return best_model, best_metrics