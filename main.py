from src.model import train_model
from src.predict import calculate_current_aqi, predict_aqi
from src.preprocess import FEATURE_COLUMNS, get_available_cities, get_city_latest_valid_record, load_processed_dataset

print("Loading valid AQI.csv-based measurements...")
df = load_processed_dataset()
try:
    model, metrics = train_model(df)
    print(f"Selected model: {metrics[0]}, MAE={metrics[1]:.2f}, RMSE={metrics[2]:.2f}, R2={metrics[3]:.4f}")
except ValueError as error:
    print(f"Model training unavailable: {error}")
    model = None

city = input("\nEnter City Name: ").strip()

latest_record = get_city_latest_valid_record(city)
if latest_record is None:
    print("No valid AQI data is available for this city in the current data source.")
    print("Supported cities:", ", ".join(get_available_cities()[:30]))
else:
    feature_values = {column: latest_record.get(column) for column in FEATURE_COLUMNS}
    current_aqi, current_category, current_advice = calculate_current_aqi(feature_values)

    print("\n--- RESULT ---")
    print("City:", latest_record.get("city"))
    print("Latest available measurement:", latest_record.get("last_update"))
    print("Current/source AQI:", round(current_aqi, 2) if current_aqi is not None else "Insufficient data")
    print("Current AQI Category:", current_category)
    print("Current Health Advisory:", current_advice)
    print("Pollutant measurements available:")
    print("PM2.5:", feature_values.get("pm25"))
    print("PM10:", feature_values.get("pm10"))
    print("NO2:", feature_values.get("no2"))
    print("SO2:", feature_values.get("so2"))
    print("CO:", feature_values.get("co"))
    print("O3:", feature_values.get("o3"))

    if model is None:
        print("Predicted AQI:", "N/A")
        print("Predicted AQI Category:", "N/A")
        print("Predicted Health Advisory:", "Model training is unavailable.")
    else:
        aqi, category, advice = predict_aqi(model, feature_values)
        print("Predicted AQI:", round(float(aqi), 2))
        print("Predicted AQI Category:", category)
        print("Predicted Health Advisory:", advice)