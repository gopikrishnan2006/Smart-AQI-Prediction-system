import pandas as pd
from src.model import train_model
from src.predict import predict_aqi
from src.api import get_pollution_data

# Load dataset
df = pd.read_csv("data/india_city_aqi_2015_2023.csv")

# Train model
print("Training model...")
model = train_model(df)

# Ask user for city
city = input("\nEnter City Name: ")

# Get live pollution data
values = get_pollution_data(city)

print("\nLive Pollution Data:", values)

# Predict AQI
aqi, risk, advice = predict_aqi(model, values)

# Output
print("\n--- RESULT ---")
print("City:", city)
print("Predicted AQI:", aqi)
print("Risk Level:", risk)
print("Advice:", advice)