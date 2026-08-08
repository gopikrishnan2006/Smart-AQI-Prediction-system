import pandas as pd
import streamlit as st

from src.model import train_model
from src.predict import calculate_current_aqi, predict_aqi
from src.preprocess import FEATURE_COLUMNS, get_available_cities, get_city_latest_valid_record, load_processed_dataset

@st.cache_resource
def load_model():
    df = load_processed_dataset()
    try:
        model, _ = train_model(df)
        return model
    except ValueError:
        return None


st.title("Smart AQI Prediction System")
st.write("Enter a city to predict its air quality.")

city = st.text_input("City name")

if st.button("Predict AQI"):
    if not city:
        st.warning("Please enter a city name.")
    else:
        model = None
        try:
            with st.spinner("Loading model and city data..."):
                model = load_model()
                latest_record = get_city_latest_valid_record(city)
                if latest_record is None:
                    st.error("No valid AQI data is available for this city in the current data source.")
                    available_cities = get_available_cities()
                    if available_cities:
                        st.write("Supported cities include:")
                        st.write(", ".join(available_cities[:30]))
                    st.stop()

                feature_values = {column: latest_record.get(column) for column in FEATURE_COLUMNS}

                st.success("Loaded city measurements")
                st.write("City:", latest_record.get("city"))
                st.write("Latest available measurement:", latest_record.get("last_update"))

                current_aqi, current_category, current_advice = calculate_current_aqi(feature_values)
                st.write("Current/source AQI:", round(current_aqi, 2) if current_aqi is not None else "Insufficient data")
                st.write("Current AQI Category:", current_category)
                st.write("Current Health Advisory:", current_advice)

                st.write("Pollutant measurements available:")
                st.write("PM2.5:", feature_values.get("pm25"))
                st.write("PM10:", feature_values.get("pm10"))
                st.write("NO2:", feature_values.get("no2"))
                st.write("SO2:", feature_values.get("so2"))
                st.write("CO:", feature_values.get("co"))
                st.write("O3:", feature_values.get("o3"))

                if model is None:
                    prediction_aqi, prediction_category, prediction_advice = (None, "N/A", "Model training is unavailable.")
                else:
                    loaded = model
                    if isinstance(loaded, tuple):
                        loaded, _ = loaded
                    prediction_aqi, prediction_category, prediction_advice = predict_aqi(loaded, feature_values)

                st.write("Predicted AQI:", round(prediction_aqi, 2) if prediction_aqi is not None else "N/A")
                st.write("Predicted AQI Category:", prediction_category)
                st.write("Predicted Health Advisory:", prediction_advice)
        except Exception as error:
            st.error(str(error))