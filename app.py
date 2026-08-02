import streamlit as st
import pandas as pd

from src.model import train_model
from src.predict import predict_aqi
from src.api import get_pollution_data


@st.cache_resource
def load_model():
    df = pd.read_csv("data/india_city_aqi_2015_2023.csv")
    return train_model(df)


st.title("Smart AQI Prediction System")
st.write("Enter a city to predict its air quality.")

city = st.text_input("City name")

if st.button("Predict AQI"):
    if not city:
        st.warning("Please enter a city name.")
    else:
        try:
            model = load_model()
            values = get_pollution_data(city)
            aqi, risk, advice = predict_aqi(model, values)

            st.success("Prediction completed")
            st.write("City:", city)
            st.write("Predicted AQI:", aqi)
            st.write("Risk level:", risk)
            st.write("Advice:", advice)

        except Exception as error:
            st.error(str(error))