import pandas as pd

def get_risk(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

def suggest(aqi):
    if aqi <= 50:
        return "Safe to go outside"
    elif aqi <= 100:
        return "Okay for outdoor activity"
    elif aqi <= 150:
        return "Limit outdoor activity"
    elif aqi <= 200:
        return "Avoid outdoor activity"
    else:
        return "Stay indoors and wear mask"

def predict_aqi(model, values):
    # Convert input to DataFrame
    sample = pd.DataFrame([values], columns=['pm25', 'pm10', 'no2', 'so2', 'co', 'o3'])

    aqi = model.predict(sample)[0]

    risk = get_risk(aqi)
    advice = suggest(aqi)

    return aqi, risk, advice