import requests

API_KEY = "8934dc3022038af8285afc86c472ca16"

def get_pollution_data(city):

    # Step 1: Get latitude & longitude from city name
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    geo_res = requests.get(geo_url).json()

    lat = geo_res[0]['lat']
    lon = geo_res[0]['lon']

    # Step 2: Get pollution data
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    data = requests.get(url).json()

    comp = data['list'][0]['components']

    pm25 = comp['pm2_5']
    pm10 = comp['pm10']
    no2 = comp['no2']
    co = comp['co']
    o3 = comp['o3']
    so2 = comp.get('so2', 10)

    return [pm25, pm10, no2, so2, co, o3]