from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    # Select features
    df = df[['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'aqi']]

    X = df[['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']]
    y = df['aqi']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model