# function.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data(file_path="clean_WW2.csv"):
    data = pd.read_csv(file_path)
    return data


def train_model(data):
    # Drop the 'Date' column as it is non-numeric
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])

    # Replace 'T' (trace precipitation) with 0.0 and convert to numeric
    data = data.replace('T', 0.0)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {"model": model, "mse": mse}

