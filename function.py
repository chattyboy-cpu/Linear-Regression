import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(file_path="clean_WW2.csv"):
    """
    Loads data from a CSV file.
    Defaults to a relative path instead of a specific user directory.
    """
    data = pd.read_csv(file_path)
    return data

def train_model(data):
    data = data.copy()
    
    # Drop Date column if it exists
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])
        
    # Replace "T" values (Trace precipitation) with 0
    data = data.replace("T", 0)
    
    # Ensure numeric data
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    data = data.dropna()
    
    # Split features and target
    # Assuming the last column is the target variable
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    
    return {"model": model, "mse": mse}
