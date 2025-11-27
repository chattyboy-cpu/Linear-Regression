import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path= r"C:\Users\lamin\OneDrive\Desktop\Datafolder\clean_WW2.csv"):
    """
    Loads data from a CSV file.
    Added low_memory=False to suppress DtypeWarning for mixed types.
    """
    # Fix 1: Handle the DtypeWarning by setting low_memory=False
    data = pd.read_csv(file_path, low_memory=False)
    return data

# ...existing code...
def train_model(data):
    # Use specific columns as requested without extra cleaning
    X = data[['MinTemp']]
    y = data['MaxTemp']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate MSE on the training data since no split was requested
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    return {"model": model, "mse": mse, "r2": r2}

