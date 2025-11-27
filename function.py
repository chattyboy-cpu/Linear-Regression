# function.py
def load_data(file_path="clean_WW2.csv"):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def train_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    
    # Create a copy to avoid modifying the original DataFrame
    data = data.copy()
    
    # Drop the Date column as it contains non-numeric string values
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])
    
    # Replace "T" (Trace) values with 0
    data = data.replace("T", 0)
    
    # Convert all columns to numeric types
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop any rows with NaNs after conversions
    data = data.dropna()
    
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {"model": model, "mse": mse}

