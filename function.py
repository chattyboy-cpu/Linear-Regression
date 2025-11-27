# functions.py
def load_data(file_path="C:\\Users\\lamin\\OneDrive\\Desktop\\Datafolder\\clean_WW2.csv"):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def train_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    data = data.dropna()
    
    x= data.iloc[:, :-1]
    y= data.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {"model": model, "mse": mse}

