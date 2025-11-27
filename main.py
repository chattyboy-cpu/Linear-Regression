
from functions import load_data, train_model
# Run the results
data = load_data(filepath)
results = train_model(data)

print(results)