
from function import load_data, train_model
# Run the results
data = load_data("clean_WW2.csv")
results = train_model(data)

print(results)