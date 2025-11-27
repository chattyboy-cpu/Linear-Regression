from functions import load_data, train_model

# Define the path to your data file
# Ensure this file exists in your project directory or provide the full path
filepath = "clean_WW2.csv" 

try:
    print(f"Loading data from {filepath}...")
    # Run the results
    data = load_data(filepath)
    
    print("Training model...")
    results = train_model(data)

    print("\n--- Results ---")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"Model: {results['model']}")

except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found. Please check the file name and path.")
except Exception as e:
    print(f"An error occurred: {e}")
