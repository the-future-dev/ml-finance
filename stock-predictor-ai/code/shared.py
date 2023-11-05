import pandas as pd
import matplotlib.pyplot as plt

def read_dataset(filename, start, end):
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data.loc[start:end]

def plot_results(price_actual_array, price_predicted_array, target_column_name):
    plt.figure(figsize=(15, 6))
    plt.plot(price_actual_array, label="Actual Price", color='blue')
    plt.plot(price_predicted_array, label="Predicted Price", color='red', linestyle='dashed')
    plt.title("Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel(target_column_name)
    plt.legend()

    plt.show()