import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def read_dataset(filename, start, end):
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data.loc[start:end]

def plot_results(price_actual_array, price_predicted_array, target_column_name, title="Actual vs Predicted"):
    plt.figure(figsize=(15, 6))
    plt.plot(price_actual_array, label="Actual Price", color='blue')
    plt.plot(price_predicted_array, label="Predicted Price", color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(target_column_name)
    plt.legend()

    plt.show()

def scatter_results(price_actual_array, price_predicted_array, target_column_name, title="Actual vs Predicted"):
    mean_abs_error(price_actual_array, price_predicted_array)
    plt.figure(figsize=(15, 6))
    plt.scatter(range(len(price_actual_array)), price_actual_array, label="Actual Price", color='blue')
    plt.scatter(range(len(price_predicted_array)), price_predicted_array, label="Predicted Price", color='red', alpha=0.5)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(target_column_name)
    plt.legend()

    plt.show()

def mean_abs_error(price_actual_array, price_predicted_array):
    mae = np.mean(np.abs(price_predicted_array - price_actual_array))
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

def evaluate_price_predictions(price_actual_array, price_predicted_array):
    """ 
    Evaluation function: given two arrays, one for the price prediction and one for the actual price,
    evaluates the performance using various metrics.
    """
    # should this be included?
    price_actual_array = np.array(price_actual_array)
    price_predicted_array = np.array(price_predicted_array)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(price_predicted_array - price_actual_array))
    
    # Mean Squared Error (MSE)
    mse = np.mean((price_predicted_array - price_actual_array)**2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((price_actual_array - price_predicted_array) / price_actual_array)) * 100
    
    # R-squared (R²)
    # Total sum of squares (SST) and residual sum of squares (SSR)
    sst = np.sum((price_actual_array - np.mean(price_actual_array))**2)
    ssr = np.sum((price_actual_array - price_predicted_array)**2)
    r_squared = 1 - (ssr/sst)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared (R²): {r_squared:.2f}")

def get_data(start, end, target_column_name='Adj Close', SEQUENCE_LENGTH=17):
    GS = read_dataset('../data/GS.csv', start, end)
    vix = read_dataset('../data/vix.csv', start, end)
    us_macro = read_dataset('../data/us_macro.csv', start, end)
    commodities = read_dataset('../data/commodities.csv', start, end)
    forex = read_dataset('../data/forex.csv', start, end)

    GS['Close_Diff'] = GS['Close'].diff()
    GS.dropna(inplace=True) # initially NAN for moving averages

    data = pd.concat([
        us_macro,
        GS,
        vix
        ], axis=1)
    data.dropna(inplace=True)

    cols = [target_column_name] + [ col for col in GS if col != target_column_name]
    target_column = list(GS.columns).index(target_column_name)
    data = GS[cols]

    # Define feature array and target array to train the model.
    data_array = np.array(data.values)
    target_array = np.array(data[target_column_name].values).reshape(-1, 1)

    # Normalize the data
    scaler_data = MinMaxScaler()
    scaler_data.fit(data_array)
    data_array = scaler_data.transform(data_array)

    scaler_target = MinMaxScaler()
    scaler_target.fit(target_array)
    target_array = scaler_target.transform(target_array)

    # Split the data
    train_size = int(len(data_array) * 0.70)
    evaluation_size = int(len(data_array) * 0.0) # no evaluation

    def create_sequences(data, target, seq_length):
        sequence_data = []
        sequence_target = []
        for i in range(seq_length, len(data)):
            sequence_data.append(data[i-seq_length:i])
            sequence_target.append(target[i])
        return np.array(sequence_data), np.array(sequence_target)

    data_sequences, target_sequences = create_sequences(data_array, target_array, SEQUENCE_LENGTH)

    return data_sequences, target_sequences

## TESTING
# price_actual_array = np.array([100, 100, 100, 100, 100])
# price_predicted_array = np.array([101, 98, 101, 101, 101])
# evaluate_price_predictions(price_actual_array, price_predicted_array)
