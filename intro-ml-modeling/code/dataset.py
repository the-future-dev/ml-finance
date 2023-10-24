import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def define_features_targets(data, window_size=5):
    """
    Define features and targets from the data.
    """
    data = data.copy()

    # 1. Predict Variation: Calculating the percentage change in closing price
    data['Pct_change'] = data['Adj Close'].pct_change() * 100  # multiply by 100 to convert to percentage

    # Moving Averages
    data['MA5'] = data['Adj Close'].rolling(window=5).mean()
    data['MA20'] = data['Adj Close'].rolling(window=20).mean()
    data['MA100'] = data['Adj Close'].rolling(window=100).mean()

    # RSI
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    data['EMA12'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 2. Normalize RSI
    data['RSI'] = data['RSI'] / 100  # RSI is usually between 0 and 100

    # 3. Distance from Moving Averages in percentage
    data['MA5_Dist_Z'] = (data['Adj Close'] - data['MA5']) / data['Adj Close'].rolling(window=5).std()
    data['MA20_Dist_Z'] = (data['Adj Close'] - data['MA20']) / data['Adj Close'].rolling(window=20).std()
    data['MA100_Dist_Z'] = (data['Adj Close'] - data['MA100']) / data['Adj Close'].rolling(window=100).std()

    # Bollinger Bands
    data['Middle_band'] = data['Adj Close'].rolling(window=20).mean()
    data['Upper_band'] = data['Middle_band'] + (data['Adj Close'].rolling(window=20).std() * 2)
    data['Lower_band'] = data['Middle_band'] - (data['Adj Close'].rolling(window=20).std() * 2)
    
    # Distance from Bollinger Bands in percentage
    data['Upper_band_Dist'] = ((data['Adj Close'] - data['Upper_band']) / data['Upper_band']) * 100
    data['Lower_band_Dist'] = ((data['Adj Close'] - data['Lower_band']) / data['Lower_band']) * 100

    # 4. Improve MACD Utilization: Binary MACD Signal
    data['MACD_Signal_Line'] = (data['MACD'] > data['MACD_signal']).astype(int)

    # 5. Z-score for Volume 
    data['Volume_Z'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()

    # Dropping Null Values
    initial_rows = len(data)
    data = data.dropna()
    rows_dropped = initial_rows - len(data)
    print(f"Due to Null values {rows_dropped} rows have been dropped.")

    # Feature & Target Definition
    features = [
                'Pct_change',
                'Volume_Z', # 0.0517
                # 'MA5_Dist_Z',
                # 'MA20_Dist_Z',
                # 'MA100_Dist_Z', 
                # 'RSI', # 0.06
                # 'MACD_Signal_Line',
                # 'Upper_band_Dist',
                # 'Lower_band_Dist'
                ]

    bins = [-np.inf, -0.3, 0.3, np.inf]
    labels = [0, 1, 2]

    X_data = []
    y_data = []
    for i in range(len(data) - window_size):
        X_data.append(data[features].iloc[i:i+window_size].values)
        next_pct_change = data['Pct_change'].iloc[i + window_size]
        # category = pd.cut([next_pct_change], bins=bins, labels=labels)[0]
        y_data.append(next_pct_change)

    X = np.array(X_data)
    y = np.array(y_data)

    return X, y

def fetch_and_save_data(ticker, start, end, filename):
    """
    Fetch data using yfinance and save it to a CSV file.
    """
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(filename)

def load_data(filename, start, end):
    """
    Load data from a CSV file for a specific period.

    Parameters:
    - filename (str): The name of the file to load the data from.
    - start (str): Start date (format: 'YYYY-MM-DD').
    - end (str): End date (format: 'YYYY-MM-DD').

    Returns:
    - data (DataFrame): A DataFrame containing the loaded data for the specified period.
    """
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data.loc[start:end]

# UNUSED
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def preprocess_data(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data = data.dropna()
    return data

def exploratory_data_analysis(data):
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA5'], label='5-day Moving Average')
    plt.plot(data['MA20'], label='20-day Moving Average')
    plt.plot(data['MA100'], label='100-day Moving Average')
    plt.legend()
    plt.show()

def data_split(X, y, train_size=0.7, val_size=0.1):
    """
    Splits the data into training, validation, and test sets.
    """
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    
    # First, split the data into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
    
    # Next, split the train+val data into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(train_size+val_size), random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_stats(data):
    """
    Calculate the mean and std dev of the data.
    
    Parameters:
    - data (DataFrame): Data to calculate stats for.
    
    Returns:
    - means (Series): Mean of each feature.
    - std_devs (Series): Standard deviation of each feature.
    """
    means = data.mean()
    std_devs = data.std()
    
    return means, std_devs

def normalize_data(data, means, std_devs):
    """
    Normalize the data.
    
    Parameters:
    - data (DataFrame): Data to be normalized.
    - means (Series): Mean of each feature.
    - std_devs (Series): Standard deviation of each feature.
    
    Returns:
    - data_normalized (DataFrame): Normalized data.
    """
    data_normalized = (data - means) / std_devs
    return data_normalized
