import yfinance as yf
import pandas as pd
import numpy as np

def new_stock(ticker):
    file_path = f'data/{ticker}.csv'

    data = yf.download(ticker, period='30d', interval='5m')
    
    data.to_csv(file_path)

def technical_indicators(ticker):
    """
    Techincal indicators: definition;
    Then distance from 'Close'.
    """
    stock_path = f'data/{ticker}.csv'
    file_path = f'data/{ticker}_ta.csv'
    df = pd.read_csv(stock_path, index_col=0, parse_dates=True)

    # Create 7 and 21 days Moving Average
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma7_diff'] = df['ma7'] - df['Close']
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['ma21_diff'] = df['ma21'] - df['Close']
    
    # Create MACD
    df['26ema'] = df['Close'].ewm(span=26).mean()
    df['26ema_diff'] = df['26ema'] - df['Close']
    df['12ema'] = df['Close'].ewm(span=12).mean()
    df['12ema_diff'] = df['12ema'] - df['Close']
    df['MACD'] = (df['12ema']-df['26ema'])

    # Create Bollinger Bands
    sd20 = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['ma21'] + (sd20*2)
    df['lower_band'] = df['ma21'] - (sd20*2)
        
    # Create Momentum (using a period of 10 days for this example)
    df['momentum'] = df['Close'] - df['Close'].shift(10)

    smoothed = df['Close'].rolling(window=5).mean().fillna(0)
    df['fourier_short'] = fourier_transform(smoothed, 15)
    df['fourier_medium'] = fourier_transform(smoothed, 7)
    df['fourier_long'] = fourier_transform(smoothed, 3)

    df['Volatility_21'] = np.log(df['Close'] / df['Close'].shift(1)).rolling(window=21).std() * np.sqrt(21)
    df['Close Diff'] = df['Close'].diff()
    df['Open Diff'] = df['Open'].diff()

    df['ATR'] = calculate_average_true_range(df, 14)
    # df['RSI'] = calculate_RSI(df, 14)
    df['OBV'] = calculate_on_balance_volume(df)

    # Refactor High, Low, Close to be in function of Open
    # df['High'] = df['High'] - df['Open']
    # df['Low'] = df['Low'] - df['Open']
    # df['Open-Close'] = df['Open'] - df['Close']
    df = df.drop('Adj Close', axis=1)

    df.to_csv(file_path)

def fourier_transform(series, n_harmonics):
    # Detrend the series (remove mean)
    detrended_series = series - series.mean()
    
    # Apply Fast Fourier Transform
    t = np.fft.fft(detrended_series)
    frequencies = np.fft.fftfreq(detrended_series.size)
    
    # Zero out all but the n_harmonics largest magnitude frequencies
    indices = np.argsort(np.abs(t))
    t[indices[:-n_harmonics]] = 0
    
    # Apply Inverse Fast Fourier Transform
    filtered_series = np.fft.ifft(t)
    
    # Add the mean value back
    return filtered_series.real + series.mean()

def calculate_average_true_range(df, period=14):
    """
    The Average True Range (ATR) is a measure of volatility.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_RSI(series, period=14):
    delta = series.diff().dropna()
    gains = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    losses = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gains / losses
    return 100 - (100 / (1 + RS))

def calculate_on_balance_volume(df):
    """
    On Balance Volume (OBV)
    """
    return (np.sign(df['Close'].diff()) * df['Volume']).cumsum()


if __name__ == "__main__":
    new_stock('PLTR')
    technical_indicators('PLTR')