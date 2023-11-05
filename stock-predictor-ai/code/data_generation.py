import os
import yfinance as yf
import pandas as pd
from fredapi import Fred
import numpy as np
import matplotlib.pyplot as plt

with open('.env', 'r') as file:
    for line in file:
        key, value = line.strip().split('=', 1)
        os.environ[key] = value

fred = Fred(api_key= os.environ['FRED_API_KEY'])

def save_correlated_assets(start, end):
    """
    Fetch correlated assets using yfinance and save it to a CSV file.
    """
    def fetch_and_save_yfinance(company_tickers, file_name):
        combined_data = None
        for asset in company_tickers:
            try:
                print(f"Downloading data for {asset}...")
                data = yf.download(asset, start=start, end=end)
                data = data[['Close']]  # Keep only the closing prices
                data.rename(columns={'Close': asset}, inplace=True)  # Rename column to the asset name
                if combined_data is None:
                    combined_data = data  # If combined_data is None, assign the data
                else:
                    combined_data = pd.concat([combined_data, data], axis=1)  # Concat along columns axis
            except Exception as e:
                print(f"Failed to retrieve data for {asset}: {e}")

        file_path = f'data/{file_name}.csv'
        combined_data.to_csv(file_path)

    company_tickers = [
        'JPM', 'MS', 'BAC', 'C', 'WFC', 'BARC.L', 'DB', 'CS', 'UBS', 'HSBC',
    ]

    forex_tickers = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'CHFUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'CADUSD=X', 'HKDUSD=X', 'CNYUSD=X', 'EURJPY=X', 'EURGBP=X'
    ]
    commodities_tickers = [
        'GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'HG=F', 'ALI=F', 'VA=F', 'PL=F', 'PA=F'
    ]

    fixed_income_tickers = ['^TNX', '^IRX',
                            #'BUND10Y', 'GILT10Y', 'JGB10Y',
                            'LQD', 'MUB', 'HYG', 'EMB', 'MBB']

    # fetch_and_save_yfinance(company_tickers, 'company')
    # fetch_and_save_yfinance(forex_tickers, 'forex')
    # fetch_and_save_yfinance(commodities_tickers, 'commodities')
    # fetch_and_save_yfinance(fixed_income_tickers, 'fixed_income')
    fetch_and_save_yfinance(['VIX'], 'vix')

def fetch_economic_data():
    us_macroeconomic_ids = {
        'gross_domestic_product': 'GDP',
        'unemployment_rate': 'UNRATE',
        'inflation_rate': 'CPIAUCNS',
        'interest_rate': 'GS10',
        'consumer_price_index': 'CPIAUCSL',
        'producer_price_index': 'PPIACO',
        'balance_of_trade': 'BOPGSTB',
        # 'federal_surplus_or_deficit': 'FYFSD',  # annual
        'industrial_production_index': 'INDPRO',
        'consumer_confidence_index': 'UMCSENT'
    }

    data_frames = {}

    for name, series_id in us_macroeconomic_ids.items():
        series = fred.get_series(series_id).dropna()
        # Resample to daily frequency, forward-fill missing values
        daily_series = series.resample('D').ffill()
        data_frames[name] = daily_series

    # Merge the data into a single DataFrame, aligning by date
    merged_data = pd.DataFrame(data_frames)

    # Back-fill any remaining missing values (for days before the first available data point)
    merged_data.fillna(method='bfill', inplace=True)

    filepath = f'data/us_macro.csv'
    merged_data.to_csv(filepath)

    print(merged_data.head())

def save_gs_data(start, end, gs_path='data/GS.csv'):
    print(f"Downloading data for Goldman Sachs...")
    data = yf.download('GS', start=start, end=end)
    data['Close_Diff'] = data['Close'].diff()
    data.to_csv(gs_path)

def read_dataset(filename):
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data.loc[start:end]

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

def smooth(series, window_size=5):
    return series.rolling(window=window_size).mean().fillna(0)

def technical_indicators(gs_path='data/GS.csv'):
    file_path = gs_path
    GS = read_dataset(file_path)
    
    # Create 7 and 21 days Moving Average
    GS['ma7'] = GS['Adj Close'].rolling(window=7).mean()
    GS['ma21'] = GS['Adj Close'].rolling(window=21).mean()
    
    # Create MACD
    GS['26ema'] = GS['Adj Close'].ewm(span=26).mean()
    GS['12ema'] = GS['Adj Close'].ewm(span=12).mean()
    GS['MACD'] = (GS['12ema']-GS['26ema'])

    # Create Bollinger Bands
    GS['20sd'] = GS['Adj Close'].rolling(window=20).std()
    GS['upper_band'] = GS['ma21'] + (GS['20sd']*2)
    GS['lower_band'] = GS['ma21'] - (GS['20sd']*2)
    
    # Create Exponential moving average
    GS['ema'] = GS['Adj Close'].ewm(com=0.5).mean()
    
    # Create Momentum (using a period of 10 days for this example)
    GS['momentum'] = GS['Adj Close'] - GS['Adj Close'].shift(10)

    GS['Smoothed'] = smooth(GS['Adj Close'])
    GS['fourier_short'] = fourier_transform(GS['Smoothed'], 15)
    GS['fourier_medium'] = fourier_transform(GS['Smoothed'], 7)
    GS['fourier_long'] = fourier_transform(GS['Smoothed'], 3)

    GS.to_csv(file_path)

def plot_fourier(gs_path='data/GS.csv'):
    file_path = gs_path
    GS = read_dataset(file_path)
    plt.figure(figsize=(15,7))
    
    plt.plot(GS['Adj Close'], label='GS Adj Close', color='black')
    plt.plot(GS['Smoothed'], label='Smoothed', color='orange')
    plt.plot(GS['fourier_short'], label='Short Term Fourier', color='red')
    plt.plot(GS['fourier_medium'], label='Medium Term Fourier', color='green')
    plt.plot(GS['fourier_long'], label='Long Term Fourier', color='blue')
    
    plt.title('GS Adj Close & Fourier Transforms')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start = "2010-01-04"
    end = "2023-10-29"
    save_gs_data(start, end, 'data/GS.csv')
    # save_correlated_assets(start, end)
    # fetch_economic_data()
    technical_indicators('data/GS.csv')
    plot_fourier('data/GS.csv')
    
