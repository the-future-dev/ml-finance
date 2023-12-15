import os
import yfinance as yf
import pandas as pd
from fredapi import Fred
import numpy as np
import matplotlib.pyplot as plt
import eurostat

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
                # data = data[['Close']]  # Keep only the closing prices
                # data.rename(columns={'Close': asset}, inplace=True)  # Rename column to the asset name
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


def read_dataset(filename):
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data.loc[start:end]

# def fourier_transform(series, n_harmonics):
#     # Detrend the series (remove mean)
#     detrended_series = series - series.mean()
    
#     # Apply Fast Fourier Transform
#     t = np.fft.fft(detrended_series)
#     frequencies = np.fft.fftfreq(detrended_series.size)
    
#     # Zero out all but the n_harmonics largest magnitude frequencies
#     indices = np.argsort(np.abs(t))
#     t[indices[:-n_harmonics]] = 0
    
#     # Apply Inverse Fast Fourier Transform
#     filtered_series = np.fft.ifft(t)
    
#     # Add the mean value back
#     return filtered_series.real + series.mean()

def technical_indicators(stock_path='data/GS.csv'):
    file_path = stock_path
    df = read_dataset(file_path)
    
    # Create 7 and 21 days Moving Average
    df['ma7'] = df['Adj Close'].rolling(window=7).mean()
    df['ma21'] = df['Adj Close'].rolling(window=21).mean()
    
    # Create MACD
    df['26ema'] = df['Adj Close'].ewm(span=26).mean()
    df['12ema'] = df['Adj Close'].ewm(span=12).mean()
    df['MACD'] = (df['12ema']-df['26ema'])

    # Create Bollinger Bands
    df['20sd'] = df['Adj Close'].rolling(window=20).std()
    df['upper_band'] = df['ma21'] + (df['20sd']*2)
    df['lower_band'] = df['ma21'] - (df['20sd']*2)
    
    # Create Exponential moving average
    df['ema'] = df['Adj Close'].ewm(com=0.5).mean()
    
    # Create Momentum (using a period of 10 days for this example)
    df['momentum'] = df['Adj Close'] - df['Adj Close'].shift(10)

    df['Smoothed'] = smooth(df['Adj Close'])
    df['fourier_short'] = fourier_transform(df['Smoothed'], 15)
    df['fourier_medium'] = fourier_transform(df['Smoothed'], 7)
    df['fourier_long'] = fourier_transform(df['Smoothed'], 3)

    df['Volatility_21'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).rolling(window=21).std() * np.sqrt(21)
    df['Close_Diff'] = df['Close'].diff()

    df.to_csv(file_path)

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

def save_macro_us(path):
    # FRED
    def fetch_fred_series(series_id):
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        return series
    # inflation_rate_data = fred.get_series('CPIAUCSL', observation_start=start, observation_end=end)  # CPI for all urban consumers
    # Fetch data from FRED
    us_interest_rate = fetch_fred_series('FEDFUNDS')  # Effective Federal Funds Rate
    us_gdp = fetch_fred_series('GDP')  # Gross Domestic Product
    us_inflation = fetch_fred_series('T5YIFR')  # 5-Year Breakeven Inflation Rate
    us_unemployment = fetch_fred_series('UNRATE')  # Unemployment Rate
    us_consumer_confidence = fetch_fred_series('UMCSENT')  # University of Michigan Consumer Sentiment Index
    us_yield_curve = fetch_fred_series('T10Y2Y')  # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    us_housing_starts = fetch_fred_series('HOUST')  # Housing Starts
    us_corporate_profits = fetch_fred_series('CP')  # Corporate Profits After Tax

    macro_data = pd.DataFrame({
        'US_Interest_Rate': us_interest_rate,
        'US_GDP': us_gdp,
        'US_Inflation': us_inflation,
        'US_Unemployment': us_unemployment,
        'US_Consumer_Confidence': us_consumer_confidence,
        'US_Yield_Curve': us_yield_curve,
        'US_Housing_Starts': us_housing_starts,
        'US_Corporate_Profits': us_corporate_profits,
    })

    macro_data.to_csv(path)

def new_stock(ticker):
    file_path = f'data/{ticker}.csv'
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(file_path)


def technical_indicators_price_change(stock_path='data/GS.csv'):
    file_path = stock_path
    df = read_dataset(file_path)
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']

    # Distance from 7, 21 days Moving Average
    df['ma7'] = df['Adj Close'] - df['Adj Close'].rolling(window=7).mean()
    df['ma21'] = df['Adj Close'] - df['Adj Close'].rolling(window=21).mean()
    
    # Create MACD
    df['26ema'] = df['Adj Close'] - df['Adj Close'].ewm(span=26).mean()
    df['12ema'] = df['Adj Close'] - df['Adj Close'].ewm(span=12).mean()
    df['MACD'] = (df['Adj Close'].ewm(span=12).mean()-df['Adj Close'].ewm(span=26).mean())

    # Create Bollinger Bands
    df['20sd'] = df['Adj Close'].rolling(window=20).std()
    df['upper_band'] = df['Adj Close'].rolling(window=21).mean() + (df['20sd']*2) - df['Adj Close']
    df['lower_band'] = df['Adj Close'].rolling(window=21).mean() - (df['20sd']*2) - df['Adj Close']
    
    # Create Exponential moving average
    df['ema'] = df['Adj Close'].ewm(com=0.5).mean() - df['Adj Close']
    
    # Create Momentum (using a period of 10 days for this example)
    df['momentum'] = df['Adj Close'] - df['Adj Close'].shift(10)

    df['Smoothed'] = smooth(df['Adj Close']) - df['Adj Close']
    df['fourier_short'] = fourier_transform(smooth(df['Adj Close']), 15)
    df['fourier_medium'] = fourier_transform(smooth(df['Adj Close']), 7)
    df['fourier_long'] = fourier_transform(smooth(df['Adj Close']), 3)

    df['Volatility_21'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).rolling(window=21).std() * np.sqrt(21)
    df['Close_Diff'] = df['Adj Close'].diff()

    df['Open_Diff'] = df['Open'].diff()

    df.to_csv(file_path)

def fourier_transform(series, n_harmonics):
    detrended_series = series - series.mean()
    t = np.fft.fft(detrended_series)
    frequencies = np.fft.fftfreq(detrended_series.size)
    
    indices = np.argsort(np.abs(frequencies))
    t[indices[n_harmonics:]] = 0
    
    filtered_series = np.fft.ifft(t)
    return filtered_series.real + series.mean()

def smooth(series, window_size=5):
    return series.rolling(window=window_size).mean().fillna(0)

def technical_indicators_ta(ticker):
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

    # df['ATR'] = calculate_average_true_range(df, 14)
    # df['RSI'] = calculate_RSI(df, 14)
    # df['OBV'] = calculate_on_balance_volume(df)

    # Refactor High, Low, Close to be in function of Open
    # df['High'] = df['High'] - df['Open']
    # df['Low'] = df['Low'] - df['Open']
    # df['Open-Close'] = df['Open'] - df['Close']
    df = df.drop('Adj Close', axis=1)

    df.to_csv(file_path)

def diff_easy(ticker):
    stock_path = f'data/{ticker}_ta.csv'
    df = pd.read_csv(stock_path, index_col=0, parse_dates=True)

    # df['High Diff'] = df['High'].diff()
    df['Low_diff'] = df['Low'].diff()

    df.to_csv(stock_path)


if __name__ == "__main__":
    start = "2001-01-01"
    end = "2023-12-07"

    new_stock('BTC')
    technical_indicators_ta('BTC')
    diff_easy('AZM.MI')
