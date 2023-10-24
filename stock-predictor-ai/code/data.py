import os
import yfinance as yf
import pandas as pd
from fredapi import Fred

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

    fetch_and_save_yfinance(company_tickers, 'company')
    fetch_and_save_yfinance(forex_tickers, 'forex')
    fetch_and_save_yfinance(commodities_tickers, 'commodities')
    fetch_and_save_yfinance(fixed_income_tickers, 'fixed_income')

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

def save_gs_data(start, end):
    print(f"Downloading data for Goldman Sachs...")
    data = yf.download('GS', start=start, end=end)
    file_path = f'data/GS.csv'
    data.to_csv(file_path)

if __name__ == "__main__":
    save_gs_data()
    save_correlated_assets()
    fetch_economic_data()
