import yfinance as yf
import pandas as pd

def new_stock(ticker):
    file_path = f'data/{ticker}.csv'
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(file_path)

if __name__ == "__main__":
    start = "2002-01-01"
    end = "2023-11-30"

    new_stock('DAL.MI')
