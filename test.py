import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, time
import pandas_datareader as web
import pandas_ta as ta
import matplotlib.pyplot as plt
from stock_utils.chart import Chart
import pprint
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt




if __name__ == '__main__':
    ticker = '0011.HK'

    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    n_days = 30

    # Get the last n days of the Close price
    last_n_days = df["Close"].tail(n_days)

    # Find the indices of the local maxima in the last n days
    maxima_indices = argrelextrema(last_n_days.values, np.greater)[0]

    # Find the values of the local maxima in the last n days
    maxima_values = last_n_days.values[maxima_indices]

    # Sort the local maxima in descending order
    sorted_maxima = sorted(maxima_values, reverse=True)

    # Get the first and second local maxima
    first_maxima = sorted_maxima[0]
    second_maxima = sorted_maxima[1]

    # Print the first and second local maxima
    print("First local maximum in the last", n_days, "days:", first_maxima , " on ", last_n_days.index[maxima_indices[maxima_values == first_maxima]][0])
    print("Second local maximum in the last", n_days, "days:", second_maxima, " on ", last_n_days.index[maxima_indices[maxima_values == second_maxima]][0])

    """

    # Define different window sizes to check for breakouts
    windows = [5, 10, 50, 100, 250]
    for window in windows:
        df[f'{window}_days_high'] = df["High"].rolling(window=window).max()
        df[f'{window}_days_low'] = df["Low"].rolling(window=window).min()

    # 10_days_after_close
    df[f'next_10_days_hign'] = df["Close"].rolling(window=10).max().shift(-10)
    df[f'profit'] = (df[f'next_10_days_hign'] - df['Close'] ) / df['Close']

    # Loop through each window size and check for breakouts
    df.to_csv('my_data.csv', index=True)
    for window in windows:

        # Detect breakouts
        for i in range(df.shape[0]):
            if df["Close"][i] > df[f'{window}_days_high'][i]:
                print(f'Upper breakout detected at ${df["Close"][i]} on {df.index[i]} for ${df[f"{window}_days_high"][i]} with window size {window} and profit = {df["profit"][i]}')
            elif df["Close"][i] < df[f'{window}_days_low'][i]:
                print(f'Lower breakout detected at ${df["Close"][i]} on {df.index[i]} for ${df[f"{window}_days_low"][i]} with window size {window} and profit = {df["profit"][i]}')
    """
   