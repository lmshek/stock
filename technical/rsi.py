import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, time
import pandas_datareader as web
import pandas_ta as ta

def rsi14(ticker, start_date = None, end_date = None, n = 10, take_profit_rate = 0.1, stop_loss_rate = 0.5):
    
    stock = yf.Ticker(ticker)
    if start_date:
        hist = stock.history(start = start_date, end = end_date)
    else:
        hist = stock.history(period="max")
    
    # Create your own Custom Strategy
    """
    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="RSI, MFI",
        ta=[
            {"kind": "rsi"},
            {"kind": "mfi", "window": 20}            
        ]
    )
    # To run your "Custom Strategy"
    hist.ta.strategy(CustomStrategy)
    """
    hist['RSI_14'] = hist.ta.rsi(close=hist['Close'])
    hist['MFI'] = ta.volume.mfi(high=hist['High'], low=hist['Low'], close=hist['Close'], volume=hist['Volume'])

    hist['target'] = np.logical_and(hist.RSI_14[-1] < 30, hist['MFI'].shift(1)[-1] < hist['MFI'][-1])

    return hist['target'][-1], hist['Close'].values[-1], hist.tail(1)

def get_stock_price(ticker, date):
    start_date = date - timedelta(days = 10)
    end_date = date

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date + timedelta(days=1))

    return hist['Close'].values[-1]

def rsi_sell(stock, buy_date, buy_price, todays_date, sell_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    current_price = get_stock_price(stock, todays_date)
    sell_price = buy_price + buy_price * sell_perc
    stop_price = buy_price - buy_price * stop_perc
    sell_date = buy_date + timedelta(days = hold_till) # selling date
    time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

    if (current_price is not None) and ((current_price < stop_price) or (current_price >= sell_price) or (todays_date >= sell_date)):
        return "SELL", current_price #if criteria is met recommend to sell
    else:
        return "HOLD", current_price #if criteria is not met hold the stock
