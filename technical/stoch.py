import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
import pandas_datareader as web
import pandas_ta as ta

def stoch_k_d(ticker, start_date = None, end_date = None, n = 10, take_profit_rate = 0.1, stop_loss_rate = 0.5):
    
    stock = yf.Ticker(ticker)
    if start_date:
        hist = stock.history(start = start_date.date(), end = end_date.date() + timedelta(days = 1))
    else:
        hist = stock.history(period="max")
    
    stoch = ta.momentum.stoch(high=hist['High'], low=hist['Low'], close=hist['Close'], k = 20, d = 5, smooth_k = 5)
    hist['STOCHk'] = stoch['STOCHk_20_5_5']
    hist['STOCHd'] = stoch['STOCHd_20_5_5']
    #hist['STOCHk_acceleration'] = (hist['STOCHk'].shift(1) - hist['STOCHk']) - (hist['STOCHk'].shift(2) - hist['STOCHk'].shift(1))

    #macd = ta.momentum.macd(hist["Close"])
    #hist['MACD'] = macd['MACD_12_26_9']
    #hist['MACDh'] = macd['MACDh_12_26_9']
    #hist['MACDs'] = macd['MACDs_12_26_9']    

    #hist['RSI_14'] = hist.ta.rsi(close=hist['Close'])
    
    #bbands = hist.ta.bbands(close=hist['Close'], length=20)
    #hist['BBL_20_2.0'] = bbands['BBL_20_2.0']
    #hist['BBM_20_2.0'] = bbands['BBM_20_2.0']
    #hist['BBU_20_2.0'] = bbands['BBU_20_2.0']
    #hist['BBB_20_2.0'] = bbands['BBB_20_2.0']
    #hist['BBP_20_2.0'] = bbands['BBP_20_2.0']

    #hist['MFI'] = ta.volume.mfi(high=hist['High'], low=hist['Low'], close=hist['Close'], volume=hist['Volume'])
    
    #hist['k_larger_d'] = hist['STOCHk'] > hist['STOCHd']
    #hist['d_larger_k_1_days_before'] = hist['STOCHd'].shift(1) > hist['STOCHk'].shift(1)
    #hist['volume_than_3days_average'] = hist['Volume'] > hist.shift(1).rolling(3).Volume.mean()
    #hist['RSI_14'] = hist.ta.rsi(close=hist['Close'])
    
    target = hist['STOCHd'].shift(3)[-1] > hist['STOCHk'].shift(3)[-1] \
        and hist['STOCHd'].shift(2)[-1] > hist['STOCHk'].shift(2)[-1] \
        and hist['STOCHk'].shift(1)[-1] > hist['STOCHd'].shift(1)[-1] \
        and hist['STOCHk'][-1] > hist['STOCHd'][-1] \
        
    if len(hist['Close'].values) == 0:
        raise (f"No Stock data for {stock} between {start_date} and {end_date}")
    

    return target, hist['Close'].values[-1], hist.tail(1)

def get_stock_price(ticker, date):
    start_date = date - timedelta(days = 10)
    end_date = date

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date.date(), end=end_date.date() + timedelta(days=1))

    return hist['Close'].values[-1]

def stoch_sell(stock, buy_date, buy_price, todays_date, sell_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    current_price = get_stock_price(stock, todays_date)
    sell_price = buy_price + buy_price * sell_perc
    stop_price = buy_price - buy_price * stop_perc
    sell_date = buy_date + timedelta(days = hold_till) # selling date
    time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

    if (current_price is not None) and ((current_price < stop_price) or (current_price >= sell_price) or (todays_date >= sell_date)):
        return "SELL", current_price #if criteria is met recommend to sell
    else:
        return "HOLD", current_price #if criteria is not met hold the stock
