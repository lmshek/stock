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
from collections import OrderedDict
from stock_utils import stock_utils
from stock_utils.bcolors import bcolors

def stoch_k_d(ticker, start_date = None, end_date = None, n = 10, take_profit_rate = 0.1, stop_loss_rate = 0.5):
    
    stock = yf.Ticker(ticker)
    if start_date:
        hist = stock.history(start = start_date.date(), end = end_date.date() + timedelta(days = 1))
    else:
        hist = stock.history(period="max")
    
    stoch = ta.momentum.stoch(high=hist['High'], low=hist['Low'], close=hist['Close'], k = 20, d = 5, smooth_k = 5)
    hist['STOCHk'] = stoch['STOCHk_20_5_5']
    hist['STOCHd'] = stoch['STOCHd_20_5_5']
    hist['STOCHk_avg'] = hist['STOCHk'].rolling(window=20).mean()
    hist['STOCHd_avg'] = hist['STOCHd'].rolling(window=20).mean()

    """
    macd = ta.momentum.macd(hist["Close"])
    hist['MACD'] = macd['MACD_12_26_9']
    hist['MACDh'] = macd['MACDh_12_26_9']
    hist['MACDs'] = macd['MACDs_12_26_9']   
    
    if hist['MACDh'][-1] < 0:
        last_positive_index = max(idx for idx, value in enumerate(hist['MACDh'][1:-1]) if value > 0)
        values_slice = hist['MACDh'][last_positive_index + 1:]
        values_slice[values_slice > 0] = 0
        hist['MACD_energy'] = np.cumsum(abs(values_slice))
    else:
        last_negative_index = max(idx for idx, value in enumerate(hist['MACDh'][1:-1]) if value < 0)
        values_slice = hist['MACDh'][last_negative_index + 1:]
        values_slice[values_slice < 0] = 0
        hist['MACD_energy'] = np.cumsum(values_slice)
    """

    #hist['RSI_14'] = hist.ta.rsi(close=hist['Close'])
    
    #bbands = hist.ta.bbands(close=hist['Close'], length=20)
    #hist['BBL_20_2.0'] = bbands['BBL_20_2.0']
    #hist['BBM_20_2.0'] = bbands['BBM_20_2.0']
    #hist['BBU_20_2.0'] = bbands['BBU_20_2.0']
    #hist['BBB_20_2.0'] = bbands['BBB_20_2.0']
    #hist['BBP_20_2.0'] = bbands['BBP_20_2.0']

    #hist['MFI'] = ta.volume.mfi(high=hist['High'], low=hist['Low'], close=hist['Close'], volume=hist['Volume'])
        
    target = hist['STOCHd'].shift(1)[-1] > hist['STOCHk'].shift(1)[-1] \
        and hist['STOCHk'][-1] > hist['STOCHd'][-1] 
    
    
    
        
    multiplier = 1
    
    if len(hist['Close'].values) == 0:
        raise (f"No Stock data for {stock} between {start_date} and {end_date}")
    

    return target, hist['Close'].values[-1], hist.tail(1), multiplier

def stoch_print(today_data):
    print(f'{bcolors.OKCYAN}Stoch K: {today_data["STOCHk"][-1]} Stoch D: {today_data["STOCHd"][-1]}{bcolors.ENDC}')

def take_first(elem):
    return elem[1][1]['STOCHk_avg'][0] - elem[1][1]['STOCHd_avg'][0]

def stoch_order(stocks):
    return OrderedDict(sorted(stocks, key = take_first, reverse = False))
    


def stoch_sell(ticker, buy_date, buy_price, todays_date, sell_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    try :
        start_date = todays_date - timedelta(days = 100)
        end_date = todays_date

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.date(), end=end_date.date() + timedelta(days=1))
        current_price = hist['Close'].values[-1]

        stoch = ta.momentum.stoch(high=hist['High'], low=hist['Low'], close=hist['Close'], k = 20, d = 5, smooth_k = 5)
        hist['STOCHk'] = stoch['STOCHk_20_5_5']
        hist['STOCHd'] = stoch['STOCHd_20_5_5']

        #macd = ta.momentum.macd(hist["Close"])
        #hist['MACD'] = macd['MACD_12_26_9']
        #hist['MACDh'] = macd['MACDh_12_26_9']
        #hist['MACDs'] = macd['MACDs_12_26_9']    
        
        sell_price = buy_price + buy_price * sell_perc        
        stop_price = buy_price - buy_price * stop_perc
        sell_date = stock_utils.get_market_real_date(buy_date, hold_till) # selling date
        today = todays_date
        time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

        if (current_price is not None):
            if (hist['STOCHd'].values[-1] > hist['STOCHk'].values[-1] \
                #and hist['STOCHd'].shift(2).values[-1] - hist['STOCHk'].shift(2).values[-1] < hist['STOCHd'].shift(1).values[-1] - hist['STOCHk'].shift(1).values[-1] \
                #and hist['STOCHd'].shift(1).values[-1] - hist['STOCHk'].shift(1).values[-1] < hist['STOCHd'].values[-1] - hist['STOCHk'].values[-1] \
            ):
                return "SELL:turning_point", current_price #if criteria is met recommend to sell
            elif (current_price < stop_price):
                return "SELL:stop_loss", current_price #if criteria is met recommend to sell
            elif (current_price >= sell_price):
                return "SELL:take_profit", current_price #if criteria is met recommend to sell
            elif (current_price >= buy_price * (1 + sell_perc * 0.8)  and stock_utils.get_market_days(buy_date, todays_date) > hold_till / 2):
                return "SELL:take_profit_at_0.8_more_than_half_of_the_journey", current_price #if criteria is met recommend to sell
            elif (todays_date >= sell_date and (hist['STOCHd'].shift(1).values[-1] > hist['STOCHk'].shift(1).values[-1]) and (hist['STOCHk'].values[-1] > hist['STOCHd'].values[-1])):
                return "SELL:already_matured_and_turning_point", current_price #if criteria is met recommend to sell
            else:
                return "HOLD", current_price #if criteria is not met hold the stock
        else:
            return "HOLD", current_price #if criteria is not met hold the stock
    except:
        return "HOLD", 0 #if criteria is not met hold the stock
