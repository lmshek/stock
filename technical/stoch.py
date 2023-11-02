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

def stoch_k_d(all_time_stock_data, start_date = None, end_date = None):
    
    hist = all_time_stock_data.loc[start_date.date():end_date.date()]       

    
    stoch = ta.momentum.stoch(high=hist['High'], low=hist['Low'], close=hist['Close'], k = 20, d = 5, smooth_k = 5)
    hist['STOCHk'] = stoch['STOCHk_20_5_5']
    hist['STOCHd'] = stoch['STOCHd_20_5_5']

    target = hist['STOCHd'].shift(2)[-1] > hist['STOCHk'].shift(2)[-1] \
        and hist['STOCHk'].shift(1)[-1] > hist['STOCHd'].shift(1)[-1] \
        and hist['STOCHk'][-1] > hist['STOCHd'][-1] 

    
    if len(hist['Close'].values) == 0:
        raise (f"No Stock data between {start_date} and {end_date}")
    
    # Find the sum of (d - k) from the last crossing point of d and k to now
    if target:        
        potential = 0
        for i in range(3, len(hist)):
            if hist['STOCHd'][-i] > hist['STOCHk'][-i]:
                potential += (hist['STOCHd'][-i] - hist['STOCHk'][-i]) * (hist['STOCHk'][-i] // 10)
            else:
                hist['potential'] = potential
                break

        
        
    

    return target, hist['Close'].values[-1], hist.tail(1), 1

def stoch_print(simulator, today_data):
    simulator.log(f'{bcolors.OKCYAN}Stoch K: {today_data["STOCHk"][-1]} Stoch D: {today_data["STOCHd"][-1]} Potential: {today_data["potential"][-1]}{bcolors.ENDC}')

def take_first(elem):
    return elem[1][1]['potential'][0]

def stoch_order(stocks):
    return OrderedDict(sorted(stocks, key = take_first, reverse = True))

def stoch_sell(stock_data, market, ticker, buy_date, buy_price, todays_date, profit_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    try :        
        hist = stock_data[ticker]
        
        today = todays_date.date()
        yesterday = stock_utils.get_market_real_date(market, todays_date, -1).date()
        day_before_yesterday = stock_utils.get_market_real_date(market, todays_date, -2).date()

        stoch = ta.momentum.stoch(high=hist['High'], low=hist['Low'], close=hist['Close'], k = 20, d = 5, smooth_k = 5)

        current_price = hist['Close'][today:today].values[-1]
        stoch_k = stoch['STOCHk_20_5_5'][today:today].values[-1]
        stoch_k_day_minus_1 = stoch['STOCHk_20_5_5'][yesterday:yesterday].values[-1]
        stoch_k_day_minus_2 = stoch['STOCHk_20_5_5'][day_before_yesterday:day_before_yesterday].values[-1]
        stoch_d = stoch['STOCHd_20_5_5'][today:today].values[-1]
        stoch_d_day_minus_1 = stoch['STOCHd_20_5_5'][yesterday:yesterday].values[-1]
        stoch_d_day_minus_2 = stoch['STOCHd_20_5_5'][day_before_yesterday:day_before_yesterday].values[-1]
        
        sell_price = buy_price + buy_price * profit_perc        
        stop_price = buy_price - buy_price * stop_perc
        sell_date = stock_utils.get_market_real_date(market, buy_date, hold_till) # selling date        
        time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

        if (current_price is not None):
            if current_price > buy_price:
                if (stoch_d > stoch_k \
                    #and stoch_d_day_minus_2 - stoch_k_day_minus_2 < stoch_d_day_minus_1 - stoch_k_day_minus_1 \
                    #and stoch_d_day_minus_1 - stoch_k_day_minus_1 < stoch_d - stoch_k \
                ):
                    return "SELL:turning_point", current_price #if criteria is met recommend to sell
                elif (current_price < stop_price):
                    return "SELL:stop_loss", current_price #if criteria is met recommend to sell
                elif (current_price >= sell_price):
                    return "SELL:take_profit", current_price #if criteria is met recommend to sell
                elif (current_price >= buy_price * (1 + profit_perc * 0.8)  and stock_utils.get_market_days(buy_date, todays_date) <= hold_till / 2):
                    return "SELL:take_profit_at_0.8_more_than_half_of_the_journey", current_price #if criteria is met recommend to sell
                elif (todays_date >= sell_date and (stoch_d_day_minus_1 > stoch_k_day_minus_1) and (stoch_k > stoch_d)):
                    return "SELL:already_matured_and_turning_point", current_price #if criteria is met recommend to sell
                else:
                    return "HOLD", current_price #if criteria is not met hold the stock
            else:
                if (current_price < stop_price):
                    return "SELL:stop_loss", current_price #if criteria is met recommend to sell
                else:
                    return "HOLD", current_price #if criteria is not met hold the stock
        else:
            return "HOLD", current_price #if criteria is not met hold the stock
    except:
        return "HOLD", 0 #if criteria is not met hold the stock
