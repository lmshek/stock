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
import math

def breakout(ticker, start_date = None, end_date = None, n = 10, take_profit_rate = 0.1, stop_loss_rate = 0.5):
    
    try:

        stock = yf.Ticker(ticker)
        if start_date:
            hist = stock.history(start = start_date.date(), end = end_date.date() + timedelta(days = 1), repair=True, raise_errors=True)
        else:
            hist = stock.history(period="max")
        
        # Define the length of the cup and handle
        cup_lengths = [30, 60, 120, 180, 240]
        handle_lengths = [5, 10, 20, 30, 40]      
        target_cup_depth = 0.0

        for i in range(len(cup_lengths)):        

            if len(hist['Close'].values) == 0:
                raise (f"No Stock data for {stock} between {start_date} and {end_date}")    

            cup_len = cup_lengths[i]
            handle_len = handle_lengths[i]

            cup_start_ind = hist.index[-1] - pd.DateOffset(days=cup_len + handle_len)
            cup_end_ind = hist.index[-1] - pd.DateOffset(days=handle_len + 1)
            handle_start_ind = hist.index[-1] - pd.DateOffset(days=handle_len)
            handle_end_ind = hist.index[-1]

            stage = 5

            # Calculate cup and handle conditions
            cup_left_high = hist['High'][cup_start_ind : cup_start_ind + pd.DateOffset(days=cup_len // stage)].mean()
            cup_left_1 = hist['Low'][cup_start_ind + pd.DateOffset(days=cup_len // stage) : cup_start_ind + pd.DateOffset(days= 2 * cup_len // stage)].mean()            
            cup_mid_low = hist['Low'][cup_start_ind + pd.DateOffset(days= 2 * cup_len // stage) : cup_start_ind + pd.DateOffset(days= 3 * cup_len // stage)].mean()
            cup_right_1 = hist['Low'][cup_start_ind + pd.DateOffset(days= 3 * cup_len // stage) : cup_start_ind + pd.DateOffset(days= 4 * cup_len // stage)].mean()            
            cup_right_high = hist['High'][cup_start_ind + pd.DateOffset(days= 4 * cup_len // stage) : cup_end_ind].mean()

            handle_left_high = hist['High'][handle_start_ind : handle_start_ind + pd.DateOffset(days=handle_len // stage)].mean()
            handle_left_1 = hist['Low'][handle_start_ind + pd.DateOffset(days=handle_len // stage) : handle_start_ind + pd.DateOffset(days= 2 * handle_len // stage)].mean()            
            handle_mid_low = hist['Low'][handle_start_ind + pd.DateOffset(days= 2 * handle_len // stage) : handle_start_ind + pd.DateOffset(days= 3 * handle_len // stage)].mean()
            handle_right_1 = hist['Low'][handle_start_ind + pd.DateOffset(days= 3 * handle_len // stage) : handle_start_ind + pd.DateOffset(days= 4 * handle_len // stage)].mean()            
            handle_right_high = hist['High'][handle_start_ind + pd.DateOffset(days= 4 *handle_len // stage) : handle_end_ind].mean()

            cup_depth = cup_left_high - cup_mid_low
            handle_depth = cup_right_high - handle_mid_low

            cup_formed = cup_left_high >= cup_left_1 and cup_left_1 >= cup_mid_low and cup_mid_low <= cup_right_1 and cup_right_1 <= cup_right_high \
                and is_similar(cup_left_high, cup_right_high, threshold=0.05)
            handle_formed = handle_depth > 0 and handle_depth <= cup_depth * 0.40 \
                and handle_left_high >= handle_left_1 and handle_left_1 >= handle_mid_low and handle_mid_low <= handle_right_1 and handle_right_1 <= handle_right_high \
                and is_similar(handle_left_high, handle_right_high, threshold= 0.05)
            going_to_breakout = cup_left_high * 0.90 < handle_right_high and handle_right_high < cup_left_high * 1.05

            vol_increase = hist['Volume'][-2] <= hist['Volume'][-1]
            price_condition = hist['Close'][-1] > 0.5

            potential_cup_and_handle = cup_formed & handle_formed & going_to_breakout & vol_increase & price_condition
            hist['Pattern'] = potential_cup_and_handle.astype(int)

            multiplier = 1

            if hist['Pattern'][-1] and target_cup_depth < cup_depth / hist['Close'][-1] and handle_depth / hist['Close'][-1] >= 0.04:
                hist['cup_len'] = cup_len
                hist['handle_len'] = handle_len
                hist['cup_depth'] = cup_depth / hist['Close']
                hist['handle_depth'] = handle_depth / hist['Close']                

                target_cup_depth = cup_depth / hist['Close'][-1]

        if target_cup_depth > 0.0:
            return 1, hist['Close'].values[-1], hist.tail(1), multiplier
        else:
            return False, 0, pd.DataFrame({}), 1
            
    except:
        return False, 0, pd.DataFrame({}), 1

def take_first(elem):
    return elem[1][1]['cup_depth'][0] / elem[1][1]['handle_depth'][0] # Risk Reward Ratio

def breakout_order(stocks):
    return OrderedDict(sorted(stocks, key = take_first, reverse = True))

def breakout_print(today_data):
    print(f'{bcolors.OKCYAN}Cup Depth: {today_data["cup_depth"][-1]}, Handle Depth: {today_data["handle_depth"][-1]}{bcolors.ENDC}')
    print(f'{bcolors.OKCYAN}Cup Length: {today_data["cup_len"][-1]}, Handle Length: {today_data["handle_len"][-1]}{bcolors.ENDC}')

def breakout_buy(simulator, stock, buy_price, buy_date, no_of_splits, cup_len, handle_len, cup_depth, handle_depth):
        """
        function takes buy price and the number of shares and buy the stock
        """

        #calculate the procedure
        n_shares = simulator.buy_percentage(buy_price, 1/no_of_splits)
        simulator.capital = simulator.capital - buy_price * n_shares
        simulator.buy_orders[stock] = [buy_price, n_shares, buy_price * n_shares, buy_date, cup_len, handle_len, cup_depth, handle_depth]


        print(f'{bcolors.OKCYAN}Bought {stock} for {buy_price} with risk reward ratio {cup_depth / handle_depth} on the {buy_date.strftime("%Y-%m-%d")} . Account Balance: {simulator.capital}{bcolors.ENDC}')

def breakout_sell(ticker, buy_date, buy_price, todays_date, cup_len, handle_len, cup_depth, handle_depth):
    try :
        start_date = todays_date - timedelta(days = 100)
        end_date = todays_date

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.date(), end=end_date.date() + timedelta(days=1))
        
        current_price = hist['Close'].values[-1]
        sell_price = buy_price + buy_price * cup_depth
        stop_price = buy_price - buy_price * handle_depth
        sell_date = stock_utils.get_market_real_date(buy_date, handle_len) # selling date        
        time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

        if (current_price is not None):
            if (current_price < stop_price):
                return "SELL:stop_loss", current_price #if criteria is met recommend to sell
            elif (current_price >= sell_price):
                return "SELL:take_profit", current_price #if criteria is met recommend to sell
            #elif (current_price >= buy_price * 1.1 and stock_utils.get_market_days(buy_date, todays_date) <= 3):
            #    return "SELL:take_profit_10pc_within_3_days", current_price #if criteria is met recommend to sell
            #elif (current_price <= buy_price * 1.03 and stock_utils.get_market_days(buy_date, todays_date) >= 3):
            #    return "SELL:did_not_breakout_within_3_days", current_price #if criteria is met recommend to sell
            elif (todays_date >= sell_date ):
                return "SELL:already_matured", current_price #if criteria is met recommend to sell
            else:
                return "HOLD", current_price #if criteria is not met hold the stock
        else:
            return "HOLD", current_price #if criteria is not met hold the stock
    except:
        return "HOLD", 0 #if criteria is not met hold the stock

def is_similar(num1, num2, threshold=0.05):
    # Calculate the absolute difference between the two numbers
    abs_diff = abs(num1 - num2)
    
    # Calculate the average of the two numbers
    avg = (num1 + num2) / 2
    
    # Calculate the percentage difference
    perc_diff = abs_diff / avg
    
    # Compare the percentage difference with the threshold
    return perc_diff <= threshold

def buy_percentage(self, buy_price, buy_perc = 1):
    """
    this function determines how much capital to spend on the stock and returns the number of shares
    """
    stock_expenditure = self.capital * buy_perc
    n_shares = math.floor(stock_expenditure / buy_price)
    return n_shares