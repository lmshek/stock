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
from pytz import timezone
import traceback

def breakout(all_time_stock_data, start_date = None, end_date = None):
    
    try:

        hist = all_time_stock_data.loc[start_date.date():end_date.date()]       

        # Define the length of the cup and handle        
        target_cup_depth = 0.0

        for pattern_length in list(range(90, 9, -1)): 
        
            # Find local minima
            local_minima_indices = argrelextrema(hist['Low'].values, np.less, order=pattern_length)[0]
            local_minima = hist.iloc[local_minima_indices]
            local_minima['Type'] = 'local_minima'
            
            # Find local maxima
            local_maxima_indices = argrelextrema(hist['High'].values, np.greater, order=pattern_length)[0]
            local_maxima = hist.iloc[local_maxima_indices]
            local_maxima['Type'] = 'local_maxima'

            # Merge local_minima and local_maxima into all_max_min
            all_max_min = pd.concat([local_minima, local_maxima]).sort_index()

            # we skip if the right hand side is local_minima
            if all_max_min.iloc[-1].Type == 'local_minima':
                continue

            grouped_max_min = pd.DataFrame({})            
            for i in range(len(all_max_min)):
                if grouped_max_min.empty:
                    grouped_max_min = grouped_max_min.append(all_max_min.iloc[i])
                    grouped_max_min_counter = 0
                else:
                    try:
                        if grouped_max_min.iloc[grouped_max_min_counter].name == all_max_min.iloc[i].name:
                            continue # ignore duplicate dates

                        if(grouped_max_min.iloc[grouped_max_min_counter].Type == all_max_min.iloc[i].Type):
                            # Same Type
                            if grouped_max_min.iloc[grouped_max_min_counter].Type == 'local_maxima':
                                # compare local_maxima
                                if all_max_min.iloc[i].High >= grouped_max_min.iloc[grouped_max_min_counter].High:
                                    grouped_max_min = grouped_max_min.drop(grouped_max_min.index[-1])
                                    grouped_max_min = grouped_max_min.append(all_max_min.iloc[i].copy())

                            else:
                                # compare local_minima
                                if all_max_min.iloc[i].Low <= grouped_max_min.iloc[grouped_max_min_counter].Low:
                                    grouped_max_min = grouped_max_min.drop(grouped_max_min.index[-1])
                                    grouped_max_min = grouped_max_min.append(all_max_min.iloc[i].copy())

                        else:
                            grouped_max_min = grouped_max_min.append(all_max_min.iloc[i].copy())
                            grouped_max_min_counter += 1 
                    except Exception as e:
                        # Print the exception message
                        print("An exception occurred:", str(e))

                        # Print the stack trace
                        traceback.print_exc()
                        raise e

            # if the last local_maxima is not within end_date - 5 days and the last item must be local_maxima, otherwise continue
            if not(end_date.replace(tzinfo=timezone('Asia/Hong_Kong')) - grouped_max_min.index[-1]  <= timedelta(days=5)):
                continue

            # Identify potential Cup and Handle patterns
            if len(grouped_max_min) >= 5: # At least 5 min max in grouped_max_min       

                ts_cup_left = grouped_max_min.iloc[-5].name
                ts_cup_mid = grouped_max_min.iloc[-4].name
                ts_cup_right = grouped_max_min.iloc[-3].name
                ts_handle_left = grouped_max_min.iloc[-3].name
                ts_handle_mid = grouped_max_min.iloc[-2].name
                ts_handle_right = grouped_max_min.iloc[-1].name

                cup_left_high = hist.loc[ts_cup_left]['High']
                cup_mid_low = hist.loc[ts_cup_mid]['Low']
                cup_right_high = hist.loc[ts_cup_right]['High']
                handle_left_high = hist.loc[ts_handle_left]['High']
                handle_mid_low = hist.loc[ts_handle_mid]['Low']
                handle_right_high = hist.loc[ts_handle_right]['High']

                cup_depth = cup_left_high - cup_mid_low
                handle_depth = handle_left_high - handle_mid_low

                cup_formed = cup_left_high > cup_mid_low and cup_mid_low < cup_right_high \
                    and is_similar(cup_left_high, cup_right_high, threshold=0.05)
                handle_formed = handle_depth > 0 and handle_depth <= cup_depth * 0.40 and (ts_handle_right - ts_handle_left).days < (ts_cup_right - ts_cup_left).days\
                    and handle_left_high > handle_mid_low and handle_mid_low < handle_right_high \
                    and is_similar(handle_left_high, handle_right_high, threshold= 0.05) \
                    and cup_depth / handle_depth <= 5.0
                going_to_breakout = is_similar(cup_left_high, handle_right_high, threshold=0.05) and is_similar(handle_right_high, hist['Close'][-1], threshold=0.05)


                vol_increase = hist['Volume'][-2] <= hist['Volume'][-1]
                #price_condition = hist['Close'][-1] > 0.5 and is_similar(hist['Close'][-1], handle_right_high, threshold=0.05)

                potential_cup_and_handle = cup_formed & handle_formed & going_to_breakout & vol_increase           

                if potential_cup_and_handle and target_cup_depth < cup_depth / hist['Close'][-1]: #and handle_depth / hist['Close'][-1] >= 0.04:
                    hist['cup_len'] = (ts_cup_right - ts_cup_left).days
                    hist['handle_len'] = (ts_handle_right - ts_handle_left).days
                    hist['cup_depth'] = cup_depth / hist['Close']
                    hist['handle_depth'] = handle_depth / hist['Close']                

                    #target_cup_depth = cup_depth / hist['Close'][-1]
                    return 1, hist['Close'].values[-1], hist.tail(1), 1
        
        
        multiplier = 1
        if target_cup_depth > 0.0:
            return 1, hist['Close'].values[-1], hist.tail(1), multiplier
        else:
            return False, 0, pd.DataFrame({}), 1
            
    except Exception as e:        
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

def breakout_sell(stock_data, market, ticker, buy_date, buy_price, todays_date, cup_len, handle_len, cup_depth, handle_depth):
    try :
        hist = stock_data[ticker]        
        
        current_price = hist['Close'][todays_date.date():todays_date.date()].values[-1]
        sell_price = buy_price + buy_price * cup_depth
        stop_price = buy_price - buy_price * handle_depth
        sell_date = stock_utils.get_market_real_date(market, buy_date, handle_len) # selling date        
        time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

        if (current_price is not None):
            if (current_price < stop_price):
                return "SELL:stop_loss", current_price #if criteria is met recommend to sell
            elif (current_price >=  buy_price * (1 + cup_depth * 2 / 3)):
                return "SELL:take_profit", current_price #if criteria is met recommend to sell
            elif (current_price >= buy_price * (1 + cup_depth / 2) and stock_utils.get_market_days(buy_date, todays_date) <= handle_len / 3):
                return "SELL:take_profit_in_early_phase", current_price #if criteria is met recommend to sell
            elif (current_price <= buy_price * (1 + cup_depth / 4) and stock_utils.get_market_days(buy_date, todays_date) >= handle_len / 2):
                return "SELL:did_not_breakout_within_half_handle", current_price #if criteria is met recommend to sell
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