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

def trend(all_time_stock_data, start_date = None, end_date = None):
    
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

            # we skip if the right hand side is local_maxima
            if all_max_min.iloc[-1].Type == 'local_maxima':
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

            # if the last local_minima is not within end_date - 3 days and the last item must be local_mimima, otherwise continue
            if not(end_date.replace(tzinfo=timezone('Asia/Hong_Kong')) - grouped_max_min.index[-1]  <= timedelta(days=3)):
                continue

            # Identify potential Cup and Handle patterns
            if len(grouped_max_min) >= 5: # At least 5 min max in grouped_max_min       
                
                ts_wave_3_max = grouped_max_min.iloc[-5].name
                ts_wave_2_min = grouped_max_min.iloc[-4].name
                ts_wave_2_max = grouped_max_min.iloc[-3].name
                ts_wave_1_min = grouped_max_min.iloc[-3].name
                ts_wave_1_max = grouped_max_min.iloc[-2].name
                ts_wave_0_min = grouped_max_min.iloc[-1].name

                wave_3_max = hist.loc[ts_wave_3_max]['High']
                wave_2_min = hist.loc[ts_wave_2_min]['Low']
                wave_2_max = hist.loc[ts_wave_2_max]['High']
                wave_1_min = hist.loc[ts_wave_1_min]['Low']
                wave_1_max = hist.loc[ts_wave_1_max]['High']
                wave_0_min = hist.loc[ts_wave_0_min]['Low']

                slope_max_wave_3 = (wave_3_max - wave_2_max) / (ts_wave_3_max - ts_wave_2_max).days
                slope_min_wave_2 = (wave_2_min - wave_1_min) / (ts_wave_2_min - ts_wave_1_min).days
                slope_max_wave_2 = (wave_2_max - wave_1_max) / (ts_wave_2_max - ts_wave_1_max).days
                slope_min_wave_1 = (wave_1_min - wave_0_min) / (ts_wave_1_min - ts_wave_0_min).days
                
                trend_formed = is_similar(slope_max_wave_3, slope_max_wave_2, 0.03) and is_similar(slope_min_wave_2, slope_min_wave_1, 0.03)
                
                price_increase = wave_0_min <= hist['Close'][-1]     

                if trend_formed and price_increase: 
                    hist['slope'] = slope_max_wave_2
                    hist['wave_1_max'] = wave_1_max
                    hist['wave_depth'] = wave_1_max - wave_1_min
                    hist['wave_length'] = (ts_wave_1_max - ts_wave_1_min).days
 
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
    return elem[1][1]['slope'][0] # slope

def trend_order(stocks):
    return OrderedDict(sorted(stocks, key = take_first, reverse = True))

def trend_print(today_data):
    print(f'{bcolors.OKCYAN}Slope: {today_data["slope"][-1]} Wave Depth: {today_data["wave_depth"][-1]} Wave Length: {today_data["wave_length"][-1]}{bcolors.ENDC}')

def trend_buy(simulator, stock, buy_price, buy_date, no_of_splits, wave_1_max, slope, wave_depth, wave_length):
        """
        function takes buy price and the number of shares and buy the stock
        """

        #calculate the procedure
        n_shares = simulator.buy_percentage(buy_price, 1/no_of_splits)
        simulator.capital = simulator.capital - buy_price * n_shares
        simulator.buy_orders[stock] = [buy_price, n_shares, buy_price * n_shares, buy_date, wave_1_max, slope, wave_depth, wave_length]


        print(f'{bcolors.OKCYAN}Bought {stock} for {buy_price} with slope {slope} on the {buy_date.strftime("%Y-%m-%d")} . Account Balance: {simulator.capital}{bcolors.ENDC}')

def trend_sell(stock_data, market, ticker, buy_date, buy_price, todays_date, wave_1_max, slope, wave_depth, wave_length):
    try :
        hist = stock_data[ticker]        
        
        current_price = hist['Close'][todays_date.date():todays_date.date()].values[-1]
        days_on_market = (todays_date.date() - buy_date.date()).days
        sell_price = slope * days_on_market + wave_1_max
        stop_price = buy_price - wave_depth
        sell_date = stock_utils.get_market_real_date(market, buy_date, wave_length)

        time.sleep(1) #to make sure the requested transactions per seconds is not exceeded

        if (current_price is not None):
            if (current_price < stop_price):
                return "SELL:stop_loss", current_price #if criteria is met recommend to sell
            elif (current_price >= sell_price):
                return "SELL:take_profit", current_price #if criteria is met recommend to sell  
            elif (todays_date >= sell_date):
                return "SELL:maturity_date", current_price #if criteria is met recommend to sell            
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