import pandas_datareader as web
from datetime import date, datetime, time
import holidays
import numpy as np
import pandas as pd
from technical.stoch import stoch_k_d, stoch_sell
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
from telegram.telegram import telegram
from datetime import datetime, timedelta, date



class stockfinder_technical:

    def __init__(self, stocks_list, model, model_version, sell_perc = 0.21, hold_till = 14, stop_perc = 0.07, no_of_recommendations = 3):
        self.stocks = stocks_list
        self.model = model
        self.model_version = model_version
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc
        self.no_of_recommendations = no_of_recommendations
        self.day = datetime.today()

    def scan(self):
    
        model_recommended_stocks = {}
        for stock in self.stocks:
            try:
                buy_signal, close_price, today_stock_data = self.get_stock_data(stock, back_to=100)
                #if prediction greater than
                if buy_signal: 
                    model_recommended_stocks[stock] = (close_price, today_stock_data)
            except Exception as err:
                print(err)
                pass   
        def take_first(elem):
            return elem[1][1]['Volume'][0]
        
        good_stocks = OrderedDict(sorted(model_recommended_stocks.items(), key = take_first, reverse = True))
  
        
        # Push the buying signal to Telegram channel
        # Get "no_of_recommendations" most good probabilities stocks
        t = telegram()
        if len(good_stocks) == 0:
            print(f'No recommendation at {datetime.now().strftime("%H:%M:%S")} by {self.model.__name__}_{self.model_version}')
            t.send_message(f'No recommendation at {datetime.now().strftime("%H:%M:%S")} by {self.model.__name__}_{self.model_version}')
        else:    
            for key in list(good_stocks)[0:self.no_of_recommendations]:
                stock = key
                current_price = good_stocks[key][0]
                sell_perc = self.sell_perc
                hold_till = self.hold_till
                stop_perc = self.stop_perc

                
                t.send_formatted_message(model_name=f"{self.model.__name__}_{self.model_version}" , stock=stock, current_price=current_price, sell_perc=sell_perc, hold_till=hold_till, stop_perc=stop_perc)


    def get_stock_data(self, stock, back_to = 100):
        """
        this function queries to yf and get data of a particular stock on a given day back to certain amount of days
        (default is 30)
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)
        buy_signal, close_price, today_stock_data = self.model(stock, start_date=start, end_date=end)
        return buy_signal, close_price, today_stock_data


if __name__ == "__main__":

    def is_time_between(begin_time, end_time, check_time=None):
        # If check time is not given, default to current Now time
        check_time = check_time or datetime.now().time()
        if begin_time < end_time:
            return check_time >= begin_time and check_time <= end_time
        else: # crosses midnight
            return check_time >= begin_time or check_time <= end_time

    #Check if today is holiday
    hk_holidays = holidays.HK()
    today = date.today()
    if(today in hk_holidays):
        exit()

    # Check if now is between 09:45 and 16:00 (market time)
    if not is_time_between(time(9,30), time(16,00)):
        exit()

    current_dir = os.getcwd()    
    #hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
    #hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()
    #stocks = list(np.unique(hsi_tech + hsi_main))       
    #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()

    stocks = []
    for stock_cat in ['hsi_integrated_large', 'hsi_integrated_medium']: #'hsi_integrated_large', 'hsi_integrated_medium',
        stocks = stocks + pd.read_csv(os.path.join(current_dir, f'stock_list/hsi/{stock_cat}.csv'))['tickers'].tolist()
    stocks = list(np.unique(stocks)) 

    
    stockfinder_technical(stocks, stoch_k_d, 'v1', sell_perc = 0.21, hold_till= 14, stop_perc = 0.07, no_of_recommendations = 3).scan()    


