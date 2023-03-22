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

    def scan_selling_signal(self):
        current_dir = os.getcwd()
        inventory_file = os.path.join(current_dir, 'inventory/hsi.csv')
        inventories = pd.read_csv(inventory_file, sep=r'\s*,\s*', lineterminator='\n')
        inventories = inventories[pd.isnull(inventories['sold_date'])]
        t = telegram()

        for index, inventory in inventories.iterrows():
            hold_till = (datetime.strptime(inventory['hold_till'], "%Y-%m-%d") - datetime.strptime(inventory['buy_date'], "%Y-%m-%d")).days
            recommended_action, current_price = stoch_sell(inventory['ticker'], datetime.strptime(inventory['buy_date'], "%Y-%m-%d"), inventory['buy_price'], self.day, \
                self.sell_perc, hold_till, self.stop_perc)
            if recommended_action == "SELL":

                message = "<u><b>SELL SIGNAL</b></u>\n" \
                    + f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                    + f"Stock: <a href=\"http://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&lang=1&titlestyle=1&vol=1&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=7&ref2para1=14&ref2para2=3&ref2para3=0&subChart3=12&ref3para1=0&ref3para2=0&ref3para3=0&subChart4=3&ref4para1=12&ref4para2=26&ref4para3=9&scheme=3&com=100&chartwidth=870&chartheight=945&stockid=00{inventory['ticker']}&period=9&type=1&logoStyle=1&\">{inventory['ticker']}</a> \n" \
                    + f"Current Price: ${round(current_price,2)} \n" \
                    + f"Bought Price: ${round(inventory['buy_price'],2)} \n" \
                    + f"Bought Date: {inventory['buy_date']} \n" \
                    + f"G/L: ${round(current_price - inventory['buy_price'], 2)} ({round((current_price - inventory['buy_price']) / inventory['buy_price'] * 100, 2)}%) \n" \
                    
                t.send_message(message)
                
        

    def scan_buying_signal(self):
    
        model_recommended_stocks = {}
        for stock in self.stocks:
            try:
                buy_signal, close_price, today_stock_data, multiplier = self.get_stock_data(stock, back_to=300)
                #if prediction greater than
                if buy_signal: 
                    model_recommended_stocks[stock] = (close_price, today_stock_data, multiplier)
            except Exception as err:
                print(err)
                pass   
        def take_first(elem):
            return elem[1][1]['RSI_14'][0]
        
        good_stocks = OrderedDict(sorted(model_recommended_stocks.items(), key = take_first, reverse = False))
  
        
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

                
                today = date.today()
                hk_holidays = holidays.HK()
                public_holiday_days = 0
                for i in range(1, hold_till):
                    if (today + timedelta(days=i)) in hk_holidays or hk_holidays._is_weekend((today + timedelta(days=i))):
                        public_holiday_days+=1

                message = "<u><b>BUY SIGNAL</b></u>\n" \
                    + f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                    + f"Stock: <a href=\"http://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&lang=1&titlestyle=1&vol=1&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=7&ref2para1=14&ref2para2=3&ref2para3=0&subChart3=12&ref3para1=0&ref3para2=0&ref3para3=0&subChart4=3&ref4para1=12&ref4para2=26&ref4para3=9&scheme=3&com=100&chartwidth=870&chartheight=945&stockid=00{stock}&period=9&type=1&logoStyle=1&\">{stock}</a> \n" \
                    + f"Current Price: ${round(current_price,2)} \n" \
                    + f"Take Profit at: ${round(current_price * (1 + sell_perc), 2)} (+{round(sell_perc * 100, 2)}%) \n" \
                    + f"Stop at: ${round(current_price * (1 - stop_perc), 2)} (-{round(stop_perc * 100, 2)}%) \n" \
                    + f"Hold till: {(today + timedelta(hold_till) + timedelta(public_holiday_days)).strftime('%Y-%m-%d')} ({hold_till} days)\n" 

                t.send_message(message)


    def get_stock_data(self, stock, back_to = 100):
        """
        this function queries to yf and get data of a particular stock on a given day back to certain amount of days
        (default is 30)
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)
        buy_signal, close_price, today_stock_data, multiplier = self.model(stock, start_date=start, end_date=end)
        return buy_signal, close_price, today_stock_data, multiplier


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

    
    sf = stockfinder_technical(stocks, stoch_k_d, 'v1', sell_perc = 0.14, hold_till= 7, stop_perc = 0.07, no_of_recommendations = 5)
    sf.scan_selling_signal()
    sf.scan_buying_signal()    


