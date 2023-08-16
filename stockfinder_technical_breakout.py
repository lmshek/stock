import pandas_datareader as web
from datetime import date, datetime, time
import holidays
import numpy as np
import pandas as pd
from technical.breakoutV2 import breakout, breakout_sell, breakout_order
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
from telegram.telegram import telegram
from datetime import datetime, timedelta, date
from stock_utils import stock_utils
import yfinance as yf
import argparse
import urllib.parse


class stockfinder_technical_breakout:

    def __init__(self, market, stocks_list, model, model_version, no_of_recommendations = 5):
        self.market = market
        self.stocks = stocks_list
        self.model = model
        self.model_version = model_version        
        self.no_of_recommendations = no_of_recommendations
        self.day = datetime.today()
        self.stock_data = {}
        self.days_before_start_date = 300

        # Get Stock Data             
        for ticker in self.stocks:
            try:
                stock = yf.Ticker(ticker)
                self.stock_data[ticker] = stock.history(start = stock_utils.get_market_real_date(self.market, self.day, -self.days_before_start_date), end = self.day + timedelta(days = 1), repair="silent", raise_errors=True, rounding=True)
                #self.stock_data[ticker] = stock.history(period="max", repair="silent", raise_errors=True, rounding=True, keepna=True)
            except Exception as e:
                continue

    def scan_selling_signal(self):
        current_dir = os.getcwd()
        inventory_file = os.path.join(current_dir, f'inventory/{self.market}_breakout.csv')
        inventories = pd.read_csv(inventory_file, sep=r'\s*,\s*', lineterminator='\n')
        inventories = inventories[pd.isnull(inventories['sold_date'])]
        t = telegram()

        for index, inventory in inventories.iterrows():            
            recommended_action, current_price = breakout_sell(self.stock_data[inventory['ticker']], self.market, inventory['ticker'], datetime.strptime(inventory['buy_date'], "%Y-%m-%d"), inventory['buy_price'], self.day, inventory['cup_len'], inventory['handle_len'], inventory['cup_depth'], inventory['handle_depth'])
            if "SELL" in recommended_action:
                currency = ''
                link = ''
                stock = inventory['ticker']
                if market == 'HK' and ".HK" in stock:
                    currency = 'HKD'
                    link = f"http://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&lang=1&titlestyle=1&vol=1&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=7&ref2para1=14&ref2para2=3&ref2para3=0&subChart3=12&ref3para1=0&ref3para2=0&ref3para3=0&subChart4=3&ref4para1=12&ref4para2=26&ref4para3=9&scheme=3&com=100&chartwidth=870&chartheight=945&stockid=00{stock}&period=9&type=1&logoStyle=1&"
                elif market == 'HK' and not ".HK" in stock:
                    currency = 'USD'
                    forex_ticker = urllib.parse.quote_plus(stock)
                    link = f"https://finance.yahoo.com/quote/{forex_ticker}/chart?p={forex_ticker}"
                elif market == 'US':
                    currency = 'USD'
                    link = f"https://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&titlestyle=1&lang=1&vol=1&stockid={stock}.US&period=6&type=1&com=70005&scheme=3&chartwidth=870&chartheight=855&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=3&ref2para1=12&ref2para2=26&ref2para3=9&subChart3=12"
                elif market == 'JP':
                    currency = 'JPY'
                    jp_stock = stock.replace('.T', '')
                    link = f"https://www.tradingview.com/chart/dPvcvEPT/?symbol=TSE%3A{jp_stock}"
                elif market == 'SG':
                    currency = 'SGD'
                    sg_stock = stock.replace('.SI', '')
                    link = f"https://www.tradingview.com/chart/dPvcvEPT/?symbol=SGX%3A{sg_stock}"

                message = f"<u><b>SELL {self.market} STOCK</b></u>\n" \
                    + f"Recommendation: {recommended_action}" \
                    + f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                    + f"Stock: <a href=\"{link}\">{inventory['ticker']}</a> \n" \
                    + f"Current Price: {currency} {round(current_price,2)} \n" \
                    + f"Bought Price: {currency} {round(inventory['buy_price'],2)} \n" \
                    + f"Bought Date: {inventory['buy_date']} \n" \
                    + f"Cup Length: {inventory['cup_len']} \n" \
                    + f"Handle Length: {inventory['handle_len']} \n" \
                    + f"Cup Depth: {inventory['cup_depth']} \n" \
                    + f"Handle Depth: {inventory['handle_depth']} \n" \
                    + f"Risk Reward Ratio: {round(inventory['cup_depth'] / inventory['handle_depth'] * 100, 2)} \n" \
                    + f"G/L: ${round(current_price - inventory['buy_price'], 2)} ({round((current_price - inventory['buy_price']) / inventory['buy_price'] * 100, 2)}%) \n" \
                    
                t.send_message(message)
                
        

    def scan_buying_signal(self):
    
        model_recommended_stocks = {}
        for stock in self.stocks:
            try:
                buy_signal, close_price, today_stock_data, multiplier = self.get_stock_data(stock)
                #if prediction greater than
                if buy_signal: 
                    model_recommended_stocks[stock] = (close_price, today_stock_data, multiplier)
            except Exception as err:
                print(err)
                pass   
        
        
        good_stocks = breakout_order(model_recommended_stocks.items())
  
        
        # Push the buying signal to Telegram channel
        # Get "no_of_recommendations" most good probabilities stocks
        t = telegram()
        if len(good_stocks) == 0:
            print(f'No recommendation at {datetime.now().strftime("%H:%M:%S")} by {self.model.__name__}_{self.model_version} in market {self.market}')
            #t.send_message(f'No recommendation at {datetime.now().strftime("%H:%M:%S")} by {self.model.__name__}_{self.model_version} in market {self.market}')
        else:    
            for key in list(good_stocks)[0:self.no_of_recommendations]:
                stock = key
                current_price = good_stocks[key][0]
                data = good_stocks[key][1]
                cup_len = data['cup_len'][-1]
                handle_len = data['handle_len'][-1]
                cup_depth = data['cup_depth'][-1]
                handle_depth = data['handle_depth'][-1]

                
                today = date.today()      
                hold_till = stock_utils.get_market_real_date(self.market, today, handle_len)


                currency = ''
                link = ''
                if market == 'HK' and ".HK" in stock:
                    currency = 'HKD'
                    link = f"http://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&lang=1&titlestyle=1&vol=1&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=7&ref2para1=14&ref2para2=3&ref2para3=0&subChart3=12&ref3para1=0&ref3para2=0&ref3para3=0&subChart4=3&ref4para1=12&ref4para2=26&ref4para3=9&scheme=3&com=100&chartwidth=870&chartheight=945&stockid=00{stock}&period=9&type=1&logoStyle=1&"
                elif market == 'HK' and not ".HK" in stock:
                    currency = 'USD'
                    forex_ticker = urllib.parse.quote_plus(stock)
                    link = f"https://finance.yahoo.com/quote/{forex_ticker}/chart?p={forex_ticker}"
                elif market == 'US':
                    currency = 'USD'
                    link = f"https://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&titlestyle=1&lang=1&vol=1&stockid={stock}.US&period=6&type=1&com=70005&scheme=3&chartwidth=870&chartheight=855&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=3&ref2para1=12&ref2para2=26&ref2para3=9&subChart3=12"
                elif market == 'JP':
                    currency = 'JPY'
                    jp_stock = stock.replace('.T', '')
                    link = f"https://www.tradingview.com/chart/?symbol=TSE%3A{jp_stock}"
                elif market == 'SG':
                    currency = 'SGD'
                    sg_stock = stock.replace('.SI', '')
                    link = f"https://www.tradingview.com/chart/?symbol=SGX%3A{sg_stock}"

                message = f"<u><b>BUY {self.market} STOCK</b></u>\n" \
                    + f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                    + f"Stock: <a href=\"{link}\">{stock}</a> \n" \
                    + f"Current Price: {currency} {round(current_price,2)} \n" \
                    + f"Take Profit at: {currency} {round(current_price * (1 + cup_depth), 2)} (+{round(cup_depth * 100, 2)}%) \n" \
                    + f"Stop at: {currency} {round(current_price * (1 - handle_depth), 2)} (-{round(handle_depth * 100, 2)}%) \n" \
                    + f"Cup Length: {cup_len}\n" \
                    + f"Handle Length: {handle_len}\n" \
                    + f"Cup Depth: {round(cup_depth, 5)}\n" \
                    + f"Handle Depth: {round(handle_depth, 5)}\n" \
                    + f"Risk Reward Ratio: {round(cup_depth / handle_depth * 100, 2)} \n" \
                    + f"Hold till: {(hold_till).strftime('%Y-%m-%d')} ({handle_len} days)\n" 

                t.send_message(message)


    def get_stock_data(self, stock):
        """
        this function queries to yf and get data of a particular stock on a given day back to certain amount of days
        (default is 30)
        """
        #get start and end dates
        end = self.day
        start = stock_utils.get_market_real_date(self.market, end, -self.days_before_start_date)
        buy_signal, close_price, today_stock_data, multiplier = self.model(self.stock_data[stock], start_date=start, end_date=end)
        return buy_signal, close_price, today_stock_data, multiplier


if __name__ == "__main__":

    def is_time_between(begin_time, end_time, check_time=None):
        # If check time is not given, default to current Now time
        check_time = check_time or datetime.now().time()
        if begin_time < end_time:
            return check_time >= begin_time and check_time <= end_time
        else: # crosses midnight
            return check_time >= begin_time or check_time <= end_time
        
    # Get the arguments from command line

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add an argument to accept a list of strings
    parser.add_argument('--market', type=str, help='Country of the Market (e.g. HK, US)')
    parser.add_argument('--stock_list', nargs='+', help='List of the stocks (e.g. hsi_main, dow_jones, nasdaq_100)')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the arguments
    market = (args.market).upper()
    stock_list = args.stock_list

    #Check if today is holiday
    market_holidays = getattr(holidays, market)()    
    today = date.today()
    if(today in market_holidays):
        exit()

    # Check if now is in Market Hours
    """
    if market == "HK" and not is_time_between(time(9,30), time(16,00)):
        exit()
    if market == "JP" and not is_time_between(time(8,00), time(14,00)):
        exit()
    if market == "SG" and not is_time_between(time(9,00), time(17,00)):
        exit()
    if market == "US" and not is_time_between(time(21,30), time(23,59)):
        exit()
    """
         
    current_dir = os.getcwd()    
    #hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
    #hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()
    #stocks = list(np.unique(hsi_tech + hsi_main))       
    #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()

    stocks = []
    for stock_cat in stock_list: #'hsi_integrated_large', 'hsi_integrated_medium',
        stocks = stocks + pd.read_csv(os.path.join(current_dir, f'stock_list/{stock_cat}.csv'))['tickers'].tolist()
    stocks = list(np.unique(stocks)) 

    
    sf = stockfinder_technical_breakout(market, stocks, breakout, 'v2', no_of_recommendations = 5)
    sf.scan_selling_signal()
    sf.scan_buying_signal()    


