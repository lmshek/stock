import pandas_datareader as web
from datetime import date, datetime, time
import holidays
import numpy as np
import pandas as pd
from technical.stoch import stoch_k_d, stoch_sell, stoch_order
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
import math


class stockfinder_technical_breakout:

    def __init__(self, market, stocks_list, model, model_version, profit_perc, stop_perc, hold_till, no_of_recommendations = 5, no_of_splits = 5):
        self.market = market
        self.stocks = stocks_list
        self.model = model
        self.model_version = model_version        
        self.no_of_recommendations = no_of_recommendations
        self.day = datetime.today()
        self.stock_data = {}
        self.earning_dates = {}
        self.benchmark_data = {}
        self.days_before_start_date = 300
        self.profit_perc = profit_perc
        self.stop_perc = stop_perc
        self.hold_till = hold_till
        self.no_of_splits = no_of_splits

        # Get Stock Data             
        for ticker in self.stocks:
            try:
                stock = yf.Ticker(ticker)
                self.stock_data[ticker] = stock.history(start = stock_utils.get_market_real_date(self.market, self.day, -self.days_before_start_date), end = self.day + timedelta(days = 1), repair=True)
                #self.earning_dates[ticker] = stock.get_earnings_dates(limit=24)
                #self.stock_data[ticker] = stock.history(period="max", repair=True, keepna=True)

                ## Remove TimeZone
                self.stock_data[ticker] = self.stock_data[ticker].tz_localize(None)
            except Exception as e:
                continue

        # Get Benchmark Index
        self.get_benchmark_data()

    def scan_selling_signal(self):
        current_dir = os.getcwd()
        inventory_file = os.path.join(current_dir, f'inventory/{self.market}_stoch.xlsx')
        inventories = pd.read_excel(inventory_file)
        inventories = inventories[pd.isnull(inventories['Sold Date'])]
        t = telegram()

        for index, inventory in inventories.iterrows():            
            recommended_action, current_price = stoch_sell(self.stock_data, self.market, inventory['Ticker'], inventory['Buy Date'], inventory['Buy Price'], self.day, self.profit_perc, self.hold_till, self.stop_perc)
            if "SELL" in recommended_action:
                currency = ''
                link = ''
                stock = inventory['Ticker']
                if market == 'HK' and ".HK" in stock:
                    currency = '$'
                    link = f"http://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&lang=1&titlestyle=1&vol=1&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=7&ref2para1=14&ref2para2=3&ref2para3=0&subChart3=12&ref3para1=0&ref3para2=0&ref3para3=0&subChart4=3&ref4para1=12&ref4para2=26&ref4para3=9&scheme=3&com=100&chartwidth=870&chartheight=945&stockid=00{stock}&period=9&type=1&logoStyle=1&"
                elif market == 'HK' and not ".HK" in stock:
                    currency = '$'
                    forex_ticker = urllib.parse.quote_plus(stock)
                    link = f"https://finance.yahoo.com/quote/{forex_ticker}/chart?p={forex_ticker}"
                elif market == 'US':
                    currency = '$'
                    link = f"https://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&titlestyle=1&lang=1&vol=1&stockid={stock}.US&period=6&type=1&com=70005&scheme=3&chartwidth=870&chartheight=855&Indicator=1&indpara1=10&indpara2=20&indpara3=50&indpara4=100&indpara5=150&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=3&ref2para1=12&ref2para2=26&ref2para3=9&subChart3=12"
                elif market == 'JP':
                    currency = '¥'
                    jp_stock = stock.replace('.T', '')
                    link = f"https://www.tradingview.com/chart/dPvcvEPT/?symbol=TSE%3A{jp_stock}"
                elif market == 'SG':
                    currency = '$'
                    sg_stock = stock.replace('.SI', '')
                    link = f"https://www.tradingview.com/chart/dPvcvEPT/?symbol=SGX%3A{sg_stock}"

                message = f"<u><b>SELL {self.market} STOCK</b></u>\n" \
                    + f"Recommendation: {recommended_action}\n" \
                    + f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                    + f"Stock: <a href=\"{link}\">{inventory['Ticker']}</a> \n" \
                    + f"Current Price: {currency} {round(current_price,2)} \n" \
                    + f"Bought Price: {currency} {round(inventory['Buy Price'],2)} \n" \
                    + f"Bought Date: {inventory['Buy Date']} \n" \
                    + f"Position: {inventory['Position']} \n" \
                    + f"K: {np.round(inventory['K'],2)} \n" \
                    + f"D: {np.round(inventory['D'],2)} \n" \
                    + f"Potential: {np.round(inventory['Potential'],2)} \n" \
                    + f"Days On Market: {stock_utils.get_market_days(inventory['Buy Date'], self.day)} \n" \
                    + f"G/L: {currency}{np.round((current_price - inventory['Buy Price']) * inventory['Position'], 2)} ({np.round((current_price - inventory['Buy Price']) / inventory['Buy Price'] * 100, 2)}%) \n" \
                    
                t.send_message(message)
                
        

    def scan_buying_signal(self):
        current_dir = os.getcwd()
        inventory_file = os.path.join(current_dir, f'inventory/{self.market}_stoch.xlsx')
        inventories = pd.read_excel(inventory_file)
        t = telegram()

        # Check market breadth
        
        if(not self.benchmark_data["buy"].empty):
            if not self.benchmark_data["buy"][-1]:
                # Declines > Advanced
                advances = self.benchmark_data["advances"][-1]
                declines = self.benchmark_data["declines"][-1]
                message = f"<u><b>Market Breadth</b></u>\n" \
                    + f"Advances: {advances}\n" \
                    + f"Declines: {declines}\n" \
                    + f"<b>DO NOT TRADE TODAY</b> \n"
                    
                t.send_message(message)
                exit()
    
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
        
        
        good_stocks = stoch_order(model_recommended_stocks.items())
  
        
        # Push the buying signal to Telegram channel
        # Get "no_of_recommendations" most good probabilities stocks        
        if len(good_stocks) == 0:
            #print(f'No recommendation at {datetime.now().strftime("%H:%M:%S")} by {self.model.__name__}_{self.model_version} in market {self.market}')
            t.send_message(f'No recommendation at {datetime.now().strftime("%H:%M:%S")} in market {self.market}')
        else:    
            for key in list(good_stocks)[0:self.no_of_recommendations]:
                stock = key
                current_price = good_stocks[key][0]
                data = good_stocks[key][1]
                k = data['STOCHk'][-1]
                d = data['STOCHd'][-1]
                potential = data['potential'][-1]
                total_potential_stocks = len(good_stocks)

                
                today = date.today()      
                hold_till_date = stock_utils.get_market_real_date(self.market, today, self.hold_till)

                position_factor = self.buy_position_factor(inventories, current_price)
                advances = self.benchmark_data["advances"][-1]
                declines = self.benchmark_data["declines"][-1]

                currency = ''
                link = ''
                if market == 'HK' and ".HK" in stock:
                    currency = '$'
                    hk_stock = stock.replace('.HK', '')
                    link = f"https://www.tradingview.com/chart/?symbol=HKEX%3A{hk_stock}"
                elif market == 'HK' and not ".HK" in stock:
                    forex_ticker = stock.replace('=X', '')
                    link = f"https://www.tradingview.com/chart/?symbol=FX_IDC%3A{forex_ticker}"
                elif market == 'US':
                    currency = '$'
                    ticker = yf.Ticker(stock)
                    info = ticker.info
                    us_exchange_market = info['exchange']
                    f"https://www.tradingview.com/chart/?symbol={us_exchange_market}%3A{stock}"
                elif market == 'JP':
                    currency = '¥'
                    jp_stock = stock.replace('.T', '')
                    link = f"https://www.tradingview.com/chart/?symbol=TSE%3A{jp_stock}"
                elif market == 'SG':
                    currency = '$'
                    sg_stock = stock.replace('.SI', '')
                    link = f"https://www.tradingview.com/chart/?symbol=SGX%3A{sg_stock}"

                message = f"<u><b>BUY {self.market} STOCK</b></u>\n" \
                    + f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                    + f"Stock: <a href=\"{link}\">{stock}</a> \n" \
                    + f"Current Price: {currency} {round(current_price,2)} \n" \
                    + f"Take Profit at: {currency} {round(current_price * (1 + self.profit_perc), 2)} (+{round(self.profit_perc * 100, 2)}%) \n" \
                    + f"Stop at: {currency} {round(current_price * (1 - self.stop_perc), 2)} (-{round(self.stop_perc * 100, 2)}%) \n" \
                    + f"Position Factor: <b><u>{np.round(position_factor, 2)}</u></b>\n" \
                    + f"No. of recommendation: {total_potential_stocks}\n" \
                    + f"K: {np.round(k,2)}\n" \
                    + f"D: {np.round(d,2)}\n" \
                    + f"Potential: {np.round(potential,2)}\n" \
                    + f"Market Breadth: {int(np.round(advances/(advances+declines) * 100, 0))}:{int(np.round(declines/(advances+declines) * 100, 0))}\n" \
                    + f"Hold till: {(hold_till_date).strftime('%Y-%m-%d')} ({int(self.hold_till)} days)\n" 

                t.send_message(message)

    def buy_position_factor(self, inventories, buy_price):
        """
        this function determines how much capital to spend on the stock and returns the number of shares
        """

        # Find buy_perc
        no_of_on_hand_stocks = len(inventories[pd.isnull(inventories['Sold Date'])])
        no_of_available_split = self.no_of_splits - no_of_on_hand_stocks
        buy_perc = 1 / no_of_available_split


        # Find the last 5 trading history
        consecutive_win = 0
        consecutive_lose = 0

        history = inventories[~pd.isnull(inventories['Sold Date'])]

        for row in reversed(history):            
            if row['Sold Price'] - row['Buy Price'] > 0:
                if consecutive_lose == 0:
                    consecutive_win += 1
                else:
                    break
            else:
                if consecutive_win == 0:
                    consecutive_lose += 1
                else:
                    break
        
        if consecutive_win > 0:
            return buy_perc * min(3, consecutive_win)
        elif consecutive_lose > 0:
            return buy_perc * max(1/3, (4 - consecutive_lose)/3)
        else:
            return buy_perc

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
    
    def get_benchmark_data(self):
        """
        get benchmark data
        """
        #get benchmark data
        if self.market == 'JP':
            benchmark_ticker = '^N225'
        elif self.market == 'HK':
            benchmark_ticker = '^HSI'
        else: 
            benchmark_ticker = '^SPX'

        benchmark = yf.Ticker(benchmark_ticker)
        end = self.day
        start = stock_utils.get_market_real_date(self.market, end, -self.days_before_start_date)
        self.benchmark_data = benchmark.history(start = start, end = end, repair=True)


        # Calculate daily price changes
        price_changes = {}
        for s in [key for key in self.stock_data.keys()]:
            if self.stock_data[s].empty:
                continue
            if str(start.date()) not in self.stock_data[s].index:
                continue
            price_changes[s] = self.stock_data[s]['Close'].pct_change()[-1] if self.stock_data[s]['Close'].pct_change()[-1] is not np.NaN else 0

        
        
        # Calculate daily advances and declines
        advances = sum((price_changes[ticker] > 0) for ticker in [key for key in price_changes.keys()])
        declines = sum((price_changes[ticker] < 0) for ticker in [key for key in price_changes.keys()])

        # Calculate the Advance-Decline Line (ADL)
        adl = advances - declines

        ## Remove TimeZone
        self.benchmark_data = self.benchmark_data.tz_localize(None)

        self.benchmark_data['buy'] = adl > 0
        self.benchmark_data['advances'] = advances
        self.benchmark_data['declines'] = declines




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
    parser.add_argument('--profit_perc', type=float, help='Profit Percentage')
    parser.add_argument('--stop_perc', type=float, help='Stop Perc')
    parser.add_argument('--hold_till', type=float, help='Days to hold the stock')
    parser.add_argument('--no_of_splits', type=float, help='Number of Splits')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the arguments
    market = (args.market).upper()
    stock_list = args.stock_list
    profit_perc = args.profit_perc
    stop_perc = args.stop_perc
    hold_till = args.hold_till
    no_of_splits = args.no_of_splits

    """
    #Check if today is holiday
    market_holidays = getattr(holidays, market)()    
    today = date.today()
    if(today in market_holidays):
        exit()
    """
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
    
    sf = stockfinder_technical_breakout(market, stocks, stoch_k_d, 'v2', profit_perc=profit_perc, stop_perc= stop_perc, hold_till=hold_till, no_of_recommendations = 5, no_of_splits = no_of_splits)
    sf.scan_selling_signal()
    sf.scan_buying_signal()    


