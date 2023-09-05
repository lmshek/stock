import numpy as np
from stock_utils.simulator_trend import simulator_trend
from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
from technical.trend import trend, trend_buy, trend_sell, trend_order, trend_print
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
import pickle
from tqdm import tqdm
import pandas_datareader.data as web
from stock_utils.bcolors import bcolors
import sys
import holidays
from telegram.telegram import telegram
from stock_utils import stock_utils
import traceback
import argparse

class backtester(simulator_trend):

    def __init__(self, market, stocks_list, model, model_version, capital, start_date, end_date, no_of_splits = 5):

        super().__init__(capital)

        yf.pdr_override()

        self.market = market
        self.stocks = stocks_list
        self.model = model
        self.model_version = model_version
        self.start_date = start_date
        self.day = start_date
        self.end_date = end_date      
        self.no_of_splits_available = no_of_splits
        self.stock_data = {}
        self.days_before_start_date = 400

        print('========== Back Test Data Parameters ============')
        print(f'Market: {self.market}')
        print(f'No. of stocks: {len(self.stocks)}')
        print(f'Model: {self.model.__name__}_{self.model_version}')        
        print(f'Start Date: {self.start_date.strftime("%Y-%m-%d")}')
        print(f'End Date: {self.end_date.strftime("%Y-%m-%d")}')        
        print(f'No of Splits: {self.no_of_splits_available}')

        #current directory
        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, 'results')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_name = f'{str(self.model.__name__)}_{self.model_version}_{self.market}_{current_datetime}'
        self.folder_dir = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.folder_dir):
            # create a new folder
            os.makedirs(self.folder_dir)

        # Get Stock Data     
        sbar = tqdm(desc = 'Downloading Stock Data', total = len(self.stocks))   
        for ticker in self.stocks:
            try:
                stock = yf.Ticker(ticker)
                self.stock_data[ticker] = stock.history(start = stock_utils.get_market_real_date(self.market, start_date, -self.days_before_start_date), end = end_date.date() + timedelta(days = 1), repair="silent", raise_errors=True, rounding=True)
                #self.stock_data[ticker] = stock.history(period="max", repair="silent", raise_errors=True, rounding=True, keepna=True)
            except Exception as e:
                # Print the exception message
                print("An exception occurred:", str(e))

                # Print the stack trace
                traceback.print_exc()
                continue
            finally:
                sbar.update(1)
        print('\n')
        sbar.close()

    def backtest(self):
        """
        Start backtesting
        """
        delta = timedelta(days = 1)
        market_holidays = getattr(holidays, self.market)()

        #progress bar to track prgrress
        total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(desc = 'Simulation Progress', total = total_days)

        while self.day <= self.end_date:
            
            # Trade when today is not holiday and weekends
            if not(self.day in market_holidays or market_holidays._is_weekend(self.day)):                

                #check if any stock should sell, or any stock hit the maturity date 
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = trend_sell(self.stock_data, self.market, s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, self.buy_orders[s][4], self.buy_orders[s][5], self.buy_orders[s][6], self.buy_orders[s][7])
                    if "SELL" in recommended_action:
                        self.sell(s, current_price, self.buy_orders[s][1], self.day, self.buy_orders[s][0], recommended_action, self.buy_orders[s][4], self.buy_orders[s][5], self.buy_orders[s][6], self.buy_orders[s][7])
                        self.no_of_splits_available += 1
                        print(f'{bcolors.HEADER}No. of splits available: {self.no_of_splits_available}{bcolors.ENDC}')
                
                # if we still have avilable splits, then scan stocks to buy; Otherwise, go to next day
                if self.no_of_splits_available > 0:                        
                    #daily scanner dict
                    self.daily_scanner = {}
                    #scan potential stocks for the day
                    self.scanner()            
                    #check if any recommended stocks to buy today
                    if list(self.daily_scanner.keys()) != []:
                        no_of_stock_buy_today = 0
                        for daily_recommanded_stock, recommand_params in self.daily_scanner.items():
                            recommanded_stock = daily_recommanded_stock
                            data = recommand_params[1]
                            recommanded_price = recommand_params[0]

                            if recommanded_stock in self.buy_orders: # if we have already bought the stock, we will not buy it again.
                                continue
                            if no_of_stock_buy_today == 0: # we only buy 1 stock in 1 day
                                trend_buy(self, recommanded_stock, recommanded_price, self.day, self.no_of_splits_available, data['wave_1_max'][-1], data['slope'][-1], data['wave_depth'][-1], data['wave_length'][-1]) # buy stock
                                self.no_of_splits_available -= 1
                                no_of_stock_buy_today += 1   
                                trend_print(data)                             
                                print(f'{bcolors.HEADER}No. of splits available: {self.no_of_splits_available}{bcolors.ENDC}')
                            else:                    
                                print(f'Missed {len(self.daily_scanner.items()) - 1} potential stocks on {self.day.strftime("%Y-%m-%d")}')                                
                                break                                
                    else:
                        print(f'No recommandations on {self.day.strftime("%Y-%m-%d")}')
                        pass
                
                self.print_bag(self.stock_data, self.day)

            #go to next day
            self.day += delta
            d += 1
            print('\n')
            pbar.update(1)
            print('\n')
            
        pbar.close()
        
        #sell the final stock and print final capital also print stock history
        #self.print_bag()
        self.print_summary()
        self.save_results()
        return
    
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

    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        daily_pbar = tqdm(desc = 'Daily Scanning Progress', total = len(self.stocks))
        for stock in self.stocks:
            try:
                #to ignore the stock if no data is available. #for aturdays or sundays etc
                buy_signal, close_price, today_stock_data, multiplier = self.get_stock_data(stock)
                #print(f'Scanned {stock}')
                #if prediction greater than
                if buy_signal: 
                    self.daily_scanner[stock] = (close_price, today_stock_data, multiplier)
            except:
                pass
            finally:
                daily_pbar.update(1)
                
        print('\n')
        daily_pbar.close()
        
        self.daily_scanner = trend_order(self.daily_scanner.items())

        
    def save_results(self):
        """
        save history dataframe create figures and save
        """           
        #save csv file
        results_df_path = os.path.join(self.folder_dir, 'history_df.csv')
        self.history_df.to_csv(results_df_path, index = False)

        #save params and results summary
        results_summary_txt_path = os.path.join(self.folder_dir, 'results_summary.txt')
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        results_summary = [self.initial_capital, self.total_gain]
        params_path = os.path.join(self.folder_dir, 'params')
        params = [self.start_date, self.end_date]

        with open(results_summary_txt_path, 'w') as fp:
            fp.write('============== PARAMS ==============\n')
            fp.write(f'Market: {self.market}\n')
            fp.write(f'No. of stocks: {len(self.stocks)}\n')
            fp.write(f'Model: {self.model.__name__}_{self.model_version} \n')
            fp.write(f'Start Date: {self.start_date} \n')
            fp.write(f'End Date: {self.end_date} \n')
            fp.write('\n')
            fp.write('============== Result Summary ==============\n')
            fp.write(f'Initial Balance: {self.initial_capital:.2f} \n')
            fp.write(f'Final Balance: {(self.initial_capital + self.total_gain):.2f} \n')
            fp.write(f'Total Gain: {self.total_gain:.2f} \n')
            fp.write(f'P/L: {(self.total_gain / self.initial_capital) * 100:.2f} % \n')
            
        # Send telegram message about the backtest result summary
        with open(results_summary_txt_path) as fp:
            contents = fp.readlines()
            t = telegram()
            t.send_message(message=''.join([str(elem) for elem in contents]))

        

        with open(results_summary_path, 'wb') as fp:
            pickle.dump(results_summary, fp)
        with open(params_path, 'wb') as fp:
            pickle.dump(params, fp)
     

    
      
if __name__ == "__main__":

    # Get the arguments from command line

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add an argument to accept a list of strings
    parser.add_argument('--market', type=str, help='Country of the Market (e.g. HK, US)')
    parser.add_argument('--stock_list', nargs='+', help='List of the stocks (e.g. hsi_main, dow_jones, nasdaq_100)')
    parser.add_argument('--start_date', type=str, help='Backtest Start Date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Backtest End Date (YYYY-MM-DD)')
    parser.add_argument('--days_before_today_as_start', type=int, help='Backtest Days Before Today as Start Date')
    parser.add_argument('--days_before_today_as_end', type=int, help='Backtest Days Before Today as End Date')
    parser.add_argument('--no_of_split', type=int, help='Max Number of Holding Stock')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the arguments
    market = (args.market).upper()
    stock_list = args.stock_list
    start_date = args.start_date
    end_date = args.end_date
    days_before_today_as_start = args.days_before_today_as_start
    days_before_today_as_end = args.days_before_today_as_end
    no_of_split = args.no_of_split
   
    # get stock tickers symobols
    current_dir = os.getcwd()

    stocks = []
    for stock_cat in stock_list: #'nasdaq_100', 'dow_jones', 'hsi_integrated_large', 'hsi_integrated_medium', 'hsi_integrated_small',
        stocks = stocks + pd.read_csv(os.path.join(current_dir, f'stock_list/{stock_cat}.csv'))['tickers'].tolist()
    stocks = list(np.unique(stocks)) 
    
    #stocks = ["ORLY"]
    #hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
    #hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()
    #stocks = list(np.unique(hsi_tech + hsi_main))        
    #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()
    

    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    except:
        start_date = datetime.now() - timedelta(days = days_before_today_as_start)

    try:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except:
        end_date = datetime.now() - timedelta(days = days_before_today_as_end) 

    if no_of_split == None:
        no_of_split = 5

    """
    Back Test different parameters
    """  
    backtester(market, stocks, trend, 'v2', 1000000, start_date = start_date, end_date = end_date, no_of_splits=no_of_split).backtest()



    

    


    

