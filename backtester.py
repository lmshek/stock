import numpy as np
from stock_utils.simulator import simulator
from stock_utils.stock_utils import get_stock_price
from models import lr_inference
from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
from models.lr_inference import LR_sell, LR_predict
from data_science.stats import create_stats
from data_science.create_figures import create_figures 
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

class backtester(simulator):

    def __init__(self, stocks_list, model, model_version, capital, start_date, end_date, threshold = 0.99, sell_perc = 0.08, hold_till = 10, stop_perc = 0.004, no_of_splits = 5):

        super().__init__(capital)

        yf.pdr_override()

        self.stocks = stocks_list
        self.model = model
        self.model_version = model_version
        self.start_date = start_date
        self.day = start_date
        self.end_date = end_date
        self.threshold = threshold        
        self.hold_till = hold_till
        self.sell_perc = sell_perc
        self.stop_perc = stop_perc
        self.no_of_splits_available = no_of_splits

        print('========== Back Test Data Parameters ============')
        print(f'No. of stocks: {len(self.stocks)}')
        print(f'Model: {self.model.__name__}_{self.model_version}')        
        print(f'Start Date: {self.start_date.strftime("%Y-%m-%d")}')
        print(f'End Date: {self.end_date.strftime("%Y-%m-%d")}')
        print(f'Threshold: {self.threshold}')
        print(f'Hold Till: {self.hold_till}')
        print(f'Sell Perc: {self.sell_perc}')
        print(f'Stop Perc: {self.stop_perc}')
        print(f'No of Splits: {self.no_of_splits_available}')

        #current directory
        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, 'results')
        folder_name = f'{str(self.model.__name__)}_{self.model_version}_{self.threshold}_{self.hold_till}_{self.sell_perc}_{self.stop_perc}'
        self.folder_dir = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.folder_dir):
            # create a new folder
            os.makedirs(self.folder_dir)

    def backtest(self):
        """
        Start backtesting
        """
        delta = timedelta(days = 1)
        hk_holidays = holidays.HK()

        #progress bar to track prgrress
        total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(desc = 'Progress', total = total_days)

        while self.day <= self.end_date:
            
            # Trade when today is not holiday and weekends
            if not(self.day in hk_holidays or hk_holidays._is_weekend(self.day)):                

                #check if any stock should sell, or any stock hit the maturity date 
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = LR_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, \
                        self.sell_perc, self.hold_till, self.stop_perc)
                    if recommended_action == "SELL":
                        self.sell(s, current_price, self.buy_orders[s][1], self.day, self.buy_orders[s][0])
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
                            recommanded_probability = recommand_params[0]
                            recommanded_price = recommand_params[2]

                            if recommanded_stock in self.buy_orders: # if we have already bought the stock, we will not buy it again.
                                continue
                            if no_of_stock_buy_today == 0: # we only buy 1 stock in 1 day
                                self.buy(recommanded_stock, recommanded_price, self.day, self.no_of_splits_available, recommanded_probability) # buy stock
                                self.no_of_splits_available -= 1
                                no_of_stock_buy_today += 1
                                print(f'{bcolors.HEADER}No. of splits available: {self.no_of_splits_available}{bcolors.ENDC}')
                            else:                    
                                print(f'Missed {len(self.daily_scanner.keys()) - no_of_stock_buy_today} other potential stocks on {self.day.strftime("%Y-%m-%d")}')
                                break
                    else:
                        print(f'No recommandations on {self.day.strftime("%Y-%m-%d")}')
                        pass

            #go to next day
            self.day += delta
            d += 1
            print('\n')
            pbar.update(1)
            print('\n')
            
        pbar.close()
        
        #sell the final stock and print final capital also print stock history
        self.print_bag()
        self.print_summary()
        self.save_results()
        return
    
    def get_stock_data(self, stock, back_to = 200):
        """
        this function queries to yf and get data of a particular stock on a given day back to certain amount of days
        (default is 30)
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)
        prediction, prediction_thresholded, close_price = self.model(self.model_version, stock, start, end, self.threshold, data_type="history", hold_till=self.hold_till)
        return prediction[0], prediction_thresholded, close_price

    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        for stock in self.stocks:
            try: #to ignore the stock if no data is available. #for aturdays or sundays etc
                prediction, prediction_thresholded, close_price = self.get_stock_data(stock)
                #if prediction greater than
                if prediction_thresholded < 1: #if prediction is zero
                    self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)
            except:
                pass
        def take_first(elem):
            return elem[1]      
        self.daily_scanner = OrderedDict(sorted(self.daily_scanner.items(), key = take_first, reverse = True))
   
    

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
        params = [self.threshold, self.hold_till, self.sell_perc, self.stop_perc, self.start_date, self.end_date]

        with open(results_summary_txt_path, 'w') as fp:
            fp.write('============== PARAMS ==============\n')
            fp.write(f'Model: {self.model.__name__}_{self.model_version} \n')
            fp.write(f'Threshold: {self.threshold} \n')
            fp.write(f'Hold Till: {self.hold_till} \n')
            fp.write(f'Sell Perc: {self.sell_perc} \n')
            fp.write(f'Stop Perc: {self.stop_perc} \n')
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
        
        """
            Create Statistics and Figures
        """
        create_stats(f'LR_predict_{self.model_version}', self.threshold, self.hold_till, self.sell_perc, self.stop_perc)
        create_figures(f'LR_predict_{self.model_version}', self.threshold, self.hold_till, self.sell_perc, self.stop_perc)

if __name__ == "__main__":
   
    # get stock tickers symobols
    current_dir = os.getcwd()

    stocks = []
    for stock_cat in ['hsi_integrated_large']: #'hsi_integrated_large', 'hsi_integrated_medium',
        stocks = stocks + pd.read_csv(os.path.join(current_dir, f'stock_list/hsi/{stock_cat}.csv'))['tickers'].tolist()
    stocks = list(np.unique(stocks)) 

    #stocks = ["6699.HK"]
    #hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
    #hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()
    #stocks = list(np.unique(hsi_tech + hsi_main))        
    #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()
    

    try:
        start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    except:
        start_date = datetime.now() - timedelta(days=365)

    try:
        end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    except:
        end_date = datetime.now() - timedelta(days = 1) 

    """
    Back Test different parameters
    """  
    backtester(stocks, LR_predict, 'v2', 100000, start_date = start_date, end_date = end_date, \
        threshold = 0.99, sell_perc= 0.1, hold_till= 10, stop_perc= 0.05, no_of_splits=3).backtest()
    """
    backtester(stocks, LR_predict, 'v2', 100000, start_date = start_date, end_date = end_date, \
        threshold = 0.95, sell_perc= 0.08, hold_till= 10, stop_perc= 0.08, no_of_splits=3).backtest()

    backtester(stocks, LR_predict, 'v2', 100000, start_date = start_date, end_date = end_date, \
        threshold = 0.95, sell_perc= 0.05, hold_till= 10, stop_perc= 0.05, no_of_splits=3).backtest()
    """
    
    

    

    


    

