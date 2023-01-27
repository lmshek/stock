import numpy as np
from stock_utils.simulator import simulator
from stock_utils.stock_utils import get_stock_price
from models import lr_inference
from datetime import datetime, timedelta
import pandas as pd
from models.lr_inference import LR_v1_sell, LR_v1_predic
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
import pickle
from tqdm import tqdm

class backtester(simulator):

    def __init__(self, stocks_list, model, capital, start_date, end_date, threshold = 0.99, sell_perc = 0.04, hold_till = 5, stop_perc = 0.005):

        super().__init__(capital)

        self.stocks = stocks_list
        self.model = model
        self.start_date = start_date
        self.day = start_date
        self.end_date = end_date
        self.status = 'buy' # the status says if the backtester is in nuy mode or sell mode
        self.threshold = threshold
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc

        #current directory
        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, 'results')
        folder_name = f'{str(self.model.__name__)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.folder_dir):
            # create a new folder
            os.makedirs(self.folder_dir)

    def backtest(self):
        """
        Start backtesting
        """
        delta = timedelta(days = 1)

        #progress bar to track prgrress
        total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(desc = 'Progress', total = total_days)

        while self.day <= self.end_date:

            #daily scanner dict
            self.daily_scanner = {}
            if self.status == 'buy':
                #scan stocks for the day
                self.scanner()
                if list(self.daily_scanner.keys()) != []:
                    recommanded_stock = list(self.daily_scanner.keys())[0]
                    recommanded_price = list(self.daily_scanner.values())[0][2]
                    self.buy(recommanded_stock, recommanded_price, self.day) # buy stock
                    print(f'Bought {recommanded_stock} for {recommanded_price} on the {self.day}')
                    self.status = 'sell' #change the status to sell
                else:
                    print('No recommandations')
                    pass
            else:
                #if the status is sell, get stock price on the day
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = LR_v1_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, \
                        self.sell_perc, self.hold_till, self.stop_perc)
                    if recommended_action == "SELL":
                        print(f'Sold {s} for {current_price} on {self.day}')
                        self.sell(s, current_price, self.buy_orders[s][1], self.day)
                        self.status = 'buy'
            
            #go to next day
            self.day += delta
            d += 1
            pbar.update(1)
        pbar.close()
        #sell the final stock and print final capital also print stock history
        self.print_bag()
        self.print_summary()
        self.save_results()
        return
    
    def get_stock_data(self, stock, back_to = 40):
        """
        this function queries to yf and get data of a particular stock on a given day back to certain amount of days
        (default is 30)
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)
        prediction, prediction_thresholded, close_price = self.model(stock, start, end, self.threshold)
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
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        results_summary = [self.initial_capital, self.total_gain]
        params_path = os.path.join(self.folder_dir, 'params')
        params = [self.threshold, self.hold_till, self.sell_perc, self.stop_perc, self.start_date, self.end_date]

        with open(results_summary_path, 'wb') as fp:
            pickle.dump(results_summary, fp)
        with open(params_path, 'wb') as fp:
            pickle.dump(params, fp)

if __name__ == "__main__":
    #stocks list
    hsi_tec = ['0020.HK', '0241.HK', '0268.HK', '0285.HK', '0700.HK', '0772.HK', '0909.HK', \
        '0981.HK', '0992.HK', '1024.HK', '1347.HK', '1810.HK', '1833.HK', '2015.HK', '2018.HK', \
        '2382.HK', '3690.HK', '3888.HK', '6060.HK', '6618.HK', '6690.HK', '9618.HK', '9626.HK', \
        '9698.HK', '9866.HK', '9869.HK', '9888.HK', '9961.HK', '9988.HK', '9999.HK']
    hsi_main = ['0001.HK','0002.HK','0003.HK','0005.HK','0006.HK','0011.HK','0012.HK','0016.HK','0017.HK','0027.HK','0066.HK','0101.HK','0175.HK','0241.HK','0267.HK','0288.HK','0291.HK','0316.HK','0322.HK','0386.HK','0388.HK','0669.HK','0688.HK','0700.HK','0762.HK','0823.HK','0857.HK','0868.HK','0881.HK','0883.HK','0939.HK','0941.HK','0960.HK','0968.HK','0981.HK','0992.HK','1038.HK','1044.HK','1088.HK','1093.HK','1109.HK','1113.HK','1177.HK','1209.HK','1211.HK','1299.HK','1378.HK','1398.HK','1810.HK','1876.HK','1928.HK','1929.HK','1997.HK','2007.HK','2020.HK','2269.HK','2313.HK','2318.HK','2319.HK','2331.HK','2382.HK','2388.HK','2628.HK','2688.HK','3690.HK','3692.HK','3968.HK','3988.HK','6098.HK','6690.HK','6862.HK','9618.HK','9633.HK','9888.HK','9988.HK','9999.HK']

    stocks = list(np.unique(hsi_tec + hsi_main))
    back = backtester(hsi_main, LR_v1_predic, 50000, datetime(2022, 1, 1), datetime(2022,12,31), threshold = 0.9, sell_perc= 0.05, hold_till= 10, stop_perc= 0.05)

    back.backtest()
    

