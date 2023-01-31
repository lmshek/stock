import pandas_datareader as web
import datetime
import time
import numpy as np
import pandas as pd
from models.lr_inference import LR_v1_sell, LR_v1_predict
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os



class stockfinder:

    def __init__(self, stocks_list, model, threshold = 0.75):
        self.stocks = stocks_list
        self.model = model
        self.threshold = threshold
        self.good_stocks = []

    def scan(self):
    
        for stock in self.stocks:
            try:
                prediction, prediction_thresholded, close_price , is_ignored = self.model(stock, '', '', self.threshold, data_type="realtime")

                if not is_ignored and prediction_thresholded < 1:
                    self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)
            except:
                pass   
        def take_first(elem):
            return elem[1]      
        self.good_stocks = self.good_stocks.append(OrderedDict(sorted(self.daily_scanner.items(), key = take_first, reverse = True)))



if __name__ == "__main__":
    current_dir = os.getcwd()
    
    hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
    hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()

    stocks = list(np.unique(hsi_tech + hsi_main))       
    #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()

    finder = stockfinder(stocks, LR_v1_predict, 0.75)
    finder.scan()

    for good_stock in finder.good_stocks:
        print(f"BUY {good_stock}")


        
    
    #


