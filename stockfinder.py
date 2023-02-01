import pandas_datareader as web
from datetime import date
import holidays
import numpy as np
import pandas as pd
from models.lr_inference import LR_v1_sell, LR_v1_predict
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
from telegram.telegram import telegram



class stockfinder:

    def __init__(self, stocks_list, model, threshold = 0.75, sell_perc = 0.08, hold_till = 5, stop_perc = 0.08, no_of_recommendations = 3):
        self.stocks = stocks_list
        self.model = model
        self.threshold = threshold
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc
        self.no_of_recommendations = no_of_recommendations

    def scan(self):
    
        model_recommended_stocks = {}
        for stock in self.stocks:
            try:
                prediction, prediction_thresholded, current_price , is_ignored = self.model(stock, '', '', self.threshold, data_type="realtime")

                if not is_ignored and prediction_thresholded < 1:
                    model_recommended_stocks[stock] = (prediction, prediction_thresholded, current_price)
            except:
                pass   
        def take_first(elem):
            return elem[1]  
        # Get 3 most good probabilities stocks
        good_stocks = OrderedDict(sorted(model_recommended_stocks.items(), key = take_first, reverse = True))[0:(self.no_of_recommendations - 1)]
        
        # Push the buying signal to Telegram channel
        for key, value in good_stocks.items():
            stock = key
            current_price = value[2]
            threshold = self.threshold
            sell_perc = self.sell_perc
            hold_till = self.hold_till
            stop_perc = self.stop_perc
            prediction_probability = value[0]

            telegram.send_formatted_message(stock=stock, prediction_probability=prediction_probability, current_price=current_price, threshold=threshold, sell_perc=sell_perc, hold_till=hold_till, stop_perc=stop_perc)





if __name__ == "__main__":

    #Check if today is holiday
    hk_holidays = holidays.HK()
    today = date.today()
    if(today in hk_holidays):
        exit()
    


    current_dir = os.getcwd()    
    hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
    hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()
    stocks = list(np.unique(hsi_tech + hsi_main))       
    #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()

    stockfinder(stocks, LR_v1_predict, threshold = 0.95, sell_perc = 0.1, hold_till= 21, stop_perc = 0.05, no_of_recommendations = 3).scan()


