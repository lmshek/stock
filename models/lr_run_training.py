import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('grayscale')

from scipy import linalg
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import time
import os
import sys
import pickle

# append path
current_dir = os.getcwd()
sys.path.append(current_dir)

from stock_utils import stock_utils
from lr_run_training_stock import LR_training_stock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from telegram.telegram import telegram

class LR_training:

    def __init__(self, model_version = "v1", threshold = 0.75, start_date = None, end_date = None, n = 10, take_profit_rate = 0.1, stop_lose_rate = 0.05, stock_cats = ['hsi_tech', 'hsi_main']):

        self.model_version = model_version
        self.threshold = threshold
        self.take_profit_rate = take_profit_rate
        self.stop_lose_rate = stop_lose_rate
        self.n = n

        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        print('========== Training Data Parameters ============')
        print(f'Start Date: {self.start_date}')
        print(f'End Date: {self.end_date}')
        print(f'Threshold: {self.threshold}')
        print(f'N: {self.n}')
        print(f'Take Profit Rate: {self.take_profit_rate}')
        print(f'Stop Lose Rate: {self.stop_lose_rate}')

        # get stock tickers symobols
        current_dir = os.getcwd()
        stocks = []
        for stock_cat in stock_cats:
            stocks = stocks + pd.read_csv(os.path.join(current_dir, f'stock_list/hsi/{stock_cat}.csv'))['tickers'].tolist()
        stocks = list(np.unique(stocks)) 

        #hsi_tech = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_tech.csv'))['tickers'].tolist()
        #hsi_main = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_main.csv'))['tickers'].tolist()
        #stocks = list(np.unique(hsi_tech + hsi_main))        
        #stocks = pd.read_csv(os.path.join(current_dir, 'stock_list/hsi/hsi_all.csv'))['tickers'].tolist()
        self.stocks = list(np.unique(stocks))

        for stock in self.stocks:
            try:
                LR_training_stock('lr', 'v3', stock, threshold=self.threshold,
                take_profit_rate=self.take_profit_rate, stop_lose_rate=self.stop_lose_rate,
                start_date=self.start_date, end_date=self.end_date, n=self.n)
            except:
                pass

if __name__ == "__main__":
    # Setup parameters
    try:
        threshold = sys.argv[1]
    except:
        threshold = 0.95

    try:
        start_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    except:
        start_date = datetime.now() - timedelta(days=10*365)

    try:
        end_date = datetime.strptime(sys.argv[3], "%Y-%m-%d")
    except:
        end_date = datetime.now() - timedelta(days = 1) 

    try: 
        n = int(sys.argv[4])
    except:
        n = 10

    try:
        take_profit_rate = sys.argv[5]
    except:
        take_profit_rate = 0.1

    try:
        stop_lose_rate = sys.argv[6]
    except:
        stop_lose_rate = 0.05

    # Start training
    run_lr = LR_training('v3', threshold=0.95, start_date= start_date, end_date=end_date, n=n, take_profit_rate=take_profit_rate, stop_lose_rate=stop_lose_rate, stock_cats=['hsi_all'])
    #run_lr = LR_training('v2', threshold=0.95, start_date= start_date, end_date=end_date, n=n, stock_cats=['hsi_integrated_large', 'hsi_integrated_medium', 'hsi_tech'])


