import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class Stock:
    def __init__(self, ticker, window_size, take_profit_at):
        self.ticker = ticker
        self.window_size = window_size
        self.take_profit_at = take_profit_at
    def get_stock_data_and_massage(self):
        self.stock = yf.Ticker(self.ticker)
        self.hist = self.stock.history(period="max")

        # massage the dataset
        self.hist['future_ndays_max'] = self.hist['Close'].rolling(window=self.window_size).max().shift(-self.window_size + 1)
        self.hist['future_ndays_min'] = self.hist['Close'].rolling(window=self.window_size).min().shift(-self.window_size + 1)
        self.hist['profit'] = (self.hist['future_ndays_max'] - self.hist['Close'] ) / self.hist['Close']
        self.hist['take_profit'] = self.hist['profit'] >= self.take_profit_at

    def get_ready_for_ann(self):
        self.get_stock_data_and_massage()

        self.X_train = self.hist[['Close', 'Volume']][:len(self.hist) - 120].values
        self.y_train = self.hist[['profit']][:len(self.hist) - 120].values

        self.X_test = self.hist[['Close', 'Volume']][len(self.hist) - 120:].values
        self.y_test = self.hist[['profit']][len(self.hist) - 120 : ].values
        
        # normalize the training data
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.X_train = self.sc.fit_transform(self.X_train)




    def get_ready_for_lstm(self):
        self.get_stock_data_and_massage()

        # normalize the training data
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.n_closed_train = self.sc.fit_transform(self.hist[['Close']][:len(self.hist) - 120])

        # create 60 timestamps and 1 outout
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        for i in range(60, len(self.n_closed_train)):
            self.X_train.append(self.n_closed_train[i-60:i, 0])
            self.y_train.append(self.hist['profit'][i])

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        # reshape
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        
        ### Prepare testing data
        self.n_closed_test = self.sc.transform(self.hist[['Close']][len(self.hist) - 120 - 60:])
        for i in range(60, 120):
            self.X_test.append(self.n_closed_test[i-60:i, 0])
            self.y_test.append(self.hist['profit'][len(self.hist) - 120 + i])
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))

    def plot(self):
        hist = self.hist.copy()

        fig, ax=plt.subplots()
        ax.plot(hist.index, hist['Close'], color="red", marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close', color="red")

        ax2 = ax.twinx()
        hist['profit'][hist['profit'] < self.take_profit_at] = np.nan


        ax2.plot(hist.index, hist['profit'], color="blue" , marker="x", linestyle="")
        ax2.set_ylabel('profit', color="blue")

        plt.title(self.ticker)
        plt.legend()
        plt.show()
