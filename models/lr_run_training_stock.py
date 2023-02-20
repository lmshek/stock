import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('grayscale')

from scipy import linalg
import math
from datetime import datetime, timedelta, date
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from telegram.telegram import telegram

class LR_training_stock:

    def __init__(self, model_name, model_version = "v1", stock = '0001.HK', 
        threshold = 0.75, take_profit_rate = 0.1, stop_lose_rate = 0.05, 
        start_date = None, end_date = None, n = 10, 
        cols_of_interest = ['Volume', 'normalized_value', 
            '10_sma', '50_sma', '200_sma', 
            '10_rsi', '50_rsi', '200_rsi', 
            'target']):

        self.model_name = model_name
        self.model_version = model_version
        self.threshold = threshold
        self.stock = stock
        self.n = n
        self.take_profit_rate = take_profit_rate
        self.stop_lose_rate = stop_lose_rate

        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        # init models
        self.scalar = MinMaxScaler()
        self.lr = LogisticRegressionCV(cv=5)

        self.cols_of_interest = ['Volume', 'normalized_value', 
            '10_sma', 
            '50_sma', 
            '200_sma', 
            '10_rsi', 
            '50_rsi', 
            '200_rsi', 
            'target']

        self.create_directory()
        self.fetch_data()
        self.create_train_test()
        self.fit_model()
        self.confusion_matrix()
        self.save_model()

    def create_directory(self):
        self.model_dir = os.path.join(os.getcwd(), f'saved_models/{self.model_name}_{self.model_version}')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.stock_dir = os.path.join(os.getcwd(), f'saved_models/{self.model_name}_{self.model_version}/{self.stock}')
        if not os.path.exists(self.stock_dir):
            os.makedirs(self.stock_dir)

    def fetch_data(self):
        self.main_df = stock_utils.create_train_data(self.stock, 
                    start_date = self.start_date, 
                    end_date = self.end_date, 
                    n = self.n,
                    cols_of_interest= self.cols_of_interest,
                    take_profit_rate= self.take_profit_rate,
                    stop_lose_rate=self.stop_lose_rate
                    )

    def create_train_test(self):
        self.main_df = self.main_df.sample(frac = 1, random_state = 3).reset_index(drop = True)
        self.main_df['target'] = self.main_df['target'].astype('category')

        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scalar.fit_transform(self.main_df)

        #test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.2, random_state=50, shuffle= True)

        print('Created test and train data...')

    def fit_model(self):

        print('Training model...')
        self.lr.fit(self.train_x, self.train_y)

        # predict the test data
        self.predictions = self.lr.predict(self.test_x)
        self.score = self.lr.score(self.test_x, self.test_y)
        print(f'Logistic regression model score: {self.score}')

        #preds with threshold
        self.predictions_proba = self.lr._predict_proba_lr(self.test_x)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)

        score_file = f'score_{self.model_version}.txt'
        score_dir = os.path.join(os.getcwd(), self.stock_dir, score_file)

        with open(score_dir, 'w') as fp:
            fp.write(f'Logistic Regression Model Score: {self.score} \n')

    def confusion_matrix(self):
        cm = confusion_matrix(self.test_y, self.predictions)
        self.cmd = ConfusionMatrixDisplay(cm)

        cm_thresholded = confusion_matrix(self.test_y, self.predictions_proba_thresholded)
        self.cmd_thresholded = ConfusionMatrixDisplay(cm_thresholded)

    def _threshold(self, predictions, threshold):
        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]

        return np.array(prob_thresholded)

    def save_model(self):
        #save models        
        model_file = f'lr_{self.model_version}.sav'
        model_dir = os.path.join(self.stock_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(self.stock_dir, scaler_file)
        pickle.dump(self.scalar, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {self.stock_dir}')
        cm_path = self.stock_dir

        #save cms
        plt.figure()
        self.cmd.plot()
        plt.savefig(f'{cm_path}/cm.jpg')

        plt.figure()
        self.cmd_thresholded.plot()
        plt.savefig(f'{cm_path}/cm_thresholded.jpg')
        print(f'Figures saved in {cm_path}')

if __name__ == "__main__":
    run_lr = LR_training_stock('lr', 'v3', '0388.HK', 
    threshold=0.95, take_profit_rate=0.1, stop_lose_rate=0.1,
    start_date= date(2012,1,1), end_date=date(2022,1,1), n=10)
    