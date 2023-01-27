import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('grayscale')

from scipy import linalg
import math
from datetime import datetime
import warnings
# warnings.filterwarnings("ignore")

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

class LR_training:

    def __init__(self, model_veresion, threshold = 0.98, start_date = None, end_date = None):

        self.model_version = model_veresion
        self.threshold = threshold

        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        
        # get stock tickers symobols
        hsi_tec = ['0020.HK', '0241.HK', '0268.HK', '0285.HK', '0700.HK', '0772.HK', '0909.HK', \
            '0981.HK', '0992.HK', '1024.HK', '1347.HK', '1810.HK', '1833.HK', '2015.HK', '2018.HK', \
            '2382.HK', '3690.HK', '3888.HK', '6060.HK', '6618.HK', '6690.HK', '9618.HK', '9626.HK', \
            '9698.HK', '9866.HK', '9869.HK', '9888.HK', '9961.HK', '9988.HK', '9999.HK']
        hsi_main = ['0001.HK','0002.HK','0003.HK','0005.HK','0006.HK','0011.HK','0012.HK','0016.HK','0017.HK','0027.HK','0066.HK','0101.HK','0175.HK','0241.HK','0267.HK','0288.HK','0291.HK','0316.HK','0322.HK','0386.HK','0388.HK','0669.HK','0688.HK','0700.HK','0762.HK','0823.HK','0857.HK','0868.HK','0881.HK','0883.HK','0939.HK','0941.HK','0960.HK','0968.HK','0981.HK','0992.HK','1038.HK','1044.HK','1088.HK','1093.HK','1109.HK','1113.HK','1177.HK','1209.HK','1211.HK','1299.HK','1378.HK','1398.HK','1810.HK','1876.HK','1928.HK','1929.HK','1997.HK','2007.HK','2020.HK','2269.HK','2313.HK','2318.HK','2319.HK','2331.HK','2382.HK','2388.HK','2628.HK','2688.HK','3690.HK','3692.HK','3968.HK','3988.HK','6098.HK','6690.HK','6862.HK','9618.HK','9633.HK','9888.HK','9988.HK','9999.HK']
    
        stocks = list(np.unique(hsi_tec + hsi_main))
        self.stocks = list(np.unique(stocks))

        #main dataframe
        self.main_df = pd.DataFrame(columns = ['Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target'])

        # init models
        self.scalar = MinMaxScaler()
        self.lr = LogisticRegression()

        # run logistic regression
        self.fetch_data()
        self.create_train_test()
        self.fit_model()
        self.confusion_matrix()
        self.save_model()


    def fetch_data(self):
        for stock in self.stocks:
            try:
                df = stock_utils.create_train_data(stock, n = 10)
                self.main_df = pd.concat([self.main_df, df])
            except:
                pass
        print(f'{len(self.main_df)} samples were fetched')

    def create_train_test(self):
        self.main_df = self.main_df.sample(frac = 1, random_state = 3).reset_index(drop = True)
        self.main_df['target'] = self.main_df['target'].astype('category')

        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scalar.fit_transform(self.main_df)

        #test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.05, random_state=50, shuffle= True)

        print('Created tset and train data...')

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
        saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
        model_file = f'lr_{self.model_version}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scalar, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {saved_models_dir}')
        cm_path = os.path.join(os.getcwd(), 'results/Confusion Matrices')

        #save cms
        plt.figure()
        self.cmd.plot()
        plt.savefig(f'{cm_path}/cm_{self.model_version}.jpg')

        plt.figure()
        self.cmd_thresholded.plot()
        plt.savefig(f'{cm_path}/cm_thresholded_{self.model_version}.jpg')
        print(f'Figures saved in {cm_path}')


if __name__ == "__main__":
    run_lr = LR_training('v2')


