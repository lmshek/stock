import numpy as np
import math
import pandas as pd
from stock_utils.bcolors import bcolors
from datetime import datetime, timedelta
from stock_utils import stock_utils
from telegram.telegram import telegram
import csv
import os
import yfinance as yf
import pandas_ta as ta

class simulator_stoch:

    def __init__(self, buying_power):
        self.buying_power = buying_power
        self.initial_capital = buying_power # keep a copy of the initial cpaital
        self.total_gain = 0
        self.buy_orders = {}
        self.history = []

        #create a pandas df to save history
        cols = ['stock', 'buy_price', 'n_shares', 'sell_price', 'net_gain', 'gain_perc', 'buy_date', 'sell_date', 'total_days_on_market', 'cup_len', 'handle_len', 'cup_depth', 'handle_depth', 'rrr', 'threshold']
        self.history_df = pd.DataFrame(columns = cols)
    
    def buy(self, stock, buy_price, buy_date, no_of_splits, k, d, potential):
    
        #function takes buy price and the number of shares and buy the stock
        #         
        n_shares = self.buy_percentage(buy_price, 1/no_of_splits)
        if n_shares > 0:
            self.buying_power = self.buying_power - buy_price * n_shares * (1 + self.commission_fee)
            self.buy_orders[stock] = [buy_price, n_shares, buy_price * n_shares, buy_date, k, d, potential]
            self.log(f'{bcolors.OKCYAN}Bought {stock} for {buy_price} with k: {np.round(k,2)}, d: {np.round(d,2)} and potential: {np.round(potential,2)} on the {buy_date.strftime("%Y-%m-%d")} . Buying Power: {self.buying_power}{bcolors.ENDC}')

            return True
        else:
            return False



    def sell(self, stock, sell_price, n_shares_sell, sell_date, buy_price, recommended_action, k, d, potential):
        """
        function to sell stock given the stock name and number of shares
        """
        buy_price, n_shares, _, buy_date, _, _, _= self.buy_orders[stock]
        sell_amount = sell_price * (n_shares_sell)

        self.buying_power = self.buying_power + sell_amount * (1 + self.commission_fee)

        if(n_shares - n_shares_sell) == 0: #if sold all
            self.history.append([stock, buy_price, n_shares, sell_price, buy_date, sell_date, k, d, potential])
            del self.buy_orders[stock]
        else:
            n_shares = n_shares - n_shares_sell
            self.buy_orders[stock][1] = n_shares
            self.buy_orders[stock][2] = buy_price * n_shares
        
        profit = sell_price - buy_price 
               
        if profit > 0:
            self.log(f'{bcolors.OKGREEN}{recommended_action} - Sold {stock} for {sell_price} (Make profit {round(profit / buy_price * 100, 2)}%) on {sell_date.strftime("%Y-%m-%d")}. Buying Power: {self.buying_power}{bcolors.ENDC}')
        else:
            self.log(f'{bcolors.FAIL}{recommended_action} - Sold {stock} for {sell_price} (Lose money {round(profit / buy_price * 100, 2)}%) on {sell_date.strftime("%Y-%m-%d")}. Buying Power: {self.buying_power}{bcolors.ENDC}')

    def buy_percentage(self, buy_price, buy_perc = 1):
        """
        this function determines how much capital to spend on the stock and returns the number of shares
        """

        # Find the last 5 trading history
        consecutive_win = 0
        consecutive_lose = 0

        for row in reversed(self.history):            
            if row[3] - row[1] > 0:
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
            stock_expenditure = self.buying_power * buy_perc * (1 + min(consecutive_win, 5) / 5)
        elif consecutive_lose > 0:
            stock_expenditure = self.buying_power * buy_perc * (1 - min(consecutive_lose, 5) / 5)
        else:
            stock_expenditure = self.buying_power * buy_perc

        if stock_expenditure > self.buying_power:
            n_shares = math.floor(self.buying_power / buy_price)
        else:
            n_shares = math.floor(stock_expenditure / buy_price)

        if n_shares < 100 and 100 * buy_price < self.buying_power:
            n_shares = 100

        ## To maximize the position
        if n_shares > 100 and int(math.ceil(n_shares / 100) * 100) * buy_price < self.buying_power:
            n_shares = int(math.ceil(n_shares / 100) * 100)
        
            
        
        return int(n_shares // 100) * 100
    
    def print_bag(self, all_time_stocks_data, date):
        """
        print current stocks holding
        """
        print_bag = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<14} {:<10} {:<10} {:<10}".format('DATE', 'STOCK', 'BUY PRICE', 'TODAY PRICE', 'GAIN%', 'SHARES', 'TODAY VALUE', 'DAYS_ON_MARKET', 'K', 'D', 'POTENTIAL')
        today_stock_value = 0.0
        for key, value in self.buy_orders.items():       
            try:    
                hist = all_time_stocks_data[key].loc[date.date():date.date()]
                close_price = hist['Close'][-1]
                days_on_market = stock_utils.get_market_days(value[3], date)
                k = value[4]
                d = value[5]
                potential = value[6]
                today_stock_value += value[1] * close_price
                if(close_price >= value[0]):                
                    print_bag += f'\n{bcolors.OKGREEN}{str(date.date()):<10} {key:<10} {np.round(value[0], 2):<10} {np.round(close_price, 2):<10}  {np.round((close_price - value[0]) / value[0] * 100, 2):<10} {value[1]:<10} {np.round(close_price * value[1], 2):<10} {days_on_market:<14} {np.round(k, 3):<10} {np.round(d, 3):<10} {np.round(potential, 3):<10}{bcolors.ENDC}'
                else:
                    print_bag += f'\n{bcolors.FAIL}{str(date.date()):<10} {key:<10} {np.round(value[0], 2):<10} {np.round(close_price, 2):<10}  {np.round((close_price - value[0]) / value[0] * 100, 2):<10} {value[1]:<10} {np.round(close_price * value[1], 2):<10} {days_on_market:<14} {np.round(k, 3):<10} {np.round(d, 3):<10} {np.round(potential, 3):<10}{bcolors.ENDC}'
            except:
                continue

        if self.buying_power + today_stock_value >= self.initial_capital:
            print_bag += f'\n{bcolors.OKGREEN}Today Capital: {np.round(self.buying_power + today_stock_value, 2)} ({np.round(((self.buying_power + today_stock_value) / self.initial_capital * 100),2)}%){bcolors.ENDC}'
            print_bag += f'\n{bcolors.OKGREEN}Buying Power: {np.round(self.buying_power, 2)}{bcolors.ENDC}'
        else:
            print_bag += f'\n{bcolors.FAIL}Today Capital: {np.round(self.buying_power + today_stock_value, 2)} ({np.round(((self.buying_power + today_stock_value) / self.initial_capital * 100),2)}%){bcolors.ENDC}'
            print_bag += f'\n{bcolors.FAIL}Buying Power: {np.round(self.buying_power, 2)}{bcolors.ENDC}'            

        if(date.day == 28):
            self.log(print_bag, True)
        else: 
            self.log(print_bag, False)

    def create_summary(self, print_results = False):
        """
        create summary
        """
        if print_results:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'SELL PRICE', 'NET GAIN', 'K', 'D', 'POTENTIAL'))

        for values in self.history:
            stock = values[0]
            buy_price = values[1]
            n_shares = values[2]
            sell_price = values[3]
            buy_date = values[4]
            sell_date = values[5]
            k = values[6]
            d = values[7]
            potential = values[8]

            net_gain = (sell_price - buy_price) * n_shares
            gain_perc = np.round((sell_price - buy_price) / buy_price * 100, 2)
            total_days_on_market = stock_utils.get_market_days(buy_date, sell_date)            
            
            self.total_gain += net_gain
            """
            self.history_df = self.history_df.append({'stock': stock, 'buy_price': buy_price, 'n_shares': n_shares, \
                'sell_price': sell_price, 'net_gain': net_gain, 'gain_perc': gain_perc, 'buy_date': buy_date, \
                'sell_date': sell_date, 'total_days_on_market': total_days_on_market, 'cup_len': cup_len, \
                'handle_len': handle_len, 'cup_depth': cup_depth, 'handle_depth': handle_depth, 'rrr': rrr}, ignore_index = True)
            """
            self.history_df = pd.concat([self.history_df, pd.DataFrame({
                'stock': [stock],
                'buy_price': [buy_price],
                'n_shares': [n_shares],
                'sell_price': [sell_price],
                'net_gain': [net_gain],
                'gain_perc': [gain_perc],
                'buy_date': [buy_date],
                'sell_date': [sell_date],
                'total_days_on_market': [total_days_on_market],
                'k': [k],
                'd': [d],
                'potential': [potential]
            })])

            if print_results:
                print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}" \
                      .format(stock, buy_price, n_shares, sell_price, np.round(net_gain, 2), k, d, potential))

    

    def print_summary(self):
        """
        prints the summary of results
        """
        self.create_summary(print_results= True)
        summary = f"""
Initial Balance: {self.initial_capital:.2f}
Final Balance: {(self.initial_capital + self.total_gain):.2f}
Total Gain: {self.total_gain:.2f}
P/L: {(self.total_gain / self.initial_capital) * 100:.2f} %
        """
        self.log(summary)

    def log(self, message, force_to_send_to_telegram = None):
        print(message)
        if self.send_to_telegram if force_to_send_to_telegram is None else force_to_send_to_telegram:
            t = telegram()
            message = message.replace(bcolors.HEADER, "")
            message = message.replace(bcolors.OKBLUE, "")
            message = message.replace(bcolors.OKCYAN, "")
            message = message.replace(bcolors.OKGREEN, "")
            message = message.replace(bcolors.WARNING, "")
            message = message.replace(bcolors.FAIL, "")
            message = message.replace(bcolors.ENDC, "")
            message = message.replace(bcolors.BOLD, "")
            message = message.replace(bcolors.UNDERLINE, "")
            t.send_message(message=message)

    def create_history_csv_header(self):
         with open(os.path.join(self.folder_dir, 'history_df.csv'), 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["STOCK", "BUY PRICE", "SHARES", "SELL PRICE", "NET GAIN", \
                             "GAIN%", "BUY DATE", "SELL DATE", "TOTAL DAYS ON MARKET", \
                                "K", "D", "POTENTIAL"])

    def save_history(self, values):
        stock = values[0]
        buy_price = values[1]
        n_shares = values[2]
        sell_price = values[3]
        buy_date = values[4]
        sell_date = values[5]
        k = values[6]
        d = values[7]
        potential = values[8]

        net_gain = (sell_price - buy_price) * n_shares
        gain_perc = np.round((sell_price - buy_price) / buy_price * 100, 2)
        total_days_on_market = stock_utils.get_market_days(buy_date, sell_date)

        row = [stock, buy_price, n_shares, sell_price, net_gain, gain_perc, buy_date, sell_date, total_days_on_market, \
                k, d, potential]

        with open(os.path.join(self.folder_dir, 'history_df.csv'), 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)