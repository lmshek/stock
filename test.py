import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, time
import pandas_datareader as web
import pandas_ta as ta
import matplotlib.pyplot as plt
from stock_utils.chart import Chart
import pprint

"""
ticker = '0700.HK'

stock = yf.Ticker(ticker)
hist = stock.history(period="max")


hist[f'bband'] = hist.ta.volatility.bband(hist['Close'])
hist[f'mfi'] = hist.ta.volume.MFIIndicator(hist['High'], hist['Low'], hist['Close'], hist['Volume'])


def recent_bars(df, tf: str = "1y"):
    # All Data: 0, Last Four Years: 0.25, Last Two Years: 0.5, This Year: 1, Last Half Year: 2, Last Quarter: 4
    yearly_divisor = {"all": 0, "10y": 0.1, "5y": 0.2, "4y": 0.25, "3y": 1./3, "2y": 0.5, "1y": 1, "6mo": 2, "3mo": 4}
    yd = yearly_divisor[tf] if tf in yearly_divisor.keys() else 0
    return int(ta.RATE["TRADING_DAYS_PER_YEAR"] / yd) if yd > 0 else df.shape[0]

Chart(hist, style="yahoo", title=ticker, verbose=False,
    last=recent_bars(hist), rpad=10,
    volume=True, midpoint=False, ohlc4=False,
    rsi=False, clr=True, macd=False, zscore=False, squeeze=False, lazybear=False,
    archermas=True, archerobv=False,
    show_nontrading=False, # Intraday use if needed
)
"""

if __name__ == '__main__':
    ticker = '2196.HK'

    stock = yf.Ticker(ticker)
    df = stock.history(period="max")

    # Create your own Custom Strategy
    """
    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
        ta=[
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 200},
            {"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "mfi", "window": 20},
            {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
        ]
    )
    # To run your "Custom Strategy"
    df.ta.strategy(CustomStrategy)
     df['BBL_minus_1days'] = (df['BBL_20_2.0'] - df['Close'] > 0).shift(1)
    df['BBL_minus_2days'] = (df['BBL_20_2.0'] - df['Close'] > 0).shift(2)
    df['BBL_minus_3days'] = (df['BBL_20_2.0'] - df['Close'] > 0).shift(3)


    df['MFI_1_day'] = df.MFI_14.shift(1)
    df['MFI_2_day'] = df.MFI_14.shift(2)
    df['MFI_3_day'] = df.MFI_14.shift(3)
    df['MFI_larger_minus_1days'] = df.MFI_14 > df.MFI_14.shift(1)
    df['MFI_larger_minus_2days'] = df.MFI_14.shift(1) > df.MFI_14.shift(2)
    df['MFI_larger_minus_3days'] = df.MFI_14.shift(2) > df.MFI_14.shift(3)

    """
    df['RSI_14'] = df.ta.rsi(close=df['Close'])
    #df['MFI'] = ta.volume.mfi(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    stoch = ta.momentum.stoch(high=df['High'], low=df['Low'], close=df['Close'], k = 20, d = 5, smooth_k = 5)
    df['STOCHk'] = stoch['STOCHk_20_5_5']
    df['STOCHd'] = stoch['STOCHd_20_5_5']

    df['STOCHk_10days_avg'] = df['STOCHk'].rolling(window=10).mean()
    df['STOCHk_20days_avg'] = df['STOCHk'].rolling(window=20).mean()
    df['STOCHk_30days_avg'] = df['STOCHk'].rolling(window=30).mean()

    macd = ta.momentum.macd(df["Close"])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDh'] = macd['MACDh_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']

    """
    bbands = df.ta.bbands(close=df['Close'], length=20)
    df['BBL_20_2.0'] = bbands['BBL_20_2.0']
    df['BBM_20_2.0'] = bbands['BBM_20_2.0']
    df['BBU_20_2.0'] = bbands['BBU_20_2.0']
    df['BBB_20_2.0'] = bbands['BBB_20_2.0']
    df['BBP_20_2.0'] = bbands['BBP_20_2.0']

    df['RSI_14'] = df.ta.rsi(close=df['Close'])
    df['volume_avg_3_days'] = df.shift(1).rolling(3).Volume.mean()
    df['vloume_than_3days_average'] = df['Volume'] > df.shift(1).rolling(3).Volume.mean()
    df['RSI_14'] = df.ta.rsi(close=df['Close'])
    """

    df['stoch_1'] = df['STOCHk_30days_avg'] < 30
    df['stoch_2'] = df['STOCHk_20days_avg'] > 70
    df['stoch_3'] = np.logical_and(20 < df['STOCHk_30days_avg'], df['STOCHk_30days_avg'] < 80)

    df['rsi_1'] = df['RSI_14'] < 27

    df['condition_3'] = df['STOCHd'].shift(3) > df['STOCHk'].shift(3)
    df['condition_4'] = df['STOCHd'].shift(2) > df['STOCHk'].shift(2)
    df['condition_5'] = df['STOCHk'].shift(1) > df['STOCHd'].shift(1)
    df['condition_6'] = df['STOCHk'] > df['STOCHd']

    #df['condition_7'] = abs(df['MACDs'].shift(3)) < abs(df['MACDs'].shift(2))
    #df['condition_8'] = abs(df['MACDs'].shift(2)) < abs(df['MACDs'].shift(1))
    #df['condition_9'] = abs(df['MACDs'].shift(1)) < abs(df['MACDs'])
    #df['condition_8'] = df['MACD'] < 0
    
    


    df['profit_after_14_days'] = (df['High'].shift(-14) - df['Close']) / df['Close']
     
    """
    target = df['BBP_20_2.0'].shift(1) < 0.5 \
        and df['BBP_20_2.0'] > 0.9 \
        and ( \
            df['STOCHk'].shift(3)[-1] > df['STOCHd'].shift(3)[-1] \
            or df['STOCHk'].shift(2)[-1] > df['STOCHd'].shift(2)[-1] \
            or df['STOCHk'].shift(1)[-1] > df['STOCHd'].shift(1)[-1] \
        )
    """
   
    
    

    df.to_csv(f'{ticker}.csv')