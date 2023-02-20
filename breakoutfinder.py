# All Imports
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

# Facebook Historial Data
stock = yf.Ticker("0001.HK")
df = stock.history(period="1y")


plt.style.use('fivethirtyeight')
df['sma20'] = df.ta.sma(length=10)
df['ema20'] = df.ta.ema(length=10)
df[['Close','sma20', 'ema20']].plot(figsize=(8,8))
plt.show()