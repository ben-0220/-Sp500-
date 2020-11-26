from collecter import DataCollecter
from plot_candle import Plot
from preprocessing import Indicator
import pandas as pd

url = 'https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC'
filepath = './SP500.csv'

#----  Data Collect  ----
Collecter = DataCollecter(url, filepath)

data = Collecter.load_data_from_fred('SP500')
histroy = Collecter.load_history_from_yahoo('^GSPC')[['Open', 'Close', 'Low', 'High', 'Volume']]

#----  Indicator ----
indicator = Indicator(histroy[:])

macd, macd_signal, macd_histogram = indicator.MACD(fast=12, slow=26, signalperiod=9)
rsi = indicator.RSI(timeperiod=14)
ma = indicator.MA(timeperiod=14, matype=0)
b_high, b_mid, b_low = indicator.BBands()
mfi = indicator.MFI(timeperiod=14)

trend = indicator.Trend(timeperiod=14)
slope = indicator.Slope(timeperiod=14)

histroy['MACD'] = macd
histroy['MACD_SIG'] = macd_signal
histroy['MACD_HIST'] = macd_histogram
histroy['RSI'] = rsi
histroy['MA'] = ma
histroy['Slope'] = slope
histroy['BBAND_HIGH'] = b_high
histroy['BBAND_MID'] = b_mid
histroy['BBAND_LOW'] = b_low
histroy['high_delay']= histroy[7:]["High"]
histroy['low_delay']= histroy[7:]["Low"]
histroy['MFI'] = mfi
histroy['Trend'] = trend
histroy['delay']= histroy[7:]["Close"]





histroy.to_csv(filepath, encoding='utf-8')

#----  Plot  ----
#Plot.plot_candles(histroy[:], technicals=[rsi], technicals_titles=['RSI'])
