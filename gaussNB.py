from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


class GNB(object):
	def __init__(self, data, train_col, test_col, normalize=False, test_size=0.2):
		self.data = data
		self.train_col = train_col
		self.test_col = test_col
		self.test_size = test_size

		self.labeled_Close = None
		self.labeled_trend = None

		self.Close_price = self.data[test_col]
		self.MACD = self.data[["MACD"]]
		self.MACD_SIG = self.data[["MACD_SIG"]]
		self.RSI = self.data[["RSI"]]
		self.MA = self.data[["MA"]]
		self.CMO = self.data[["CMO"]]
		self.PCT = self.data[["PCT"]]

		self.normalize = normalize

		self.bnb = None

		self.preprocessing()

	def preprocessing(self):
		#self.data = self.data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
		self.Close_price = np.log(self.Close_price)
		self.labeled_Close = self.Close_price.apply(pd.qcut, q=10, labels=False)

		self.MACD = self.MACD.apply(pd.qcut, q=20, labels=False)
		self.MACD_SIG = self.MACD_SIG.apply(pd.qcut, q=20, labels=False)
		self.RSI = self.RSI.apply(pd.qcut, q=20, labels=False)
		
		self.MA = np.log(self.MA)
		self.MA = self.MA.apply(pd.qcut, q=10, labels=False)
		self.CMO = self.CMO.apply(pd.qcut, q=20, labels=False)
		
		self.labeled_trend = self.PCT.copy()
		self.labeled_trend[self.labeled_trend > 0] = 1
		self.labeled_trend[self.labeled_trend <= 0] = 0
		print(self.labeled_trend)
		
	def build_model(self):
		self.bnb = BernoulliNB()

	def train_model(self):
		pass