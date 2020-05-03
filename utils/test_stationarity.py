import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA 
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from load_data import DataLoader
import math

def stat_check(timeseries, window = 50):
	batch_len = 500000
	timeseries = timeseries.set_index('time')['signal']
	for i in range(int(len(timeseries)/500000) - 1):
		batch = timeseries.iloc[batch_len*i:batch_len*(i+1)]
		#temporary measure
		#batch = batch.iloc[:50000]
		rollmean = batch.rolling(window).mean()
		rollstd = batch.rolling(window).std()

		#print('Results of Dickey Fuller Test')
		#adft = adfuller(batch)
		#output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
		#for key,values in adft[4].items():
		#	output['critical value (%s)'%key] =  values
		#p-value greater than 0.05 means non-stationary
		#print(output)
		plt.figure()
		plt.plot(batch, color = 'blue', label = 'original')
		plt.plot(rollmean, color = 'red', label = 'Rolling Mean')
		plt.plot(rollstd, color = 'black', label = 'Rolling Std')
		plt.legend(loc='best')
		plt.title('Rolling Mean and Standard Deviation')
	plt.show(block = False)

def remove_trend_seasonal(timeseries, model = 'multiplc')