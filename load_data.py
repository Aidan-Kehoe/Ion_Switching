import pandas as pd
import numpy as np
import pywt


SAMPLE_RATE = 25
SIGNAL_LEN = 1000


class DataLoader(object):

	def __init__(self, train_path, test_path, periods=None):
		self.train_path = train_path
		self.test_path = test_path
		self.test_data = pd.read_csv(self.test_path)
		self.train_data = pd.read_csv(self.train_path)
		self.signal_len = SIGNAL_LEN
		self.periods = periods


	def get_val_data(self):
		#no test_data for now, since no open_channels
		return [self.train_data[:4000000], self.train_data[4000000:]]

	def get_test_data(self):
		return self.test_data

	def get_denoised_data(self, signal = None):
		if signal is not None:
			y = self._denoise_signal(signal)
		else:
			y = self._denoise_signal(self.train_data["signal"])
		return y

	def get_smoothed_data(self, signal = None):
		if signal is not None:
			y = self._average_smoothing(signal)
		else:
			y = self._average_smoothing(self.train_data["signal"])
		return y

	def get_signals(self, train_data=None, test_data=None):
		#is this even useful
		if train_data is None:
			train_data = self.train_data
		if test_data is None:
			test_data = self.test_data
		train_signals = []
		train_targets = []
		test_signals = []
		test_targets = []
		#change this to appropriate length based on data length
		for i in range(4000):
			min_lim = self.signal_len * i
			max_lim = self.signal_len * (i + 1)

			train_signals.append(list(train_data["signal"].iloc[min_lim : max_lim]))
			train_targets.append(train_data["open_channels"].iloc[max_lim])
			test_signals.append(list(test_data["signal"].iloc[min_lim : max_lim]))
			test_targets.append(test_data["open_channels"].iloc[max_lim])
	
		train_signals = np.array(train_signals)
		train_targets = np.array(train_targets)
		test_signals = np.array(test_signals)
		test_targets = np.array(test_targets)
		return train_signals, train_targets, test_signals, test_targets

	def get_partitioned_signals(self):
		pass

	def _maddest(self, d, axis = None):
		return np.mean(np.absolute(d - np.mean(d, axis)), axis)

	def _denoise_signal(self, x, wavelet='db4', level=1):
		coeff = pywt.wavedec(x, wavelet, mode="per")
		sigma = (1/0.6745) * self._maddest(coeff[-level])

		uthresh = sigma * np.sqrt(2*np.log(len(x)))
		coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

		return pywt.waverec(coeff, wavelet, mode='per')

	def _average_smoothing(self, signal, kernel_size=3, stride=1):
		sample = []
		start = 0
		end = kernel_size
		while end <= len(signal):
			start = start + stride
			end = end + stride
			sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
		return np.array(sample)

	def shift_data(self, data):
		if self.periods is None:
			periods = [1,2,3]
		else:
			periods = self.periods

		data_transformed = data_transformed.copy()
		for p in periods:
			data_transformed[f"{self.column}_shifted_{p}"] = data_transformed[self.column].shift(
					periods=p, fill_value = 0)
		return data_transformed