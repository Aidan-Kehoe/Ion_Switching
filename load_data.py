import pandas as pd
import numpy as np
import pywt


SAMPLE_RATE = 25
SIGNAL_LEN = 1000
TEST_PATH = "../data/test.csv"
TRAIN_PATH = "../data/train.csv"
SUBMISSION_PATH = "../data/sample_submission.csv"


class DataLoader(object):

	def __init__(self):
		self.test_data = pd.read_csv(TEST_PATH)
		self.train_data = pd.read_csv(TRAIN_PATH)
		self.signal_len = SIGNAL_LEN

	def get_data(self):
		#no test_data for now, since no open_channels
		return [self.train_data[:4000002], self.train_data[4000002:]]

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

	def get_signals(self):
		train_signals = []
		train_targets = []
		test_signals = []
		test_targets = []
		#change this to appropriate length based on data length
		for i in range(4000):
			min_lim = self.signal_len * i
			max_lim = self.signal_len * (i + 1)

			train_signals.append(list(self.train_data["signal"][min_lim : max_lim]))
			train_targets.append(self.train_data["open_channels"][max_lim])
			test_signals.append(list(self.test_data["signal"][min_lim : max_lim]))
			test_targets.append(self.test_data["open_channels"][max_lim])
	
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