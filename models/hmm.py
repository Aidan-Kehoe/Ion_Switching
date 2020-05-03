import sys
sys.path.append('..')
from pomegranate import *
import pandas as pd 
import numpy as np 
from load_data import *

class HMM_Classifier(object):

	def __init__(self,
				 train,
				 test,
				 batch_size=4000):
		self.train = train
		self.test = test
		self.batch_size = batch_size

	def generate_batches(self,training=True):
		if training:
			train = self.train
		batch_size = self.batch_size
		train['group'] = train.groupby(train.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
		train.loc[:,'group'].astype(np.uint16)
		data = train.groupby('group')['signal'].apply(list).reset_index(name='sequence').sequence.values.tolist()
		labels = train.groupby('group')['open_channels'].apply(list).reset_index(name='target').target.values.tolist()
		#Do you have to add start and end-None?
		data = data[:5]
		labels = labels[:5]
		return data, labels

	def fit_hmm(self, data, labels, no_states=11):
		model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution,
			n_components = no_states,
			X = data,
			labels = labels,
			algorithm = 'labeled')
		model.bake()
		model.fit(data, labels=labels, algorithm = 'labeled', n_jobs = 4)
		return model



data_loader = DataLoader("../data/train.csv","../data/test.csv")
train, test = data_loader.get_val_data()
hmm = HMM_Classifier(train, test)
data, labels = hmm.generate_batches()
print('HMM_Classifier')
hmm.fit_hmm(data, labels)
print('hi')
print(model)