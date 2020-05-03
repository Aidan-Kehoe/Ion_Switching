import sys
sys.path.append('..')

import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import StandardScaler
from statistics import *
from load_data import DataLoader
import scipy

TEST_PATH = "../data/test.csv"
TRAIN_PATH = "../data/train.csv"
SUBMISSION_PATH = "../data/sample_submission.csv"

class EnsembleClassifier(object):

	def __init__(self,
				feature_set = None,
				 meta_classifier = None,
				 meta_params = None):

		self.feature_set = DataLoader().get_data()
		self.meta_classifier = meta_classifier
		self.meta_params = meta_params
		self.dense_data = None
		self.train_targets= None
		self.test_targets = None
		self.clfs = []
		self.sub_data = pd.DataFrame()
		self.kf = KFold(n_splits = 3, random_state = None, shuffle = False)

	def reset(self):
		self.sub_data = pd.DataFrame()


	def load_data(self, feature_set = None, CrossVal = False):
		if feature_set is not None:
			self.feature_set = feature_set

		X_train = self.feature_set[0]['signal']
		X_test = self.feature_set[1]['signal']
		y_train = self.feature_set[0]["open_channels"]
		y_test = self.feature_set[1]["open_channels"]
		self.feature_set= [X_train, X_test]
		self.train_label = y_train
		self.test_label	= y_test

	def add(self, classifier, classifier_type = None, data_type = None, params = None):
		if classifier_type == 'meta':
			self.meta_classifier = classifier
		else:
			self.clfs.append(classifier)
		if data_type is not None:
			self.data_type.append(data_type)


	def train_learners(self, proba):
		for i, clf in enumerate(self.clfs):
			y = self.train_label
			X = self.feature_set[0]
			clf.fit(X, y)

			pred_list_a = []
			pred_list_b = []
			for train_index, test_index in self.kf.split(X):
				#WILL THIS SHUFFLE THE TIME SERIES DATA??
				X_train_fold, X_test_fold = X[train_index], X[test_index]
				y_train_fold, y_test_fold = y[train_index], y[test_index]
				clf.fit(X_train_fold, y_train_fold)
				if proba:
					pred_list_a.extend([row[0] for row in clf.predict_proba(X_test_fold)])
					pred_list_b.extend([row[1] for row in clf.predict_proba(X_test_fold)])
				else:
					pred_list_a.extend(clf.predict(X_test_fold))
			if proba:
				self.sub_data['predictions_a_' + str(i)] = pred_list_a
				self.sub_data['predictions_b_' + str(i)] = pred_list_b
			else:
				self.sub_data['predictions_' + str(i)] = pred_list_a
		return self.sub_data


	def train_meta(self, proba):
		self.meta_classifier.fit(self.sub_data, self.train_label)
		return self.meta_classifier

	def fit(self, proba = True):
		self.load_data()
		self.train_learners(proba)
		self.train_meta(proba)


	def predict(self, proba = True):
		self.sub_data = pd.DataFrame()
		y = self.test_label

		for i, clf in enumerate(self.clfs):
			X = self.feature_set[1]

			if proba:
				self.sub_data['predictions_a' + str(i)] = [row[0] for row in clf.predict_proba(X)]
				self.sub_data['predictions_b' + str(i)] = [row[1] for row in clf.predict_proba(X)]
			else:
				self.sub_data['predictions_' + str(i)] = clf.predict(X)

		#return self.meta_classifier.score(self.sub_data, y)
		return self.meta_classifier.predict(self.sub_data)
	'''
	def run_cross_val(self, file = 'data/processed/datasets/data_preprocessed_final.csv', proba = True):
		df = pd.read_csv(file)[:10000]
		df = df.dropna()
		#df = df.reset_index(["Unnamed: 0", "id"]).drop(["Unnamed: 0"], axis = 1)
		k_fold = KFold(n_splits = 3, random_state = 22)
		scores = []
		for train_index, test_index in k_fold.split(df):
			self.reset()
			kg = TextFeatures(df)
			train_features = kg.get_all_features(df.iloc[train_index])
			test_features = kg.get_all_features(df.iloc[test_index])
			text, _, _ = kg.get_tfidf(df)
			train_text = text[train_index]
			test_text = text[test_index]
			self.load_data(dense_features = [train_features, test_features],
						   sparse_data = [train_text, test_text], CrossVal = True)
			self.train_learners(proba)
			self.train_meta(proba)
			scores.append(self.predict(proba))
		return mean(scores)

		#if feature_file is None:
		#	kg = TestFeatures(df)
		#	features = kg.get_all_features()
	'''
	def backprop_weights(self):
		meta_weights = self.meta_classifier.coef_[0]
		sub_weights = []
		for clf in self.clfs:
			sub_weights.append(clf.coef_)

		sub_weights = [met_weight * weights for (weights, met_weight) in zip(sub_weights, meta_weights)]
		sum_weights = self.dense_data[1] @ sub_weights[0].T + self.sparse_data[1] @ sub_weights[1].T
		sig = scipy.special.expit(sum_weights)
		sig = (sig > 0.5).astype(int)
		return sig, sub_weights

	def get_feature_names(self):
		#IGNORE THIS FOR NOW, DEFINITELY BUGGY
		feature_df_pos = pd.DataFrame()
		feature_df_neg = pd.DataFrame()
		feature_df_dense_pos = pd.DataFrame()
		feature_df_dense_neg = pd.DataFrame()
		feature_names = self.countvect.get_feature_names()
		_, weights = self.backprop_weights()
		print(sum(weights[0][0]) + sum(weights[1][0]))
		for i, class_label in enumerate([0,1]):
			if class_label == 1:
				index = np.argsort(weights[1][0])[::-1]
				class_ = sorted(weights[1][0], reverse = True)
				index_dense = np.argsort(weights[0][0])[::-1]
				dense_class = sorted(weights[0][0], reverse = True)
				feature_df_pos['feature_names'] = [feature_names[j] for j in index]
				feature_df_pos['class1'] = np.abs(class_)
				feature_df_dense_pos['feature_names'] = [self.dense_names[j] for j in index_dense]
				feature_df_dense_pos['class1'] = np.abs(dense_class)
			else:
				index = np.argsort(-weights[1][0])[::-1]
				class_ = sorted(-weights[1][0], reverse = True)
				index_dense = np.argsort(-weights[0][0])[::-1]
				dense_class = sorted(-weights[0][0], reverse = True)
				feature_df_neg['feature_names'] = [feature_names[j] for j in index]
				feature_df_neg['class1'] = np.abs(class_)
				feature_df_dense_neg['feature_names'] = [self.dense_names[j] for j in index_dense]
				feature_df_dense_neg['class1'] = np.abs(dense_class)
		
			
		return feature_df_pos, feature_df_dense_pos, feature_df_neg, feature_df_dense_neg