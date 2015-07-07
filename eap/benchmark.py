from sklearn.metrics import mean_squared_error
from math import sqrt

import utilities as util
import csv as csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor

from numpy import argsort

import random
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


data = pd.read_csv('/Users/sayghosh/code/kaggle/eap/quizz.train', delimiter='\t')
print "Number of positives" , sum(data['click']==1) #ctr = 5.874%

print data.shape
test = pd.read_csv('/Users/sayghosh/code/kaggle/eap/quizz.test', delimiter='\t')
print data.columns
print data.shape
print test.columns
print test.shape
print data.describe()

#Subsampling of non-clicks data
clicks = data[data['click']==1.0]
print "Shape of the click data", clicks.shape

nonclicks = data[data['click']==0.0]
print "Shape of the non-click data", nonclicks.shape
rows = random.sample(nonclicks.index, len(nonclicks.index)/6)
nonclicks_5=nonclicks.ix[rows]

data = clicks.append(nonclicks_5,ignore_index=True)
print "New undersampled data set size", data.shape

#Oversampling of clicks
data.append(clicks, ignore_index=True)
data.append(clicks, ignore_index=True)
data.append(clicks, ignore_index=True)
data.append(clicks, ignore_index=True)





#Data preparation for cross validation testing
np.random.seed(0)
rows = random.sample(data.index, len(data.index)/10)
data_10 = data.ix[rows]
#train = data
train = data.drop(rows)
print train.shape

#train.to_csv('training_90.csv', index = False)
#data_10.to_csv('training_10_cv.csv', index = False)

#data_10 = pd.read_csv('training_10_cv.csv')
#train = pd.read_csv('training_90.csv')
#train = pd.read_csv('training.csv')


#preds = np.zeros((test.shape[0], 2))
#preds_10 = np.zeros((data_10.shape[0],2))


xtrain, ytrain, xtest, x_cv, y_cv = np.array(train)[:,:20], np.array(train)[:,20], np.array(test)[:,:20], np.array(data_10)[:,:20], np.array(data_10)[:,20],
print 'train' , np.array(train).shape
print xtrain[1]
print 'xtrain' ,xtrain.shape
print 'ytrain' , ytrain.shape
print 'test' , xtest.shape
estimators=100
#sup_vec = svm.SVC(C=11000, verbose = 2, probability=True)
#sup_vec = RandomForestRegressor(n_estimators=estimators, verbose=2, n_jobs=-1, max_leaf_nodes=100)
#sup_vec = ExtraTreesRegressor(n_estimators=estimators, verbose=2, n_jobs=-1, max_leaf_nodes=100)


#sup_vec =  AdaBoostRegressor(RandomForestRegressor(n_estimators=100, verbose=2, n_jobs = -1),n_estimators=100)
sup_vec =  AdaBoostRegressor(ExtraTreesRegressor(n_estimators=100, verbose=2, n_jobs=-1),n_estimators=160, loss='exponential')

#sup_vec =  AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=300)

#dt_stump = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)
#dt_stump.fit(xtrain, ytrain)
#dt_stump_err = 1.0 - dt_stump.score(xtrain, ytrain)
#n_estimators = 400
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
#learning_rate = 1.

#sup_vec = AdaBoostClassifier(
#    base_estimator=dt_stump,
#    learning_rate=learning_rate,
#    n_estimators=n_estimators,
#    algorithm="SAMME.R")

sup_vec.fit(xtrain, ytrain)
#print preds.head(15)
#print sup_vec.predict(xtest).astype(float)
#preds['PredictedProbability'] = sup_vec.predict(xtest).astype(float)

preds = sup_vec.predict(xtest).astype(float)
preds_10 = sup_vec.predict(x_cv).astype(float)

#Print the output for submission
index = preds.argsort()[-5000:][::-1]

print len(preds)
predictions_index = open('/Users/sayghosh/code/kaggle/eap/predictions.csv', "wb")

predictions_index.writelines(["%s\n" % item  for item in index])
predictions_file = open('/Users/sayghosh/code/kaggle/eap/predictions_ctr.csv', "wb")

predictions_file.writelines(["%s\n" % item  for item in preds])

print sup_vec.feature_importances_
#Print the error metrics
print np.min(preds_10), np.mean(preds_10), np.max(preds_10), np.median(preds_10)
err, cerr = util.mce(preds_10, y_cv, threshold=np.mean(preds_10))
print "Mis classification count" , err
print "Click mis classification", cerr

rms = sqrt(mean_squared_error(y_cv, preds_10))
print "RMS", rms
print "Number of clicks in the cross validation set", sum(y_cv)
print "eap Misclassification Error", util.eapError(preds_10, y_cv)
precision, recall, f1score = util.F1_score(y_cv, preds_10, np.mean(preds_10))
print "Precision(Of all the predicted clicks what fraction actually got clicks)={%f}, Recall(Of clicks what fraction did we actually predict correctly)={%f}, f1score={%f}"%(precision,recall,f1score)
