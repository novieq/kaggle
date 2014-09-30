import csv as csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

import random
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


data = pd.read_csv('train.csv', header=0)
print data.shape
test = pd.read_csv('test.csv', header=0)
preds = pd.read_csv('svm_benchmark.csv', header=0)
print preds.columns

#random.seed(1)
#rows = random.sample(data.index, len(data.index)/10)
#data_10 = data.ix[rows]
#train = data.drop(rows)
#print train.shape
train = data
#train.to_csv('training_90.csv', index = False)
#data_10.to_csv('training_10_cv.csv', index = False)

#data_10 = pd.read_csv('training_10_cv.csv')
#train = pd.read_csv('training_90.csv')
#train = pd.read_csv('training.csv')


#preds = np.zeros((test.shape[0], 2))
#preds_10 = np.zeros((data_10.shape[0],2))


xtrain, ytrain, xtest = np.array(train)[:,1:], np.array(train)[:,0], np.array(test)[:,:]
print 'train' , np.array(train).shape
print 'xtrain' ,xtrain.shape
print 'ytrain' , ytrain.shape
print 'test' , xtest.shape
sup_vec = svm.SVC(C=11000, verbose = 2, probability=True)
#sup_vec = RandomForestRegressor(n_estimators=100)
sup_vec.fit(xtrain, ytrain)
print preds.head(15)
#print sup_vec.predict(xtest).astype(float)
#preds['PredictedProbability'] = sup_vec.predict(xtest).astype(float)

preds['PredictedProbability'] = sup_vec.predict_proba(xtest).astype(float)
#preds_10[:,1] = sup_vec.predict(data_10).asfloat()
print preds.head(15)

preds.to_csv('/Users/sayghosh/code/kaggle/predictingbiologicalresponse/predictions.csv', index = False);
#predictions_file = open('/Users/sayghosh/code/kaggle/predictingbiologicalresponse/predictions.csv', "wb")
#open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["MoleculeId","PredictedProbability"])
#open_file_object.writerows(zip(preds[:,0], preds[:,1]))
#predictions_file.close()