import csv as csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from numpy import argsort

import random
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
#preds = pd.read_csv('svm_benchmark.csv', header=0)
#print preds.columns

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


xtrain, ytrain, xtest = np.array(train)[:,:20], np.array(train)[:,20], np.array(test)[:,:20]
print 'train' , np.array(train).shape
print xtrain[1]
print 'xtrain' ,xtrain.shape
print 'ytrain' , ytrain.shape
print 'test' , xtest.shape
estimators=1
#sup_vec = svm.SVC(C=11000, verbose = 2, probability=True)
sup_vec = RandomForestRegressor(n_estimators=estimators, verbose=2)
sup_vec.fit(xtrain, ytrain)
#print preds.head(15)
#print sup_vec.predict(xtest).astype(float)
#preds['PredictedProbability'] = sup_vec.predict(xtest).astype(float)

preds = sup_vec.predict(xtest).astype(float)
#preds_10[:,1] = sup_vec.predict(data_10).asfloat()
#output = zip(test[:,19],preds)
print preds
index = preds.argsort()[-5000:][::-1]

print len(preds)
predictions_index = open('/Users/sayghosh/code/kaggle/eap/predictions.csv', "wb")

predictions_index.writelines(["%s\n" % item  for item in index])
predictions_file = open('/Users/sayghosh/code/kaggle/eap/predictions_ctr.csv', "wb")

predictions_file.writelines(["%s\n" % item  for item in preds])

#predictions_file = open('/Users/sayghosh/code/kaggle/predictingbiologicalresponse/predictions.csv', "wb")
#open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["MoleculeId","PredictedProbability"])
#open_file_object.writerows(zip(preds[:,0], preds[:,1]))
#predictions_file.close()