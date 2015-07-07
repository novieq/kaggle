'''
Created on Oct 27, 2014

@author: sayghosh
'''
#Modified from Dmitry Dryomov (YSDA) code from Tradeshift Classification Challenge in kaggle
from datetime import datetime
import os
import pandas as pd
data_dir=os.getcwd() + '/'#/Users/sayghosh/code/kaggle/tradeshift/'
start = datetime.now()
print 'Reading training data'
train = pd.read_csv(data_dir + 'train')

print 'Training data shape', train.shape
print("\nTime! \n\n\t %s"%str(datetime.now()-start))

sample_size = 1700000
ratio = train.shape[0] / sample_size

train_sample = train[
    [hash(id) % ratio == 0 for id in train['id']]
]

train_sample.shape

train_sample.to_csv(data_dir + 'train_sample', index = False)

del train

import pandas as pd

train_sample = pd.read_csv(data_dir + 'train_sample')

labels = pd.read_csv(data_dir + 'trainLabels')

labels.columns

train_with_labels = pd.merge(train_sample, labels, on = 'id')

train_with_labels.shape

from collections import Counter

#Counter([name[0] for name in train_with_labels.columns])

del labels
del train_sample

print 'Reading test data'
test = pd.read_csv(data_dir + 'test')
print test.shape
print("\Time! \n\n\t %s"%str(datetime.now()-start))

from sklearn.feature_extraction import DictVectorizer
import numpy as np
import scipy as scp

X_numerical = []
X_test_numerical = []

vec = DictVectorizer()

names_categorical = []

train_with_labels.replace('YES', 1, inplace = True)
train_with_labels.replace('NO', 0, inplace = True)
#train_with_labels.replace('nan', np.NaN, inplace = True)

test.replace('YES', 1, inplace = True)
test.replace('NO', 0, inplace = True)
#test.replace('nan', np.NaN, inplace = True)

#Separate out the numerical and the categorical features. The numerical features go to X_numerical and the categorical features go to X_sparse using dictvectorizer
for name in train_with_labels.columns :    
    if name.startswith('x') :
        #map(function, iterable..)
        #Apply function to every item of iterable and return a list of the results.
        column_type, _ = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key = lambda x: x[1])
        print name, column_type
        # LOL expression
        if column_type == str(str) :
            train_with_labels[name] = map(str, train_with_labels[name])
            test[name] = map(str, test[name])

            names_categorical.append(name)
            print 'Categorical Variable', name, len(np.unique(train_with_labels[name]))
        else :
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_numerical.append(test[name].fillna(-999))
        print("Time! \t %s"%str(datetime.now()-start))

X_numerical = np.column_stack(X_numerical)
X_test_numerical = np.column_stack(X_test_numerical)


print 'Names of Categorical Variables', names_categorical
#When feature values are strings, this transformer will do a binary one-hot (aka one-of-K) coding: one boolean-valued feature is constructed for each of the possible string values that the feature can take on.
print("[Feature generation for categorical variables] \t %s"%str(datetime.now()-start)) 
X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())
print("Time! \t %s"%str(datetime.now()-start))


print 'X_numerical.shape',X_numerical.shape, 'X_sparse.shape',X_sparse.shape, 'X_test_numerical.shape',X_test_numerical.shape, 'X_test_sparse.shape',X_test_sparse.shape


from sklearn.externals import joblib

joblib.dump(
    (X_numerical, X_sparse, X_test_numerical, X_test_sparse),
    data_dir + 'X.dump',
    compress = 1,
)


from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

log_loss_scorer = make_scorer(log_loss, needs_proba = True)

y_columns = [name for name in train_with_labels.columns if name.startswith('y')]

X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(
    X_numerical, 
    X_sparse, 
    train_with_labels[y_columns].values,
    test_size = 0.5
)

X_meta = [] 
X_test_meta = []
del train_with_labels
print "Build Ensembles"

for i in range(y_base.shape[1]) :
    print i
    print("\nTime! \n\n\t %s"%str(datetime.now()-start))
    y = y_base[:, i]
    if len(np.unique(y)) == 2 : 
        rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, verbose=2)
        rf.fit(X_numerical_base, y) # Fit on 50% of the training data
        X_meta.append(rf.predict_proba(X_numerical_meta)) # Predict on rest 50% of the training data where we have some predictions
        X_test_meta.append(rf.predict_proba(X_test_numerical)) # Predict the meta scores for all the test data

        svm = LinearSVC(C=5000)
        svm.fit(X_sparse_base, y)
        X_meta.append(svm.decision_function(X_sparse_meta))
        X_test_meta.append(svm.decision_function(X_test_sparse))
        
#column_stack will convert 1D array into 2D arrays
X_meta = np.column_stack(X_meta)
X_test_meta = np.column_stack(X_test_meta)

# <codecell>

print 'X_meta', X_meta.shape, 'X_test_meta', X_test_meta.shape



p_test = []
print("\nStarting to predict using final meta model! \n\n\t %s"%str(datetime.now()-start))

for i in range(y_base.shape[1]) :
    y = y_meta[:, i]

    constant = Counter(y)
    constant = constant[0] < 4 or constant[1] < 4
    
    predicted = None
    
    if constant :
        # Best constant
        constant_pred = np.mean(list(y_base[:, i]) + list(y_meta[:, i]))
        
        predicted = np.ones(X_test_meta.shape[0]) * constant_pred
        print "%d is constant like: %f" % (i, constant_pred)
    else :
  #      AdaBoostRegressor(ExtraTreesRegressor(n_estimators=100, verbose=2, n_jobs=-1),n_estimators=160, loss='exponential')

        rf = RandomForestClassifier(n_estimators=100, n_jobs = -1, verbose=2)
        print X_meta.shape, X_numerical_meta.shape, y.shape, i
        rf.fit(X_meta, y)
    #    rf.fit(scp.sparse.hstack([X_meta, X_numerical_meta, X_sparse_meta]), y)
        #predicted = rf.predict_proba(scp.sparse.hstack([X_test_meta, X_test_numerical, X_test_sparse]))

        predicted = rf.predict_proba(X_test_meta)

        predicted = predicted[:, 1]
        
        #scores = cross_val_score(rf, np.hstack([X_meta, X_numerical_meta]), y, cv = 4, n_jobs = 1, scoring = log_loss_scorer)
#        print i, 'RF log-loss: %.4f +- %.4f, mean = %.6f' %(np.mean(scores), np.std(scores), np.mean(predicted))

    #Add each column's predictions into p_test as an array and then convert it into an n-darray
    p_test.append(
        predicted
    )
    
p_test = np.column_stack(p_test)
print("\nPredictions Calculated. Printing..! \n\n\t %s"%str(datetime.now()-start))



print 'Predictions Shape', p_test.shape

# <codecell>

import gzip

def save_predictions(name, ids, predictions) :
    out = gzip.open(name, 'w')
    print >>out, 'id_label,pred'
    for id, id_predictions in zip(test['id'], p_test) :
        for y_id, pred in enumerate(id_predictions) :
            if pred == 0 or pred == 1 :
                pred = str(int(pred))
            else :
                pred = '%.6f' % pred
            print >>out, '%d_y%d,%s' % (id, y_id + 1, pred)

# <codecell>

save_predictions('quick_start.csv.gz', test['id'].values, p_test)
print("\nAll Done! \n\n\t %s"%str(datetime.now()-start))

# <codecell>

