"""
__Author__:     Triskelion <info@mlwave.com>
Description:     Starter benchmark code to create a Random Forest 
                submission for a Text Classification competition.
                Can save datasets as vectorized CSV to use in R.
                Made to fit on 8GB RAM laptop. Downgrade/upgrade at will.
                
Platform:         Kaggle <https://kaggle.com>
Organizer:         Tradeshift <https://tradeshift.com/>
Contest title:     Tradeshift Text Classification
Contest URL:    https://www.kaggle.com/c/tradeshift-text-classification/
Data URL:        https://www.kaggle.com/c/tradeshift-text-classification/data
Code Licensed:    http://opensource.org/licenses/MIT
"""

## Parameters & Settings ####################################################

# Data locations
loc_train = "/Users/sayghosh/code/kaggle/tradeshift/train"
loc_test = "/Users/sayghosh/code/kaggle/tradeshift/test"
loc_labels = "/Users/sayghosh/code/kaggle/tradeshift/trainLabels"

# Will be created
loc_test_reduced = "/Users/sayghosh/code/kaggle/tradeshift/test.vec.csv"
loc_train_reduced = "/Users/sayghosh/code/kaggle/tradeshift/train.vec.csv.1000l"
loc_kaggle_submission = "/Users/sayghosh/code/kaggle/tradeshift/kaggle.tradeshift.RF16.1m.csv"

# General
random_state = 1414707707
verbosity = 2

# Mah Poor Memory!
max_train_lines_read = 200 # set 0 to read all
nr_estimators_rf = 5
n_cpu_jobs = -1 # set to -1 for all cores, but know that pooling is costly

## Imports / Dependencies ###################################################

from collections import defaultdict
from datetime import datetime
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import svm
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression


## Functions ################################################################

def reduce_data(loc_csv, loc_train, loc_out):
    """
        Clean, reduce and vectorize dataset, save as CSV
        
        Input:     loc_csv (location to dataset)
                loc_train (location to train dataset)
                loc_out (location of to-be-saved CSV file)
        Output:    A vectorized dataset in CSV format
    """
    # We want to turn categorical features into count features
    # because one-hot-encoding would curse us with dimensionality
    # rendering tree-based models unmanageable.
    # We store the counts from the train set for every unique hash
    # inside a dictionary, and use this to replace hashes with their
    # count occurrence.

    print("Vectorizing:\n\t %s \nSaving as CSV data:\n\t %s"%(loc_csv,loc_out))
    
    start = datetime.now()
    
    hash_counter = defaultdict(float)
    
    for e, line in enumerate( open(loc_train, "rb") ):
        if e > 0: #skip header
            row = line.strip().split(",") # Poor man's CSV reader
            for k in row:
                if "=" in k: #contains hash
                    hash_counter[k] += 1
    
    # open file handler for the output CSV file
    with open( loc_out, "wb") as outfile:
        # Now vectorize the dataset
        for e, line in enumerate( open(loc_csv, "rb") ):
            sample = []
            row = line.strip().split(",")
            if e == 0:
                # Write headers, skip the ID (first column)
                outfile.write(",".join(row[1:])+"\n")
            else:
                # For every value in row, skip the ID
                for k in row[1:]: 
                    # Vectorize hash values to counts
                    if "=" in k:
                        if k in hash_counter:
                            sample.append(hash_counter[k])
                        else:
                            sample.append(0.0)
                    # Vectorize YES/NO values to 0-1 binary
                    elif k == "YES":
                        sample.append(1.0)
                    elif k == "NO":
                        sample.append(0.0)
                    # Fill NaN/Empty values with -999
                    elif k == "":
                        sample.append(-999.0)
                    # Keep numerical values as is
                    else:
                        sample.append(float(k))
                # Write features to file separated by comma
                outfile.write("%s\n"%",".join([str(f) for f in sample]))
                if e % 100000 == 0:
                    print("%s\t%s"%(e,str(datetime.now()-start)))

    print("Done reducing in:\n\t %s\n"%str(datetime.now()-start))

def load_data(loc_csv, nrows=0):
    """
        Loads the data. If nrows=0 then load all, else load first n rows.
    """
    print("Loading data at %s shaped:"%loc_csv)
    if nrows > 0:
        df = pd.read_csv(loc_csv, nrows=nrows)
    else:
        df = pd.read_csv(loc_csv)
    print(df.shape)
    return df

## Main script ################################################################
    
if __name__ == "__main__":
    start = datetime.now()
    
    #Run at least once the reduce_data on train and test
#    reduce_data(loc_train, loc_train, loc_train_reduced)
#    reduce_data(loc_test, loc_train, loc_test_reduced)
    
    X = load_data(loc_train_reduced, nrows=max_train_lines_read)
    y = load_data(loc_labels, nrows=max_train_lines_read)
    y = y.drop("id", axis=1) # dont need ID in labels
    X_submission = load_data(loc_test_reduced)

    
    n_folds=5
    
    print n_folds
    skf = list(StratifiedKFold(y[0], n_folds))
    print skf
    
    clfs = [RandomForestClassifier(n_estimators=nr_estimators_rf, n_jobs=n_cpu_jobs, random_state=random_state, verbose=verbosity), 
            ExtraTreesClassifier(n_estimators=nr_estimators_rf, n_jobs=n_cpu_jobs, random_state=random_state, verbose=verbosity), 
      #      GradientBoostingClassifier(n_estimators=nr_estimators_rf, n_jobs=n_cpu_jobs, random_state=random_state, verbose=verbosity), 
       #     svm.SVC(C=5000.0, verbose = 2, shrinking = False)
       ]
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    #We don't need no education
    del X
    
    # Predicting and writing Kaggle submission
    print("\nWriting Kaggle submission to %s"%loc_kaggle_submission)
    
    with open( loc_kaggle_submission, "wb" ) as outfile:
        outfile.write("id_label,pred\n")
        for e, pred in enumerate( y_submission ):
            for n, p in enumerate( pred ):
                outfile.write("%s_y%s,%s\n"%(e+1700001,n+1,p))
    
    print("\nAll done! \n\n\t %s"%str(datetime.now()-start))
    
    
    
    