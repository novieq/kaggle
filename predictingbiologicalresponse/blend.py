from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_submission = load_data.load()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    number_of_estimators = 5
    number_of_jobs=1
    clfs = [RandomForestClassifier(n_estimators=number_of_estimators, n_jobs=number_of_jobs, criterion='gini'),
            RandomForestClassifier(n_estimators=number_of_estimators, n_jobs=number_of_jobs, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=number_of_estimators, n_jobs=number_of_jobs, criterion='gini'),
            ExtraTreesClassifier(n_estimators=number_of_estimators, n_jobs=number_of_jobs, criterion='entropy')]

    print "Creating train and test sets for blending."
    print 'X', X.shape, 'X_submission', X_submission.shape
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

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    preds = pd.read_csv('svm_benchmark.csv', header=0)
    print preds.columns
    preds['PredictedProbability'] = y_submission
    print "Saving Results."
    #np.savetxt(fname='/Users/sayghosh/code/kaggle/predictingbiologicalresponse/submissions.csv', X=y_submission, fmt='%0.9f')
    preds.to_csv('/Users/sayghosh/code/kaggle/predictingbiologicalresponse/predictions.csv', index = False);

    
