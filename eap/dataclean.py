from math import sqrt
import random

from numpy import argsort
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import csv as csv
import numpy as np
import pandas as pd
import utilities as util
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('/Users/sayghosh/code/kaggle/eap/quizz.train', delimiter='\t')
cleandata = pd.read_csv('/Users/sayghosh/code/kaggle/eap/quizz.train.clean', delimiter='\t')

print data.shape
print data.describe()
print cleandata.shape
print cleandata.describe()

plt.figure()
#cleandata.plot(subplots=True)
#cleandata.hist() #Gives the histograms of all the columns. For every column, except click the histogram has a bar on the far left indicating presence of outliers
cleandata.boxplot()
plt.show()

