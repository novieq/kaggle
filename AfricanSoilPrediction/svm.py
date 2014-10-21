import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import preprocessing
from scipy import stats
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression


import random
from sklearn.metrics import mean_squared_error
from math import sqrt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#data = pd.read_csv('training.csv')
test = pd.read_csv('sorted_test.csv')

#random.seed(1)
#rows = random.sample(data.index, len(data.index)/10)
#data_10 = data.ix[rows]
#train = data.drop(rows)
#train.to_csv('training_90.csv', index = False)
#data_10.to_csv('training_10_cv.csv', index = False)

data_10 = pd.read_csv('training_10_cv.csv')
train = pd.read_csv('training_90.csv')
#train = pd.read_csv('training.csv')

train = train[train['P']<5]
train = train[train['SOC']<4]
#train = train[train['Ca']<4]
#train = pd.read_csv('training.csv')

print train.describe()
labels = train[['Ca','P','pH','SOC','Sand']].values
labels_10 = data_10[['Ca','P','pH','SOC','Sand']].values


train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
data_10.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

pickTrain = [ 'BSAN', 'BSAS', 'BSAV', 'CTI', 'ELEV', 'EVI', 'LSTD', 'REF1', 'REF3', 'REF7', 'RELI', 'TMAP', 'TMFI', 'm7280.04', 'm7008.13', 'm6643.64', 'm5521.26', 'm5309.13', 'm5212.7', 'm5058.43', 'm4319.81', 'm3895.55', 'm3698.84', 'm3696.91', 'm3685.34', 'm3675.7', 'm3621.7', 'm3586.99', 'm3529.13', 'm3332.43', 'm3074.01', 'm2896.59', 'm2877.31', 'm2850.31', 'm2537.89', 'm2534.03', 'm2516.68', 'm2505.11', 'm2364.33', 'm2017.2', 'm2003.7', 'm1934.27', 'm1907.28', 'm1872.56', 'm1870.63', 'm1841.71', 'm1816.64', 'm1806.99', 'm1795.42', 'm1689.36', 'm1662.36', 'm1660.43', 'm1645', 'm1637.29', 'm1583.29', 'm1544.72', 'm1521.58', 'm1450.22', 'm1436.72', 'm1432.87', 'm1407.8', 'm1388.51', 'm1361.51', 'm1353.8', 'm1348.01', 'm1330.66', 'm1326.8', 'm1322.94', 'm1321.01', 'm1317.16', 'm1315.23', 'm1226.52', 'm1203.38', 'm1193.73', 'm1176.38', 'm1172.52', 'm1137.81', 'm1068.38', 'm1066.45', 'm1058.74', 'm1031.74', 'm917.961', 'm682.685', 'm632.544', 'm617.116']
rfTrain = train[pickTrain]
rfTest = test[pickTrain]
rfTest_10 = data_10[pickTrain]
#rowCount = 3578
rowCount = 3593
xtrain, xtest, xdata_10, rfTrainR = np.array(train)[:,:rowCount], np.array(test)[:,:rowCount], np.array(data_10)[:,:rowCount], np.array(rfTrain)
n_components=150
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(xtrain)
#xtrain_pca = pca.transform(xtrain)
#xtest_pca = pca.transform(xtest)
#xdata_10_pca = pca.transform(xdata_10)


preds = np.zeros((xtest.shape[0], 5))
preds_10 = np.zeros((xdata_10.shape[0],5))
for i in range(5):
    if(i==4):
        clfs = [RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose = 2),
            ExtraTreesRegressor(n_estimators=100, n_jobs=-1, verbose = 2),
            GradientBoostingRegressor(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50, verbose = 2),
            svm.SVR(C=10.0, verbose = 2, shrinking = False),
            svm.SVR(C=5000.0, verbose = 2, shrinking = False)]
     #   sup_vec =  AdaBoostRegressor(svm.SVR(C=5000.0, verbose = 2, gamma = 2),n_estimators=20, loss='exponential')
        
        dataset_blend_train = np.zeros((xtrain.shape[0], len(clfs)))
        dataset_blend_test = np.zeros((xtest.shape[0], len(clfs)))
        dataset_blend_train_10 = np.zeros((xdata_10.shape[0], len(clfs)))
        
        for j, clf in enumerate(clfs):
            print j, clf
            clf.fit(xtrain, labels[:,i])
            dataset_blend_train[:,j] = clf.predict(xtrain).astype(float)
            dataset_blend_test[:,j] = clf.predict(xtest).astype(float)
            dataset_blend_train_10[:,j] = clf.predict(xdata_10).astype(float)
            
            
        sup_vec = svm.SVR(C=100.0, verbose = 2, shrinking = False)
#        min = labels[:,i].min()
#        alpha = .00000001
#        yy = (np.log(labels[:,i]-labels[:,i].min()+alpha))
        sup_vec.fit(dataset_blend_train, labels[:,i])
        preds[:,i] = sup_vec.predict(dataset_blend_test).astype(float)
   #     preds[:,i] = np.exp(preds[:,i]-alpha + min)
        preds_10[:,i] = sup_vec.predict(dataset_blend_train_10).astype(float)
    elif(i==1): #P
        clfs = [#RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose = 2),
            #ExtraTreesRegressor(n_estimators=10, n_jobs=-1, verbose = 2),
            #GradientBoostingRegressor(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50, verbose = 2),
            svm.SVR(C=10.0, verbose = 2, shrinking = False),
            svm.SVR(C=5000.0, verbose = 2, shrinking = False)]
     #   sup_vec =  AdaBoostRegressor(svm.SVR(C=5000.0, verbose = 2, gamma = 2),n_estimators=20, loss='exponential')
        
        dataset_blend_train = np.zeros((xtrain.shape[0], len(clfs)))
        dataset_blend_test = np.zeros((xtest.shape[0], len(clfs)))
        dataset_blend_train_10 = np.zeros((xdata_10.shape[0], len(clfs)))
        
        for j, clf in enumerate(clfs):
            print j, clf
            clf.fit(xtrain, labels[:,i])
            dataset_blend_train[:,j] = clf.predict(xtrain).astype(float)
            dataset_blend_test[:,j] = clf.predict(xtest).astype(float)
            dataset_blend_train_10[:,j] = clf.predict(xdata_10).astype(float)
            
            
        sup_vec = svm.SVR(C=100.0, verbose = 2, shrinking = False)
#        min = labels[:,i].min()
#        alpha = .00000001
#        yy = (np.log(labels[:,i]-labels[:,i].min()+alpha))
        sup_vec.fit(dataset_blend_train, labels[:,i])
        preds[:,i] = sup_vec.predict(dataset_blend_test).astype(float)
   #     preds[:,i] = np.exp(preds[:,i]-alpha + min)
        preds_10[:,i] = sup_vec.predict(dataset_blend_train_10).astype(float)
   #     preds_10[:,i] = np.exp(preds_10[:,i]-alpha + min)
    elif(i==2): #Ph
#        sup_vec =  AdaBoostRegressor(svm.SVR(C=5000.0, verbose = 2, gamma = 2),n_estimators=20, loss='exponential')

        sup_vec = svm.SVR(C=12000, verbose = 2, degree = 5)
        #sup_vec.fit(preprocessing.scale(xtrain[:,:3578].astype(float)), labels[:,i])
        #preds[:,i] = sup_vec.predict(preprocessing.scale(xtest[:,:3578].astype(float))).astype(float)
        #preds_10[:,i] = sup_vec.predict(preprocessing.scale(xdata_10[:,:3578].astype(float))).astype(float)
        sup_vec.fit(xtrain[:,:3578], labels[:,i])
        preds[:,i] = sup_vec.predict(xtest[:,:3578]).astype(float)
        preds_10[:,i] = sup_vec.predict(xdata_10[:,:3578]).astype(float)
    elif(i==3):#Soc
        sup_vec = svm.SVR(C=11000, verbose = 2, degree = 10, shrinking = False)
        #sup_vec.fit(preprocessing.scale(xtrain[:,:3578].astype(float)), labels[:,i])
        #preds[:,i] = sup_vec.predict(preprocessing.scale(xtest[:,:3578].astype(float))).astype(float)
        #preds_10[:,i] = sup_vec.predict(preprocessing.scale(xdata_10[:,:3578].astype(float))).astype(float)
        sup_vec.fit(xtrain[:,:3578], labels[:,i])
        preds[:,i] = sup_vec.predict(xtest[:,:3578]).astype(float)
        preds_10[:,i] = sup_vec.predict(xdata_10[:,:3578]).astype(float)
    else:
        sup_vec = svm.SVR(C=11000.0, verbose = 2, degree = 5)
        sup_vec.fit(xtrain, labels[:,i])
        preds[:,i] = sup_vec.predict(xtest).astype(float)
        preds_10[:,i] = sup_vec.predict(xdata_10).astype(float)

sample = pd.read_csv('sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('beating_benchmark.csv', index = False)
rmsCa = sqrt(mean_squared_error(labels_10[:,0], preds_10[:,0]))
rmsP = sqrt(mean_squared_error(labels_10[:,1], preds_10[:,1]))
rmsPh = sqrt(mean_squared_error(labels_10[:,2], preds_10[:,2]))
rmsSOC = sqrt(mean_squared_error(labels_10[:,3], preds_10[:,3]))
rmsSand = sqrt(mean_squared_error(labels_10[:,4], preds_10[:,4]))
rms = rmsCa + rmsP + rmsPh + rmsSOC + rmsSand
print "C = 10000.0, RMS = %f, rmsCa=%f rmsP=%f, rmsPh=%f, rmsSoc=%f, rmsSand=%f"%  (rms/5, rmsCa, rmsP, rmsPh, rmsSOC, rmsSand)

