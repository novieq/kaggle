import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import Imputer

train_df = pd.read_csv('/Users/sayghosh/code/kaggle/AfricanSoilPrediction/training.csv', header=0)

training_y = train_df[['Ca','P','pH','SOC','Sand']]
training_x = train_df.drop(['PIDN','Ca','P','pH','SOC','Sand','Depth'], axis=1)
#print train_df
pickTrain = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']

x = training_x[pickTrain]
y = training_y.values
#===============================================================================
# imp=Imputer(missing_values=['NaN','None'],strategy='mean',axis=1)
# new_x = imp.fit_transform(x)
# new_y = imp.fit_transform(y)
# new_test_df = imp.transform(test_df)
#===============================================================================


clf_1 = DecisionTreeRegressor(max_depth=2)
clf_2 = DecisionTreeRegressor(max_depth=5)
clf_3 = DecisionTreeRegressor(max_depth=8)

clf_1.fit(x, y)

test_df = pd.read_csv('/Users/sayghosh/code/kaggle/AfricanSoilPrediction/sorted_test.csv', header=0)        # Load the train file into a dataframe
ids = test_df['PIDN']

new_test_df = test_df[pickTrain]
y_1 = clf_1.predict(new_test_df)
#print y_1

predictions_file = open("/Users/sayghosh/code/kaggle/AfricanSoilPrediction/predictions1.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PIDN","Ca","P","pH","SOC","Sand"])
open_file_object.writerows(zip(ids,y_1[:,0],y_1[:,1],y_1[:,2],y_1[:,3],y_1[:,4]))
predictions_file.close()
