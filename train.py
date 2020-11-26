import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import roc_curve, auc


data = pd.read_csv(r'C:\Users\ben\Desktop\model\SP500.csv',encoding = 'utf-8')
data = data.dropna(axis=0,how = 'any')
x = data.iloc[:,1:16]
x = x.to_numpy()
close_data = x[:,1:2]
#print(close_data)
y = data.iloc[:,16:17]
y_column = y.to_numpy()
y = []
for i in y_column:
	y.append(i[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

XGBmodel = XGBClassifier()
XGBmodel.fit(x_train,y_train)
XGB_predict = XGBmodel.predict(x_test)

output = pd.DataFrame([])
output["real"] = y_test
output["predict"] = XGB_predict

output.to_csv('result.csv', encoding='utf-8')
accuracy = accuracy_score(y_test, XGB_predict)
print(accuracy*100)

false_positive_rate,true_positive_rate,thresholds = roc_curve(y_test,XGB_predict)
a = auc(false_positive_rate, true_positive_rate)
print("auc=",a)