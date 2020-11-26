import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree



data = pd.read_csv(r'C:\Users\ben\Desktop\pythonML\SP500.csv',encoding = 'utf-8')
data = data.dropna(axis=0,how = 'any')
data = data[22000:]

#x = data.iloc[:,[1,5,6,7,8,9,10,12,13,14]]
x = data.iloc[:,1:15]
x = x.to_numpy()

y = data.iloc[:,16:17]
y_column = y.to_numpy()
y = []
for i in y_column:
	y.append(i[0])

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.4,random_state = 42)

clf = tree.DecisionTreeClassifier(max_depth = 12)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)

#output = pd.DataFrame([])
#output["real"] = y_test
#output["predict"] = XGB_predict

#output.to_csv('result.csv', encoding='utf-8')
accuracy = accuracy_score(y_test, y_predict)
print(accuracy*100)


#err = []

#for i in range(len(y_test)):
#    if y_test[i] != y_predict[i]:
#        err.append(x_test[i,0])
#    else:
#        err.append(np.nan)

