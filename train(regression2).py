import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing




def denormalize(df,norm_value):
    original_value = df['Close'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    return denorm_value

def normalize(df):
    newdf = df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    newdf['Open'] = min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['Close']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['Low']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['High']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['Volume']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['MACD']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['MACD_SIG']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['MACD_HIST']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['RSI']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['MA']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['Slope']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['BBAND_HIGH']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['BBAND_MID']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['BBAND_LOW']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['MFI']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['high_delay']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['low_delay']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    #newdf['Trend']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['delay']= min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))

    return newdf
data = pd.read_csv(r'C:\Users\ben\Desktop\model\SP500.csv',encoding = 'utf-8')
data = data.dropna(axis = 0,how = 'any')
data_norm = normalize(data)

x = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,12,13,14,17]]
#x = data_norm.iloc[:,1:18]
x = x.to_numpy()



y_data = data_norm.iloc[:,19:20]
y_column = y_data.to_numpy()
y = []
for i in y_column:
    y.append(i[0])

    
    
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,shuffle = False) 
                                                            
#high_data = denormalize(data,x_test[:,14:15])
#low_data = denormalize(data,x_test[:,15:16])

high=[]
low=[]

#for i in high_data:
#    high.append(i[0])
#for i in low_data:
#    low.append(i[0])
reg = xgb.XGBRegressor(max_depth=50)
reg.fit(x_train,y_train) 
y_pred = reg.predict(x_test)   
y_pred_denorm = denormalize(data,reg.predict(x_test))
#y_test = denormalize(data,y_test)
#print(y_pred)

#print(y_test)
results=[]#1 對 #0 錯

#for i in range(len(low)):
#    if(y_pred_denorm[i,0] <= high[i])and(y_pred_denorm[i,0] >= low[i]):
#        results.append(1)
#    else:
#        results.append(0)     
#one = 0
#zero = 0
#for i in results:
#    if (i == 0):
#        zero = zero + 1        
#    else:
#        one = one +1
        
#print(one/(zero+one)*100)

plt.plot(y_pred)
plt.plot(y_test)
#plt.plot(y_pred[:300])
#plt.plot(y_test[:300])
plt.savefig(r'C:\Users\ben\Desktop\model\figure_test', dpi=1000)
plt.show()
