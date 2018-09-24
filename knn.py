#knn

import numpy as np
import pandas as pd
import random

data = pd.read_csv('breast_cancer.csv',sep=',')
data.drop(['id'],axis = 1,inplace=True)


X = data.drop(['diagnosis'],axis = 1)
y = np.array(data['diagnosis'].tolist())

scaleData = pd.DataFrame(preprocessing.scale(X))

shuf = np.random.choice(5,len(scaleData),replace=True)

def distance(x,y):
	return np.sqrt(sum((x-y)**2))

fold = 5
K = 5
fold_accuracy = []

for i in np.arange(fold):
    X_test = scaleData.iloc[np.where(shuf == i)]
    X_train = scaleData.iloc[np.where(shuf != i)]
    y_test = y[np.where(shuf == i)]
    y_train = y[np.where(shuf != i)]
    y_pred = []
    
    for j in np.arange(len(X_test)):
        dist = []
        
        for k in np.arange(len(X_train)):
            d = distance(X_test.iloc[j],X_train.iloc[k])
            dist.append(d)
            
        min_values = sorted(dist)[0:K]
        min_index = np.where(pd.DataFrame(dist).isin(min_values))[0]
        classes = y[min_index]
        
        if list(classes).count('M') >= np.ceil(K/2):
            y_pred.append('M')
        else:
            y_pred.append('B')
        
    acc = sum(y_pred == y_test) / len(y_test)
    print(acc)
    fold_accuracy.append(acc)
    
accuracy = np.mean(fold_accuracy)
print(accuracy)
    
