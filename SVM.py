
"Linear and Non-linear SVM classification using CV accuracy "

"Import Liabraries"
import numpy as np
import pandas as pd
from sklearn import preprocessing,datasets,svm

"Import dataset"
data = pd.read_csv('breast_cancer.csv',sep=',')
#data = pd.DataFrame(datasets.load_breast_cancer().data)
y = datasets.load_breast_cancer().target

"scale data"
scaleData = pd.DataFrame(preprocessing.scale(data))
"folds size"
fold = 5
"Shuffle data"
shuf = np.random.choice(fold,len(scaleData),replace=True)
"different types of kenels"
kernel = ["linear","poly","rbf","sigmoid"]
#C = np.arange(1,11)
Accuracy = []
C = [0.01,1,10,100,1000,10000]

"CV measure for different kernels"
for k in kernel:
    for c in C: 
        accuracy = []
        for i in np.arange(fold):
            "divide data into test and train"
            X_test = scaleData.iloc[np.where(shuf == i)]
            X_train = scaleData.iloc[np.where(shuf != i)]
            y_test = y[np.where(shuf == i)]
            y_train = y[np.where(shuf != i)]
            
            "Create model for SVM"
            model = svm.SVC(C = c,kernel = k)
            "fit train data in model"
            model.fit(X_train,y_train)
            "predict test data using fitted model"
            pred = model.predict(X_test)
            "find accuracy for model"
            acc = model.score(X_test,y_test)
            accuracy.append(acc)
        "CV Accuracy for different Kernels"
        Accuracy.append(np.mean(accuracy))
        print("Accuracy for kernel ",k,"and C  ",c,"is",np.mean(accuracy))
