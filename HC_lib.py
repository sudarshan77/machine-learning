
"""Hirarchical clustering using sklearn and scipy"""

"Import libraries"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import hierarchical
from sklearn import datasets

"Import data"
data = pd.DataFrame(datasets.load_iris().data)
shuf = np.random.choice(len(data),size=len(data),replace=False)
y = pd.DataFrame(datasets.load_iris().target)
newData = data.iloc[shuf]
newY = y.iloc[shuf]

"Create dendogram using scipy"
dendogram = sch.dendrogram(sch.linkage(newData,method="ward"))

"Create AgglomerativeClustering model using sklearn"
model = hierarchical.AgglomerativeClustering(n_clusters=3,linkage="ward",affinity="euclidean")

"Fit data using created model"
model.fit(newData)
y_pred = model.fit_predict(newData)

"accuracy"
accuracy = sum(list(newY) == y_pred)/len(newY)
print(accuracy)
plt.show()
