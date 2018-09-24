

"Import libraries"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.decomposition import PCA

"Import Datasets"
#data = pd.DataFrame(datasets.load_iris().data)
#y = list(datasets.load_iris().target)
data = pd.read_csv("glass.csv",sep=",")
data.drop(['Type'],axis=1,inplace=True)

"Apply DBSCAN model and its parameters"
model = DBSCAN(eps=0.5,min_samples=5,metric="euclidean",leaf_size=30)

"Fit data in given model"
model.fit(data)

"classified clusters"
model.labels_

"PCA decomposition for plotting"
pca = PCA(n_components=2).fit(data)
pca_2D = pca.transform(data)

"Plot clusters and Noise"
for i in np.arange(pca_2D.shape[0]):
    if model.labels_[i] == 0:
        c1 = plt.scatter(pca_2D[i,0],pca_2D[i,1],c="r",marker='+')
    elif model.labels_[i] == 1:
        c2 = plt.scatter(pca_2D[i,0],pca_2D[i,1],c="g",marker='o')
    elif model.labels_[i] == 2:
        c3 = plt.scatter(pca_2D[i,0],pca_2D[i,1],c="y",marker='.')
    elif model.labels_[i] == 3:
        c4 = plt.scatter(pca_2D[i,0],pca_2D[i,1],c="k",marker='^')
    elif model.labels_[i] == -1:
        c5 = plt.scatter(pca_2D[i,0],pca_2D[i,1],c="b",marker='*')

plt.legend([c1,c2,c3,c4,c5],['cluster 1','cluster 2','cluster 3','cluster 4','Noise'])
plt.title('DBSCAN clusters and noise')
plt.show()
