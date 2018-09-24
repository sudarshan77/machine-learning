

"Import Libraries"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

"Import datasets"
data = pd.DataFrame(datasets.load_iris().data)
shuf = np.random.choice(len(data),size=len(data),replace=False)
y = pd.DataFrame(datasets.load_iris().target)
newData = data.iloc[shuf]
newY = y.iloc[shuf]

"Distance functon"
def distance(x,y):
    return np.sqrt(sum((x - y)**2))

"Radius and minimum number of neighbors"
eps = 0.3
nbr = 5

corePt = []
boundaryPt = []
noisePt = []
cluster = []

"Loop to find out core , boundary and noise"
for i in np.arange(len(newData)):
    dist = []
    frame1 = pd.DataFrame()
    frame1 = frame1.append(newData.iloc[i])
    frame1_mean = frame1.mean(axis=0)
    for j in np.arange(i+1,len(newData)):
        frame2 = pd.DataFrame()
        frame2 = frame2.append(newData.iloc[j])
        frame2_mean = frame2.mean(axis=0)
        d = distance(frame1_mean,frame2_mean)
        dist.append(d)
    
    nbrs = []
    for k in np.arange(len(dist)):
        if dist[k] <= eps:
            nbrs.append(dist[k])
    
    ind = np.where(pd.DataFrame(dist).isin(nbrs))[0]
    
    "Clusters"
    cluster.append([shuf[i],ind])
    
    "Decision for example to fall in core ,neighbor and noise"
    if(len(ind) >= nbr):
        corePt.append(shuf[i])
    elif(len(ind) < nbr):
        boundaryPt.append(shuf[i])
    elif(len(ind) == 0):
        noisePt.append(shuf[i])
        
        
