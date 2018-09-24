
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import preprocessing,datasets

data = pd.DataFrame(datasets.load_iris().data)
randomShuf = np.random.choice(np.arange(len(data)),len(data),replace=True)
shuf = np.random.choice(3,len(data),replace=True)
p = data.shape[1]
clusterNum = 3
itrmax = 100
mu = []
PriorProb = []
cluster = []
PostProb = []

for i in np.arange(clusterNum):
    cluster1 = data.iloc[shuf == i]
    pi = len(cluster1) / len(data)
    PriorProb.append(pi)
    cluster.append(cluster1)
    mu = cluster1.mean(axis=0)
    CovMat = cluster1.cov()
    invCovMat = np.linalg.inv(CovMat)
    norm = []
    for j in np.arange(data.shape[0]):
        ex = np.exp(-(0.5*float(np.mat(data.iloc[j] - mu) * np.mat(invCovMat) * np.mat(data.iloc[j]-mu).T)))
        N = 1/(2*np.pi)**(p/2) * 1/(np.linalg.det(CovMat))**(1/2) * ex
        norm.append(N)
    PostProb.append(norm)

pi1 = len(data.iloc[shuf == 0]) / len(data)
pi2 = len(data.iloc[shuf == 1]) / len(data)
pi3 = len(data.iloc[shuf == 2]) / len(data)
mu = [pi1,pi2,pi3]

probabilityMat = pd.DataFrame(PostProb).T
itr = 0
while(True):
    W = []
    for i in np.arange(data.shape[0]):
        tp = probabilityMat.iloc[i] * PriorProb
        w = []
        for j in np.arange(clusterNum):
            w.append(tp[j]/ sum(tp))
        W.append(w)
        
    ric = (pd.DataFrame(W))
    
    phi = []
    for i in np.arange(clusterNum):
        phi.append(sum(ric[i])/data.shape[0])
    
    newMuList = []
    for j in np.arange(ric.shape[1]):
        tpMu = []    
        for i in np.arange(p):
            tpMu.append(sum(ric[j] * data[i])/sum(ric[j]))
        newMuList.append(tpMu)
    
    newMu = pd.DataFrame(newMuList)
    
    SigmaList = []
    for j in np.arange(ric.shape[1]):
        sigma = 0
        for i in np.arange(len(data)):
            sigma = sigma + ric[j][i] * (np.mat(data.iloc[i] - newMu.iloc[j])).T * (np.mat(data.iloc[i] - newMu.iloc[j]))
        SigmaList.append(sigma/sum(ric[j]))
        
    PriorProb = phi
    CovMat = SigmaList.copy()
    PostProb = []
    for i in np.arange(clusterNum):
        PostProb.append(multivariate_normal.pdf(data,newMu.iloc[i],CovMat[i]))
        
    probabilityMat = pd.DataFrame(PostProb).T
    itr += 1
    print(itr)
    if abs(sum(mu) - sum(newMu)) < 0.0001:
        break
    if itr > itrmax:
        break
    mu = newMu
