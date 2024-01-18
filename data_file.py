
import pandas as pd
import numpy as np
import sub_fun as sf
import pickle
import scipy.cluster.hierarchy as sch
import copy


## Response matrix: microbial abundance data 
Y = pd.read_csv('Y1.csv').to_numpy()  
Y = Y[:,range(2,Y.shape[1])]
Y = Y.astype('int')

## Save original data 
Yo = pd.read_csv('Y1.csv').to_numpy()  
Yo = Yo[:,range(2,Yo.shape[1])]
Yo = Yo.astype('int')


## Computation of the geometric mean:  
errx = 1e-5
delta  = np.empty(Y.shape[0])  
for i in range(Y.shape[0]):
    delta[i] = sf.get_geomean(Y[i], errx)
    
T = np.exp(np.mean(np.log(Y+delta.min()), axis=1))
Bs = np.sum(Y != 0, axis = 1)
Yi = (Y != 0) + 0

# Correction for the geometric mean 
T_i = np.exp(np.mean(np.log(Y.T+delta), axis=0))
Y = (Y.T+delta).T
Y = Y.astype('int')

## Geochemical covariates 
X = pd.read_csv('X.csv').iloc[:,1:].to_numpy()    
X = np.subtract(X, np.mean(X, axis = 0)) # mean centering
X = X/np.std(X,axis=0)                   # scaling 


## Spatio-temporal indicators
Z = pd.read_csv('Z.csv')
I = Z.to_numpy()[:,range(1,Z.shape[1])]   
     
# B biome indicator 
Ifac = I[:,0]
fac = np.unique(Ifac)
B = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    B[np.where(Ifac == fac[i]),i] = 1
    
# Longhurst province indicator for spatial location
Ifac = I[:,1]
fac = np.unique(Ifac)
S = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    S[np.where(Ifac == fac[i]),i] = 1
    

# Q quarter indicator for time;
Ifac = I[:,4]
fac = np.unique(Ifac)
Q = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    Q[np.where(Ifac == fac[i]),i] = 1
    
n,q = Y.shape
# Ordering of the rows and columns for the visulaization of the data
r_ord = sch.linkage(Y,metric = 'euclidean',\
                           optimal_ordering = True, method="ward")
r_ord = sch.leaves_list(r_ord)
c_ord = sch.linkage(Y.T,metric = 'euclidean',\
                           optimal_ordering = True, method="ward")
c_ord = sch.leaves_list(c_ord)

