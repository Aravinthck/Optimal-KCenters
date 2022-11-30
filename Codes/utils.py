import gurobipy as gp
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import warnings
import seaborn as sns
import random
plt.style.use('default')
from sklearn.preprocessing import StandardScaler
from collections.abc import Iterable
from scipy import stats

from timeit import default_timer
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.cluster import kmeans_plusplus
from matplotlib.ticker import FormatStrFormatter

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.center_initializer import random_center_initializer


from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES

from pyclustering.cluster.kmedians import kmedians
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils.metric import distance_metric,type_metric


from sklearn.neighbors import KDTree, BallTree
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


def std_scale(data,f):
    data = data.copy()
    scaler = StandardScaler()
    data.iloc[:,0:f] = scaler.fit_transform(data.iloc[:,0:f])
    # self.scaler = scaler
    return data, scaler


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def kmeans_WarmStart(data, K, randState):
    data = data.copy()
    kmeans = KMeans(n_clusters=K, init = 'k-means++',  random_state=randState).fit(data)
    best_centers = kmeans.cluster_centers_

    dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }

    best_centers = best_centers[best_centers[:, 0].argsort()]


    assignCluster = [dictCluster.get(x) for x in kmeans.labels_  ]
    
    binaryAssignVar = np.zeros((data.shape[0], K))

    for i , k in enumerate(assignCluster):        
        binaryAssignVar[i,k] = 1
        
    return best_centers, assignCluster, binaryAssignVar

    
def initConstraints(data, K, addConstrs = 5, randState = 0,ratio = 2):   
    data = data.copy()
    initConstrsEdge = set()
    initConstrsNear = set()

    X = data.to_numpy()
    centers, assignCluster, binaryAssignVar = kmeans_WarmStart(data, K, randState)
    distManh = manhattan_distances(X,centers)
    distManhCls = distManh*binaryAssignVar
    argSortDist = np.argsort(distManhCls, axis = 0)

    minIndx = np.sum(binaryAssignVar==0,axis = 0)
    initConstrsEdge = set(argSortDist[-addConstrs:].flatten())

    for k in range(K):
        initConstrsNear.update(set(argSortDist[minIndx[k]:minIndx[k]+int(addConstrs/ratio)][:,k].flatten()))

    return centers, assignCluster, binaryAssignVar, initConstrsEdge, initConstrsNear, 

    

def getDistAssignMat(X, center, Cik = []):

    dist = manhattan_distances(X,center)
    trueCik = None
    if len(Cik) == 0:
        trueCik = np.zeros_like(dist)
        trueCik[np.arange(len(dist)), dist.argmin(1)] = 1
        distM = dist*trueCik
    else:
        distM = dist*Cik
        

    return distM, trueCik


def getOptimalValue(X,center, outliersCnt = 0):
    distM , _ =   getDistAssignMat(X,center) 
    
    if outliersCnt == 0:
        maxErr = np.max(distM)
    else:
        distMO = distM.copy()
        outIndxFlat = np.argpartition(distM.flatten(), -outliersCnt)[-outliersCnt:] 
        outThres = distM.flatten()[outIndxFlat[0]]
        distMO[distMO >= outThres] = 0
        maxErr = np.max(distMO)

    return maxErr



def getClusterAssign(Cik, outliers = []):

    ClusterAssign = Cik.argmax(axis = 1) + 1

    if len(outliers) !=0:
        ClusterAssign[outliers] = 0
        
    return ClusterAssign 



def getConstraintPts(distM,K,outliersCnt = 0 ):

    trueOutIndx = []

    if outliersCnt>0:
        
        distMO = distM.copy()
        # outIndxFlat = np.argsort(distM.flatten())[-(outliersCnt+K):]
        outIndxFlat = np.argpartition(distM.flatten(), -outliersCnt)[-outliersCnt:] 

        # outThres = distM.flatten()[outIndxFlat[K]]
        outThres = distM.flatten()[outIndxFlat[0]]

        # print('True Outliers threshold: ', outThres)

        trueOutIndx = np.floor(outIndxFlat/K).astype(int)

        # print('True outliers: ', trueOutIndx )

        # trueoutliers_list.append(trueOutIndx[-outliersCnt:] )

        distMO[distMO >= outThres] = 0

        # max_error = np.max(distMO, axis=0)
        # print("max error excluding outliers: ", (max_error) )

        maxDistPtsK = list(np.argmax(distMO,axis = 0))

        # print('actual max pts when outliers',maxDistPtsK)

        maxError = np.array([distMO[j,i] for i , j in enumerate(maxDistPtsK)])


        # print("max error excluding outliers (argmax): ", (maxError) )


    else:

        maxDistPtsK = list(np.argmax(distM,axis = 0))
        maxError = np.array([distM[j,i] for i , j in enumerate(maxDistPtsK)])
        # print('actual max pts without outliers',maxDistPtsK)
        # print("max error excluding outliers (argmax): ", (maxError) )
        distMO = distM

    
    return maxDistPtsK, trueOutIndx, maxError, distMO



def getCentroids(X, K, trueCik, trueOutIndx = []):
    n,f = X.shape

    center = np.zeros((K,f))

    for k in range(K):
        outliers = 0

        for i in range(n):
            if i not in trueOutIndx:
                center[k,:] = center[k,:] + X[i,:]*trueCik[i,k]
            else:
                if trueCik[i,k] == 1:
                    # print(i,k)
                    outliers+=1
        # print(outliers)
        center[k,:] = center[k,:]/(np.sum(trueCik[:,k]) - outliers)

    return center 

def getSSE(X, center, labels, trueOutIndx = []):

    sse = 0

    for i in range(len(X)):
        if i not in trueOutIndx:
            j = labels[i]-1
            sse = sse + euclidean_distances(X[i,:].reshape(1,-1),center[j,:].reshape(1,-1))**2


    return sse
 
def getMoreConstraintPts(distM, K,cg_pts,n,trueCik):

    ptConstrs = []
    argSort = np.argsort(distM, axis = 0)

    for k in range(K):
        addpts_k = []
        argMaxIndx = -2 
        add_pt = argSort[argMaxIndx][k]

        while add_pt in cg_pts and argMaxIndx>(-n+1): 
            argMaxIndx-=1
            add_pt = argSort[argMaxIndx][k]
        addpts_k.extend([add_pt])


        argMinIndx = sum(trueCik[:,k]==0)
        if argMinIndx == len(trueCik):
            # all values in the column are zezo, can't find the closest pts to center
            continue
        add_pt = argSort[argMinIndx][k]

        while add_pt in cg_pts and argMinIndx<n-1:
            argMinIndx+=1
            add_pt = argSort[argMinIndx][k]
            
            # print('add pts', add_pt)
        addpts_k.extend([add_pt])

        # print('Max Min', k, addpts_k)

        ptConstrs.extend(addpts_k)    

    # print('Add more point constraints: ', ptConstrs)

    return ptConstrs


def getOutliersCnt(distM,K,cutoff):

    mad = np.zeros((1,K))
    med = np.zeros((1,K))
    mean = np.zeros((1,K))

    for k in range(K):
        dist_tmpk = [d for d in distM[:,k] if d > 0 ]
        med[0,k] = np.median(dist_tmpk)
        mean[0,k] = np.mean(dist_tmpk)
        mad[0,k] = stats.median_abs_deviation(dist_tmpk)
    # print('Median: ', med)
    # print('MAD: ', mad)
    # print('Mean: ', mean)
    sigmaK = 1.4826*mad  

    thres =  cutoff*sigmaK

    # print('threshold: ', thres)
    outcnt = np.sum(distM > thres)
    
    return outcnt




