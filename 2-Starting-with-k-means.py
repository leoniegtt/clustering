"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time
import math

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = '../artificial/'
name="atom.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
#plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
inertie=[]
silouette=[]
davies=[]
calinski=[]
duree=[]
for k in range (7,8) :
    
    ts = time.time()
    duree.append(ts)

    #méthode k means
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    
    #méthode minibatch
    #model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie.append(model.inertia_)
    silouette.append(metrics.silhouette_score(datanp,labels))
    davies.append(metrics.davies_bouldin_score(datanp,labels))
    calinski.append(metrics.calinski_harabasz_score(datanp,labels))
    centroids = model.cluster_centers_


    #plt.figure(figsize=(6, 6))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
    plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
    #print("labels", labels)

    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(centroids)
    print(dists)


    #print(centroids[0])
    min_dist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    max_dist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    moy_dist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(k) : 
        dist = euclidean_distances(datanp[labels==i,:],centroids[i].reshape(1,-1) )
        min_dist[i]= np.min(dist)
        max_dist[i]=np.max(dist)
        moy_dist[i]=np.mean(dist)
    
    print("min dist")
    print(min_dist)
    print("max dist" )
    print(max_dist)
    print("moy dist")
    print(moy_dist)


ts = time.time()
duree.append(ts)

#plt.plot(duree)
#plt.show()    
        

#print(inertie)
plt.plot(inertie)
plt.show()

# print(silouette)
plt.plot(silouette)
#plt.show()
#print(datanp[labels==0])

plt.plot(davies)
#plt.show()
#print(datanp[labels==0])

plt.plot(calinski)
#plt.show()
#print(datanp[labels==0])
