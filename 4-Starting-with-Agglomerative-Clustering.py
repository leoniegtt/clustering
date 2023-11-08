import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = '../artificial/'
name="engytime.arff"

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
plt.show()



### FIXER la distance
# 

tps1 = time.time()
seuil_dist=5
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='ward', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (ward, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
duree=[]
davies=[]
for k in range (2,11) :
    ts = time.time()
    duree.append(ts)
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage='complete', n_clusters=k)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    kres = model.n_clusters_
    leaves=model.n_leaves_
    #print(labels)
    #print(kres)
    davies.append(metrics.davies_bouldin_score(datanp,labels))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Clustering agglomératif (single, n_cluster= "+str(k)+") "+str(name))
    
    plt.show()
    
    
    print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

ts = time.time()
duree.append(ts)

# plt.plot(duree)
# plt.show()   

#centroids = model.cluster_centers_


# #plt.figure(figsize=(6, 6))
# plt.scatter(f0, f1, c=labels, s=8)
# plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
# plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
# #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
# plt.show()

# print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
# #print("labels", labels)

# from sklearn.metrics.pairwise import euclidean_distances
# dists = euclidean_distances(centroids)
# print(dists)

#######################################################################