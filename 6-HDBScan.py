from sklearn.datasets import make_blobs
import pandas as pd
import hdbscan

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = '../artificial/'
name="compound.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(datanp)
labels = clusterer.labels_
plt.scatter(datanp[:, 0], datanp[:, 1], c=labels , s=8)
plt.title("HDBSCAN Clusters")
plt.show()
probabilities = clusterer.probabilities_
condensed_tree = clusterer.condensed_tree_
min_spanning_tree = clusterer.minimum_spanning_tree_
outlier_scores = clusterer.outlier_scores_