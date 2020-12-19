import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.cluster import KMeans
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx

#plt.ion() #Makes plots interactive in ipython
plt.ioff() #Runs code without opening figures 

#Define a function that will create line to distinguish SFGs from QGs
def seperationline(x):
    m=(-2.6- -0.3)/(9.1-11.7)
    y=m*(x-11.7)-0.3
    return y

#Define a function will help visualize the matricies in the Spectral Clustering
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

#Importing All MaNGA Data from DPRall Schema
data=pd.read_csv('CompleteTable.csv')

#Generating Distinction Line 
x1=np.linspace(7,13,len(data))
y1=seperationline(x1)


#Plotting Data
# plt.title('Mass Vs SFR of Galaxies in MaNGA')
# plt.xlabel('Log of Mass')
# plt.ylabel('Log of SFR')
# plt.hist2d(np.log10(data.loc[:,'nsa_sersic_mass']),np.log10(data.loc[:,'sfr_tot']), cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
# plt.plot(x1,y1)
# plt.colorbar()
# plt.show()


a=0.10 #Variable Parameter to find GVG region threshold

#Using np.where to classify galaxies as QG, SFG or GVG
QG=np.where(np.log10(data.loc[:,'sfr_tot'])<seperationline(np.log10(data.loc[:,'nsa_sersic_mass']-a))) #Creates an array of the indicies where this condition is true
QG=data.loc[QG] #Taking the rows of data that correspond to QGs

SFG=np.where(np.log10(data.loc[:,'sfr_tot'])>seperationline(np.log10(data.loc[:,'nsa_sersic_mass']+a)))
SFG=data.loc[SFG]

GVG=np.where((np.log10(data.loc[:,'sfr_tot'])<seperationline(np.log10(data.loc[:,'nsa_sersic_mass']))+a) & (np.log10(data.loc[:,'sfr_tot'])>seperationline(np.log10(data.loc[:,'nsa_sersic_mass']))-a))
GVG=data.loc[GVG]

#Testing if this was successful using a plot 
plt.title('Mass Vs SFR of Galaxies in MaNGA')
plt.xlabel('Log of Mass')
plt.ylabel('Log of SFR')
plt.scatter(np.log10(QG.loc[:,'nsa_sersic_mass']),np.log10(QG.loc[:,'sfr_tot']),c='red',label='QGs')
plt.scatter(np.log10(SFG.loc[:,'nsa_sersic_mass']),np.log10(SFG.loc[:,'sfr_tot']), c='blue', label='SFGs')
plt.scatter(np.log10(GVG.loc[:,'nsa_sersic_mass']),np.log10(GVG.loc[:,'sfr_tot']), c='green', label='GVGs')
plt.plot(x1,y1)
plt.legend()
plt.savefig('Distinction Line Classification.png')
plt.show()
plt.figure()

data2=data[(data.nsa_sersic_mass>0)&(data.sfr_tot>0)] #Define data 2 to get rid of -inf values in the data frame



#Chosing best Epsilon paramter used in DBScan for the data

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))
distances , indices = nbrs.kneighbors(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances) #y value of this figure at max inflection will give us ideal epsilon for DB scan 
plt.title('Optimal Epsilon for DB Scan')
plt.show()
plt.figure()


#Using sci kit learn DBSCAN function (min sample is 62, as this gets bigger we get very small and distinct groups, smaller and the two groups merge)

clustering= DBSCAN(eps=0.15, min_samples=62).fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']])) #esp is radius epsilion from point and min_samples is min amount of samples in neighbourhood
cluster= clustering.labels_ 

random= np.random.randint(0,len(data2),len(data2)) #Creating random integer array to use for resampling 

#Define Resampling Technique 
clustering_random=DBSCAN(eps=0.15, min_samples=62).fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']].iloc[random]))
cluster_random=clustering_random.labels_

plt.scatter(np.log10(data2['nsa_sersic_mass']),np.log10(data2['sfr_tot']),c=cluster,alpha=0.2)
plt.title('DB Scan Clustering')
plt.colorbar()
plt.show()
plt.figure()

#Plot the resample 
plt.scatter(np.log10(data2['nsa_sersic_mass']).iloc[random],np.log10(data2['sfr_tot']).iloc[random],c=cluster_random,alpha=0.2)
plt.title('DB Scan Clustering Resampling')
plt.colorbar()
plt.show()
plt.figure()



# Fit K-means with Scikit
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
kmeans.fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))

# Predict the cluster for all the samples
P = kmeans.predict(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))

colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', P))
plt.scatter(np.log10(data2['nsa_sersic_mass']), np.log10(data2['sfr_tot']), c=colors, marker="o", picker=True)
plt.title('K-Means Clustering')
plt.show()
plt.figure()



#Spectral Cluestering Approach with SciKit 
W1 = pairwise_distances(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]), metric="euclidean") #Create similarity matrix by finding distances between each pair of points
# vectorizer = np.vectorize(lambda x: 1 if x < 0.15 else 0) #Adjacency matrix conditon based on distance conditions
# W = np.vectorize(vectorizer) (W) #Takes similarity matrix and makes it adjacency matrix
W= (W1<0.15).astype(int) #Takes similarity matrix and makes it adjacency matrix by turning any entry that satifies condition to 1 and any that does not to 0
print(W)

#Generating Plot to Visualize nodes we created above
G = nx.random_graphs.erdos_renyi_graph(10, 0.5)
#draw_graph(G)
W = nx.adjacency_matrix(G)
#print(W.todense())

# Constructing the degree matrix by summing all the elements of the corresponding row in the adjacency matrix.
D = np.diag(np.sum(np.array(W.todense()), axis=1))


# Constructing the laplacian matrix by subtracting the adjacency matrix from the degree matrix
L = D - W

#If the graph (W) has K connected components, then L has K eigenvectors with an eigenvalue of 0.
e, v = np.linalg.eig(L)

#Actually using spectral clustering with sci-kit learn
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
sc_clustering = sc.fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))
plt.scatter(np.log10(data2[['nsa_sersic_mass']]), np.log10(data2[['sfr_tot']]), c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('Spectral Clustering')
plt.show()
plt.figure()

#Creating Bins of data using Initial Sample Selection Graph to create PDF of GVGs
bin_edges=np.linspace(7,13,num=6)

bin1= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[0])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[1]))
bin1=data2.loc[bin1]
 
bin2= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[1])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[2]))
bin2=data2.loc[bin2]

# bin3= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[2])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[3]))
# bin3=data2.loc[bin3]

# bin4= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[3])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[4]))
# bin4=data2.loc[bin4]

# bin5= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[4])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[5]))
# bin5=data2.loc[bin5]









