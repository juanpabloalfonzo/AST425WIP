import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.stats import norm
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
from sklearn.preprocessing import StandardScaler
from marvin.tools.maps import Maps
from sklearn.decomposition import PCA

plt.ion() #Makes plots interactive in ipython
#plt.ioff() #Runs code without opening figures 

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

#Define a function to help us find the intersection(s) of the PDF Gaussians and return an array of the x-coordinate(s)
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

#Define a Function that does PCA using certain Marvin Maps and makes profile and scree plots.
#Returns PCA Vector Componets, and Ratio Variance as an array
#Single does one specific galaxy at a time, multi is equppied to take a dataframe of galaxies at once in the plateifu input
#Galaxy profile plot checks if data frame or string is being inputted and uses the corresponding function as needed

def galaxy_profile_plot_single(plateifu,Num_PCA_Vectors):
    
    maps = Maps(plateifu=plateifu)
    print(maps)
    # get an emission line map
    haflux = maps.emline_gflux_ha_6564
    values_flux = haflux.value
    ivar_flux = haflux.ivar
    mask_flux = haflux.mask
    #haflux.plot()

    maps = Maps(plateifu=plateifu)
    print(maps)
    # get an emission line map
    ha_vel = maps.emline_gvel_ha_6564
    values_vel = ha_vel.value
    ivar_vel = ha_vel.ivar
    mask_vel = ha_vel.mask
    #ha_vel.plot()

    maps = Maps(plateifu=plateifu)
    print(maps)
    # get an emission line map
    ha_sigma = maps.emline_sigma_ha_6564
    values_sigma = ha_sigma.value
    ivar_sigma = ha_sigma.ivar
    mask_sigma = ha_sigma.mask
    #ha_sigma.plot()

    maps = Maps(plateifu=plateifu)
    print(maps)
    # get an emission line map
    ha_ew = maps.emline_gew_ha_6564
    values_ew = ha_vel.value
    ivar_ew = ha_vel.ivar
    mask_ew = ha_vel.mask
    #ha_ew.plot()

    maps = Maps(plateifu=plateifu)
    print(maps)
    # get an emission line map
    stellar_vel = maps.stellar_vel
    values_stellar_vel = stellar_vel.value
    ivar_stellar_vel = stellar_vel.ivar
    mask_stellar_vel = stellar_vel.mask
    #stellar_vel.plot()

    maps = Maps(plateifu=plateifu)
    print(maps)
    # get an emission line map
    stellar_sigma = maps.stellar_sigma
    values_stellar_sigma = stellar_sigma.value
    ivar_stellar_sigma = stellar_sigma.ivar
    mask_stellar_sigma = stellar_sigma.mask
    #stellar_vel.plot()
   
    values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten()])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    #principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

    x = np.arange(len(pca.explained_variance_ratio_))

    #PCA Screeplot
    plt.bar(x,pca.explained_variance_ratio_)
    pc_names=[]
    for i in range(len(pca.explained_variance_ratio_)):
        pc_names.append('PC'+str(i))
    plt.xticks(x,(pc_names))
    plt.title('Scree Plot '+str(plateifu))
    plt.xlabel('Principal components')
    plt.ylabel('Variance Explained')
    plt.show()
    plt.figure()

    #PCA Profile Plot
    
    variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma']}
    variables=pd.DataFrame(data=variables)
    
    for i in range(Num_PCA_Vectors):    
        plt.plot(variables.loc[0,:],pca.components_[i,:],label='PC'+str(i))
    plt.title('Component Pattern Profiles ' +str(plateifu))
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot_multi(plateifu,Num_PCA_Vectors):
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        print(maps)
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        print(maps)
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        print(maps)
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        print(maps)
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        print(maps)
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        print(maps)
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()
    
        values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten()])
        values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
        pca = PCA(n_components=Num_PCA_Vectors)
        principalComponents = pca.fit_transform(values)
        #principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

        x = np.arange(len(pca.explained_variance_ratio_))

        #PCA Screeplot
        plt.bar(x,pca.explained_variance_ratio_)
        pc_names=[]
        for a in range(len(pca.explained_variance_ratio_)):
            pc_names.append('PC'+str(a))
        plt.xticks(x,(pc_names))
        plt.title('Scree Plot '+str(plateifu.iloc[i]))
        plt.xlabel('Principal components')
        plt.ylabel('Variance Explained')
        plt.show()
        plt.figure()

        #PCA Profile Plot
        
        variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma']}
        variables=pd.DataFrame(data=variables)
        
        for b in range(Num_PCA_Vectors):    
            plt.plot(variables.loc[0,:],pca.components_[b,:],label='PC'+str(b))
        plt.title('Component Pattern Profiles '+ str(plateifu.iloc[i]))
        plt.ylabel('Correlation')
        plt.xlabel('Variable')
        plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
        plt.legend()
        plt.show()
        plt.figure()
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot(plateifu,Num_PCA_Vectors):
    if np.size(plateifu)<2:
        return(galaxy_profile_plot_single(plateifu,Num_PCA_Vectors))
    else:
        return(galaxy_profile_plot_multi(plateifu,Num_PCA_Vectors))


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
# plt.title('SFR vs Mass of Galaxies in MaNGA')
# plt.xlabel(r'$log(M/M_{\odot})$')
# plt.ylabel(r'$log(SFR/M_{\odot})$')
# plt.scatter(np.log10(QG.loc[:,'nsa_sersic_mass']),np.log10(QG.loc[:,'sfr_tot']),c='red',label='QGs')
# plt.scatter(np.log10(SFG.loc[:,'nsa_sersic_mass']),np.log10(SFG.loc[:,'sfr_tot']), c='blue', label='SFGs')
# plt.scatter(np.log10(GVG.loc[:,'nsa_sersic_mass']),np.log10(GVG.loc[:,'sfr_tot']), c='green', label='GVGs')
# plt.plot(x1,y1)
# plt.legend()
# plt.savefig('Distinction Line Classification.png')
# plt.show()
# plt.figure()

data2=data[(data.nsa_sersic_mass>0)&(data.sfr_tot>0)] #Define data 2 to get rid of -inf values in the data frame



#Chosing best Epsilon paramter used in DBScan for the data

# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))
# distances , indices = nbrs.kneighbors(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances) #y value of this figure at max inflection will give us ideal epsilon for DB scan 
# plt.title('Optimal Epsilon for DB Scan')
# plt.savefig('EpislionDBscan.png')
# plt.show()
# plt.figure()


#Using sci kit learn DBSCAN function (min sample is 62, as this gets bigger we get very small and distinct groups, smaller and the two groups merge)

clustering= DBSCAN(eps=0.15, min_samples=62).fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']])) #esp is radius epsilion from point and min_samples is min amount of samples in neighbourhood
cluster= clustering.labels_ 

random= np.random.randint(0,len(data2),len(data2)) #Creating random integer array to use for resampling 

#Define Resampling Technique 
clustering_random=DBSCAN(eps=0.15, min_samples=62).fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']].iloc[random]))
cluster_random=clustering_random.labels_

# plt.scatter(np.log10(data2['nsa_sersic_mass']),np.log10(data2['sfr_tot']),c=cluster,alpha=0.2)
# plt.title('DB Scan Clustering')
# plt.colorbar()
# plt.savefig('DBscanClustering.png')
# plt.show()
# plt.figure()

# #Plot the resample 
# plt.scatter(np.log10(data2['nsa_sersic_mass']).iloc[random],np.log10(data2['sfr_tot']).iloc[random],c=cluster_random,alpha=0.2)
# plt.title('DB Scan Clustering Resampling')
# plt.colorbar()
# plt.show()
# plt.figure()



# Fit K-means with Scikit
# kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
# kmeans.fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))

# # Predict the cluster for all the samples
# P = kmeans.predict(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))

# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', P))
# plt.scatter(np.log10(data2['nsa_sersic_mass']), np.log10(data2['sfr_tot']), c=colors, marker="o", picker=True)
# plt.title('K-Means Clustering')
# plt.savefig('K-Means.png')
# plt.show()
# plt.figure()



# #Spectral Cluestering Approach with SciKit 
# W1 = pairwise_distances(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]), metric="euclidean") #Create similarity matrix by finding distances between each pair of points
# # vectorizer = np.vectorize(lambda x: 1 if x < 0.15 else 0) #Adjacency matrix conditon based on distance conditions
# # W = np.vectorize(vectorizer) (W) #Takes similarity matrix and makes it adjacency matrix
# W= (W1<0.15).astype(int) #Takes similarity matrix and makes it adjacency matrix by turning any entry that satifies condition to 1 and any that does not to 0
# print(W)

# #Generating Plot to Visualize nodes we created above
# G = nx.random_graphs.erdos_renyi_graph(10, 0.5)
# #draw_graph(G)
# W = nx.adjacency_matrix(G)
# #print(W.todense())

# # Constructing the degree matrix by summing all the elements of the corresponding row in the adjacency matrix.
# D = np.diag(np.sum(np.array(W.todense()), axis=1))


# # Constructing the laplacian matrix by subtracting the adjacency matrix from the degree matrix
# L = D - W

# #If the graph (W) has K connected components, then L has K eigenvectors with an eigenvalue of 0.
# e, v = np.linalg.eig(L)

#Actually using spectral clustering with sci-kit learn
# sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
# sc_clustering = sc.fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']]))
# plt.scatter(np.log10(data2[['nsa_sersic_mass']]), np.log10(data2[['sfr_tot']]), c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
# plt.title('Spectral Clustering')
# plt.savefig('SpectralClustering.png')
# plt.show()
# plt.figure()

#Creating Bins of data using Initial Sample Selection Graph to create PDF of GVGs
bin_edges=np.linspace(7,13,num=6)

bin1= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[0])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[1]))
bin1=data2.iloc[bin1]
 
bin2= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[1])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[2]))
bin2=data2.iloc[bin2]

bin3= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[2])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[3]))
bin3=data2.iloc[bin3]

bin4= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[3])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[4]))
bin4=data2.iloc[bin4]

bin5= np.where((np.log10(data2.loc[:,'nsa_sersic_mass'])>bin_edges[4])& (np.log10(data2.loc[:,'nsa_sersic_mass'])<bin_edges[5]))
bin5=data2.iloc[bin5]

#Separating Galaxies in each bin into SFG and QG based on the previous definition 
bin1_SFG=np.where(np.log10(bin1.loc[:,'sfr_tot'])>seperationline(np.log10(bin1.loc[:,'nsa_sersic_mass'])))
bin1_SFG=bin1.iloc[bin1_SFG]

bin1_QG=np.where(np.log10(bin1.loc[:,'sfr_tot'])<seperationline(np.log10(bin1.loc[:,'nsa_sersic_mass'])))
bin1_QG=bin1.iloc[bin1_QG]

bin2_SFG=np.where(np.log10(bin2.loc[:,'sfr_tot'])>seperationline(np.log10(bin2.loc[:,'nsa_sersic_mass'])))
bin2_SFG=bin2.iloc[bin2_SFG]

bin2_QG=np.where(np.log10(bin2.loc[:,'sfr_tot'])<seperationline(np.log10(bin2.loc[:,'nsa_sersic_mass'])))
bin2_QG=bin2.iloc[bin2_QG]

bin3_SFG=np.where(np.log10(bin3.loc[:,'sfr_tot'])>seperationline(np.log10(bin3.loc[:,'nsa_sersic_mass'])))
bin3_SFG=bin3.iloc[bin3_SFG]

bin3_QG=np.where(np.log10(bin3.loc[:,'sfr_tot'])<seperationline(np.log10(bin3.loc[:,'nsa_sersic_mass'])))
bin3_QG=bin3.iloc[bin3_QG]

bin4_SFG=np.where(np.log10(bin4.loc[:,'sfr_tot'])>seperationline(np.log10(bin4.loc[:,'nsa_sersic_mass'])))
bin4_SFG=bin4.iloc[bin4_SFG]

bin4_QG=np.where(np.log10(bin4.loc[:,'sfr_tot'])<seperationline(np.log10(bin4.loc[:,'nsa_sersic_mass'])))
bin4_QG=bin4.iloc[bin4_QG]

bin5_SFG=np.where(np.log10(bin5.loc[:,'sfr_tot'])>seperationline(np.log10(bin5.loc[:,'nsa_sersic_mass'])))
bin5_SFG=bin5.iloc[bin5_SFG]

bin5_QG=np.where(np.log10(bin5.loc[:,'sfr_tot'])<seperationline(np.log10(bin5.loc[:,'nsa_sersic_mass'])))
bin5_QG=bin5.iloc[bin5_QG]

#Creating Histograms for each bin and fitting a Gaussian to them

x2=np.linspace(0,3) #Useful to define x-axis of Gaussians for each bin 

b=np.where(bin1_SFG.loc[:,'specindex_1re_dn4000']>-999) #Getting rid of the -999 entires 
bin1_SFG=bin1_SFG.iloc[b]

#Defining the mean and std of the GVG Gaussian as the intersection point between SFG and QG Gaussians and 1/2 of the QG std respectively, as in Angthopo et al. (2019)
# bin1_GVG_mean=solve(np.mean(bin1_SFG.loc[:,'specindex_1re_dn4000']),np.mean(bin1_QG.loc[:,'specindex_1re_dn4000']),np.std(bin1_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin1_QG.loc[:,'specindex_1re_dn4000']))
# bin1_GVG_std=0.5*np.std(bin1_QG.loc[:,'specindex_1re_dn4000'])

#Bin 1
# plt.title('7 < log M < 8.2 ($M_{\odot}$)')
# plt.hist(bin1_SFG.loc[:,'specindex_1re_dn4000'], color= 'blue', label= 'SFG', density='True', alpha=0.4)
# plt.hist(bin1_QG.loc[:,'specindex_1re_dn4000'], color= 'red', label='QG', density='True', alpha=0.4)
# plt.plot(x2,norm.pdf(x2,np.mean(bin1_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin1_SFG.loc[:,'specindex_1re_dn4000'])), color= 'blue', label='Gaussian Distribution SFG')
# plt.plot(x2,norm.pdf(x2,np.mean(bin1_QG.loc[:,'specindex_1re_dn4000']),np.std(bin1_QG.loc[:,'specindex_1re_dn4000'])), color= 'red', label='Gaussian Distribution QG')
# # plt.plot(x2,norm.pdf(x2,bin1_GVG_mean,bin1_GVG_std), color= 'green', label='Gaussian Distribution GVG')
# plt.xlabel('Mean Dn(4000) at 1 Effective Radius')
# plt.ylabel('Frequency (Normalized)')
# plt.legend()
# plt.savefig('Bin1.png')
# plt.show()
# plt.figure()

b=np.where(bin2_QG.loc[:,'specindex_1re_dn4000']>-999) #Getting rid of the -999 entires again 
bin2_QG=bin2_QG.iloc[b]

#Bin 2

bin2_GVG_mean=solve(np.mean(bin2_SFG.loc[:,'specindex_1re_dn4000']),np.mean(bin2_QG.loc[:,'specindex_1re_dn4000']),np.std(bin2_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin2_QG.loc[:,'specindex_1re_dn4000']))
bin2_GVG_mean=bin2_GVG_mean[0] #Take the intersection point we care about 
bin2_GVG_std=0.5*np.std(bin2_QG.loc[:,'specindex_1re_dn4000'])

# plt.title('8.2 < log M < 9.4 ($M_{\odot}$)')
# plt.hist(bin2_SFG.loc[:,'specindex_1re_dn4000'], color= 'blue', label= 'SFG', density='True', alpha=0.4)
# plt.hist(bin2_QG.loc[:,'specindex_1re_dn4000'], color= 'red', label='QG', density='True', alpha=0.4)
# plt.plot(x2,norm.pdf(x2,np.mean(bin2_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin2_SFG.loc[:,'specindex_1re_dn4000'])), color= 'blue', label='Gaussian Distribution SFG')
# plt.plot(x2,norm.pdf(x2,np.mean(bin2_QG.loc[:,'specindex_1re_dn4000']),np.std(bin2_QG.loc[:,'specindex_1re_dn4000'])), color= 'red', label='Gaussian Distribution QG')
# plt.plot(x2,norm.pdf(x2,bin2_GVG_mean,bin2_GVG_std), color= 'green', label='Gaussian Distribution GVG')
# plt.xlabel('Mean Dn(4000) at 1 Effective Radius')
# plt.ylabel('Frequency (Normalized)')
# plt.xlim(0.8,2.2)
# plt.legend()
# plt.savefig('Bin2.png')
# plt.show()
# plt.figure()

b=np.where(bin3_QG.loc[:,'specindex_1re_dn4000']>-999) #Getting rid of the -999 entires again 
bin3_QG=bin3_QG.iloc[b]


#Bin 3

bin3_GVG_mean=solve(np.mean(bin3_SFG.loc[:,'specindex_1re_dn4000']),np.mean(bin3_QG.loc[:,'specindex_1re_dn4000']),np.std(bin3_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin3_QG.loc[:,'specindex_1re_dn4000']))
bin3_GVG_mean=bin3_GVG_mean[1]
bin3_GVG_std=0.5*np.std(bin3_QG.loc[:,'specindex_1re_dn4000'])

# plt.title('9.4 < log M < 10.6 ($M_{\odot}$)')
# plt.hist(bin3_SFG.loc[:,'specindex_1re_dn4000'], color= 'blue', label= 'SFG', density='True', alpha=0.4)
# plt.hist(bin3_QG.loc[:,'specindex_1re_dn4000'], color= 'red', label='QG', density='True', alpha=0.4)
# plt.plot(x2,norm.pdf(x2,np.mean(bin3_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin3_SFG.loc[:,'specindex_1re_dn4000'])), color= 'blue', label='Gaussian Distribution SFG')
# plt.plot(x2,norm.pdf(x2,np.mean(bin3_QG.loc[:,'specindex_1re_dn4000']),np.std(bin3_QG.loc[:,'specindex_1re_dn4000'])), color= 'red', label='Gaussian Distribution QG')
# plt.plot(x2,norm.pdf(x2,bin3_GVG_mean,bin3_GVG_std), color= 'green', label='Gaussian Distribution GVG')
# plt.xlabel('Mean Dn(4000) at 1 Effective Radius')
# plt.ylabel('Frequency (Normalized)')
# plt.legend()
# plt.savefig('Bin3.png')
# plt.show()
# plt.figure()

b=np.where(bin4_QG.loc[:,'specindex_1re_dn4000']>-999) #Getting rid of the -999 entires again 
bin4_QG=bin4_QG.iloc[b]

#Bin 4

bin4_GVG_mean=solve(np.mean(bin4_SFG.loc[:,'specindex_1re_dn4000']),np.mean(bin4_QG.loc[:,'specindex_1re_dn4000']),np.std(bin4_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin4_QG.loc[:,'specindex_1re_dn4000']))
bin4_GVG_mean=bin4_GVG_mean[1]
bin4_GVG_std=0.5*np.std(bin4_QG.loc[:,'specindex_1re_dn4000'])

# plt.title('10.6 < log M < 11.8  ($M_{\odot}$)')
# plt.hist(bin4_SFG.loc[:,'specindex_1re_dn4000'], color= 'blue', label= 'SFG', density='True', alpha=0.4)
# plt.hist(bin4_QG.loc[:,'specindex_1re_dn4000'], color= 'red', label='QG', density='True', alpha=0.4)
# plt.plot(x2,norm.pdf(x2,np.mean(bin4_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin4_SFG.loc[:,'specindex_1re_dn4000'])), color= 'blue', label='Gaussian Distribution SFG')
# plt.plot(x2,norm.pdf(x2,np.mean(bin4_QG.loc[:,'specindex_1re_dn4000']),np.std(bin4_QG.loc[:,'specindex_1re_dn4000'])), color= 'red', label='Gaussian Distribution QG')
# plt.plot(x2,norm.pdf(x2,bin4_GVG_mean,bin4_GVG_std), color= 'green', label='Gaussian Distribution GVG')
# plt.xlabel('Mean Dn(4000) at 1 Effective Radius')
# plt.ylabel('Frequency (Normalized)')
# plt.legend()
# plt.savefig('Bin4.png')
# plt.show()
# plt.figure()

b=np.where(bin5_QG.loc[:,'specindex_1re_dn4000']>-999) #Getting rid of the -999 entires again 
bin5_QG=bin5_QG.iloc[b]

#Bin 5
# plt.title('11.8 < log M < 13 ($M_{\odot}$)')
# plt.hist(bin5_SFG.loc[:,'specindex_1re_dn4000'], color= 'blue', label= 'SFG', density='True', alpha=0.4)
# plt.hist(bin5_QG.loc[:,'specindex_1re_dn4000'], color= 'red', label='QG', density='True', alpha=0.4)
# plt.plot(x2,norm.pdf(x2,np.mean(bin5_SFG.loc[:,'specindex_1re_dn4000']),np.std(bin5_SFG.loc[:,'specindex_1re_dn4000'])), color= 'blue', label='Gaussian Distribution SFG')
# plt.plot(x2,norm.pdf(x2,np.mean(bin5_QG.loc[:,'specindex_1re_dn4000']),np.std(bin5_QG.loc[:,'specindex_1re_dn4000'])), color= 'red', label='Gaussian Distribution QG')
# plt.xlabel('Mean Dn(4000) at 1 Effective Radius')
# plt.ylabel('Frequency (Normalized)')
# plt.legend(loc='upper left')
# plt.savefig('Bin5.png')
# plt.show()
# plt.figure()

#Using the GVG Gaussians we can now extract the GVGs from each bin 
bin2_GVG=np.where((bin2.loc[:,'specindex_1re_dn4000']<bin2_GVG_mean+bin2_GVG_std) & (bin2.loc[:,'specindex_1re_dn4000']>bin2_GVG_mean-bin2_GVG_std))
bin2_GVG=bin2.iloc[bin2_GVG]

bin3_GVG=np.where((bin3.loc[:,'specindex_1re_dn4000']<bin3_GVG_mean+bin3_GVG_std) & (bin3.loc[:,'specindex_1re_dn4000']>bin3_GVG_mean-bin3_GVG_std))
bin3_GVG=bin3.iloc[bin3_GVG]

bin4_GVG=np.where((bin4.loc[:,'specindex_1re_dn4000']<bin4_GVG_mean+bin4_GVG_std) & (bin4.loc[:,'specindex_1re_dn4000']>bin4_GVG_mean-bin4_GVG_std))
bin4_GVG=bin4.iloc[bin4_GVG]

#Removing GVG overlap with SFG and QG data frames 
bin2_QG=bin2_QG.loc[bin2_QG.index.difference(bin2_GVG.index),] #Takes QG dataframe and removes any indencies (row number) that match those in the GVG dataframe
bin2_SFG=bin2_SFG.loc[bin2_SFG.index.difference(bin2_GVG.index),]

bin3_QG=bin3_QG.loc[bin3_QG.index.difference(bin3_GVG.index),] 
bin3_SFG=bin3_SFG.loc[bin3_SFG.index.difference(bin3_GVG.index),]

bin4_QG=bin4_QG.loc[bin4_QG.index.difference(bin4_GVG.index),] 
bin4_SFG=bin4_SFG.loc[bin4_SFG.index.difference(bin4_GVG.index),]


#Putting all bins of galaxies into 3 big data frames
SFG=pd.concat([bin1_SFG,bin2_SFG,bin3_SFG,bin4_SFG,bin5_SFG])
QG=pd.concat([bin1_QG,bin2_QG,bin3_QG,bin4_QG,bin5_QG])
GVG=pd.concat([bin2_GVG,bin3_GVG,bin4_GVG])


# plt.title('Stellar Sigma at 1 Re for Galaxies in Bin 2')
# plt.hist(bin2_GVG.loc[:,'stellar_sigma_1re'], color='green', density='True')
# plt.hist(bin2_SFG.loc[:,'stellar_sigma_1re'], color='blue', density='True')
# plt.hist(bin2_QG.loc[:,'stellar_sigma_1re'], color='red', density='True')
# plt.show()
# plt.figure()

###############################################################
###############################################################

                     #START OF PCA ANALYSIS

#################################################################
#################################################################



# #Import varoius maps from Marvin and use them for PCA

# maps = Maps(plateifu=bin3_GVG.loc[98,'plateifu'])
# print(maps)
# # get an emission line map
# haflux = maps.emline_gflux_ha_6564
# values_flux = haflux.value
# ivar_flux = haflux.ivar
# mask_flux = haflux.mask
# #haflux.plot()

# maps = Maps(plateifu=bin3_GVG.loc[98,'plateifu'])
# print(maps)
# # get an emission line map
# ha_vel = maps.emline_gvel_ha_6564
# values_vel = ha_vel.value
# ivar_vel = ha_vel.ivar
# mask_vel = ha_vel.mask
# #ha_vel.plot()

# maps = Maps(plateifu=bin3_GVG.loc[98,'plateifu'])
# print(maps)
# # get an emission line map
# ha_sigma = maps.emline_sigma_ha_6564
# values_sigma = ha_sigma.value
# ivar_sigma = ha_sigma.ivar
# mask_sigma = ha_sigma.mask
# #ha_sigma.plot()

# maps = Maps(plateifu=bin3_GVG.loc[98,'plateifu'])
# print(maps)
# # get an emission line map
# ha_ew = maps.emline_gew_ha_6564
# values_ew = ha_vel.value
# ivar_ew = ha_vel.ivar
# mask_ew = ha_vel.mask
# #ha_ew.plot()

# maps = Maps(plateifu=bin3_GVG.loc[98,'plateifu'])
# print(maps)
# # get an emission line map
# stellar_vel = maps.stellar_vel
# values_stellar_vel = stellar_vel.value
# ivar_stellar_vel = stellar_vel.ivar
# mask_stellar_vel = stellar_vel.mask
# #stellar_vel.plot()

# maps = Maps(plateifu=bin3_GVG.loc[98,'plateifu'])
# print(maps)
# # get an emission line map
# stellar_sigma = maps.stellar_sigma
# values_stellar_sigma = stellar_sigma.value
# ivar_stellar_sigma = stellar_sigma.ivar
# mask_stellar_sigma = stellar_sigma.mask
# #stellar_vel.plot()



# #PCA Analysis

# #First we must standardize the data 
# values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten()])
# values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1


# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(values)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# #PCA Plot
# # plt.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2])
# # plt.ylabel('PC 2')
# # plt.xlabel('PC 1')
# # #plt.xlim(-1,1)
# # #plt.ylim(-1,1)
# # plt.show()
# # plt.figure()



# x = np.arange(len(pca.explained_variance_ratio_))
# #PCA Screeplot
# plt.bar(x,pca.explained_variance_ratio_)
# for i in range(len(pca.explained_variance_ratio_)):
#   plt.xticks(x,('PC'+str(i)))
# plt.title('Scree Plot')
# plt.xlabel('Principal components')
# plt.ylabel('Variance Explained')
# plt.show()
# plt.figure()

# #PCA Profile Plot
# PC0flux=pca.components_[0,0]
# PC0vel=pca.components_[0,1]
# PC0ew=pca.components_[0,2]
# PC0sigma=pca.components_[0,3]
# PC0stellar_vel=pca.components_[0,4]
# PC0stellar_sigma=pca.components_[0,5]
# PC0_values=np.column_stack([PC0flux,PC0vel,PC0ew,PC0sigma,PC0stellar_vel,PC0stellar_sigma])
# variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma']}
# variables=pd.DataFrame(data=variables)

# PC1_values=np.column_stack([pca.components_[1,0],pca.components_[1,1],pca.components_[1,2],pca.components_[1,3],pca.components_[1,4], pca.components_[1,5]])
# PC2_values=np.column_stack([pca.components_[2,0],pca.components_[2,1],pca.components_[2,2],pca.components_[2,3],pca.components_[2,4],pca.components_[2,5]])

# plt.title('Component Pattern Profiles')
# plt.plot(variables.loc[0,:],PC0_values[0,:],label='PC0')
# plt.plot(variables.loc[0,:],PC1_values[0,:],label='PC1')
# plt.plot(variables.loc[0,:],PC2_values[0,:],label='PC2')
# plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
# plt.legend()
# plt.show()
# plt.figure()

# galaxy1=galaxy_profile_plot(bin3_GVG.loc[4810,'plateifu'],3)
# PC4810=galaxy1[0]

galaxy2=galaxy_profile_plot(bin3_GVG.loc[98,'plateifu'],3)
PC98=galaxy2[0]

a=galaxy_profile_plot(bin5_QG.loc[:,'plateifu'],3)