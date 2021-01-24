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


def galaxy_profile_plot(plateifu,Num_PCA_Vectors):

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
    plt.title('Scree Plot')
    plt.xlabel('Principal components')
    plt.ylabel('Variance Explained')
    plt.show()
    plt.figure()

    #PCA Profile Plot
    
    variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma']}
    variables=pd.DataFrame(data=variables)
    PC0_values=np.column_stack([pca.components_[0,0],pca.components_[0,1],pca.components_[0,2],pca.components_[0,3],pca.components_[0,4],pca.components_[0,5]])
    PC1_values=np.column_stack([pca.components_[1,0],pca.components_[1,1],pca.components_[1,2],pca.components_[1,3],pca.components_[1,4], pca.components_[1,5]])
    PC2_values=np.column_stack([pca.components_[2,0],pca.components_[2,1],pca.components_[2,2],pca.components_[2,3],pca.components_[2,4],pca.components_[2,5]])


    
    plt.plot(variables.loc[0,:],PC0_values[0,:],label='PC0')
    plt.plot(variables.loc[0,:],PC1_values[0,:],label='PC1')
    plt.plot(variables.loc[0,:],PC2_values[0,:],label='PC2')
    plt.title('Component Pattern Profiles')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array


a=galaxy_profile_plot('7977-3704',3)
    

