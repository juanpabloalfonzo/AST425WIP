import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import pearsonr
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
from marvin import config
from marvin.tools.image import Image
from sklearn.decomposition import PCA
from sklearn.utils import resample

#set config attributes and turn on global downloads of Marvin data
config.setRelease('DR15')
config.mode = 'local'
config.download = True

plt.ion()

def galaxy_profile_plot_single(plateifu,Num_PCA_Vectors):

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    haflux = maps.emline_gflux_ha_6564
    values_flux = haflux.value
    ivar_flux = haflux.ivar
    mask_flux = haflux.mask
    #haflux.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_vel = maps.emline_gvel_ha_6564
    values_vel = ha_vel.value
    ivar_vel = ha_vel.ivar
    mask_vel = ha_vel.mask
    #ha_vel.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_sigma = maps.emline_sigma_ha_6564
    values_sigma = ha_sigma.value
    ivar_sigma = ha_sigma.ivar
    mask_sigma = ha_sigma.mask
    #ha_sigma.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_ew = maps.emline_gew_ha_6564
    values_ew = ha_vel.value
    ivar_ew = ha_vel.ivar
    mask_ew = ha_vel.mask
    #ha_ew.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    stellar_vel = maps.stellar_vel
    values_stellar_vel = stellar_vel.value
    ivar_stellar_vel = stellar_vel.ivar
    mask_stellar_vel = stellar_vel.mask
    #stellar_vel.plot()

    maps = Maps(plateifu=plateifu)
    
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
    
    for i in range(Num_PCA_Vectors):    
        plt.plot(variables.loc[0,:],pca.components_[i,:],label='PC'+str(i))
    plt.title('Component Pattern Profiles')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_PCA_single(plateifu,Num_PCA_Vectors):
    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    haflux = maps.emline_gflux_ha_6564
    values_flux = haflux.value
    ivar_flux = haflux.ivar
    mask_flux = haflux.mask
    #haflux.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_vel = maps.emline_gvel_ha_6564
    values_vel = ha_vel.value
    ivar_vel = ha_vel.ivar
    mask_vel = ha_vel.mask
    #ha_vel.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_sigma = maps.emline_sigma_ha_6564
    values_sigma = ha_sigma.value
    ivar_sigma = ha_sigma.ivar
    mask_sigma = ha_sigma.mask
    #ha_sigma.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_ew = maps.emline_gew_ha_6564
    values_ew = ha_vel.value
    ivar_ew = ha_vel.ivar
    mask_ew = ha_vel.mask
    #ha_ew.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    stellar_vel = maps.stellar_vel
    values_stellar_vel = stellar_vel.value
    ivar_stellar_vel = stellar_vel.ivar
    mask_stellar_vel = stellar_vel.mask
    #stellar_vel.plot()

    maps = Maps(plateifu=plateifu)
    
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
    

    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array
 
def galaxy_profile_plot_multi(plateifu,Num_PCA_Vectors,Num_Variables):
    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
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
        pca_components[:,:,i]=pca.components_
        pca_explained_variance_ratio_[i,:]=pca.explained_variance_ratio_
        
    return(pca_components,pca_explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_PCA_multi(plateifu,Num_PCA_Vectors,Num_Variables):
    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
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
        
        
        pca_components[:,:,i]=pca.components_
        pca_explained_variance_ratio_[i,:]=pca.explained_variance_ratio_
        
    return(pca_components,pca_explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot(plateifu,Num_PCA_Vectors,Num_Variables):
    if np.size(plateifu)<2:
        return(galaxy_profile_plot_single(plateifu,Num_PCA_Vectors))
    else:
        return(galaxy_profile_plot_multi(plateifu,Num_PCA_Vectors,Num_Variables))

def galaxy_PCA(plateifu,Num_PCA_Vectors,Num_Variables):
    if np.size(plateifu)<2:
        return(galaxy_PCA_single(plateifu,Num_PCA_Vectors))
    else:
        return(galaxy_PCA_multi(plateifu,Num_PCA_Vectors,Num_Variables))

def galaxy_profile_plot_combined(plateifu,Num_PCA_Vectors,Num_Variables):
    
    values_flux_combined=[]
    values_vel_combined=[]
    values_ew_combined=[]
    values_sigma_combined=[]
    values_stellar_vel_combined=[]
    values_stellar_sigma_combined=[]
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_656len4
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux_combined.append(values_flux.flatten())
        values_vel_combined.append(values_vel.flatten())
        values_ew_combined.append(values_ew.flatten())
        values_sigma_combined.append(values_sigma.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())
        values_stellar_sigma_combined.append(values_stellar_sigma.flatten())

    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_flux=np.zeros(len(plateifu))
    ha_vel=np.zeros(len(plateifu))
    ha_ew=np.zeros(len(plateifu))
    ha_sigma=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    stellar_sigma=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_flux[j_galaxy]=len(values_flux_combined[j_galaxy])
        ha_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        ha_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])
        ha_ew[j_galaxy]=len(values_ew_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        stellar_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_flux_combined1=np.zeros(int(sum(ha_flux)))
    values_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_ew_combined1=np.zeros(int(sum(ha_ew)))
    values_sigma_combined1=np.zeros(int(sum(ha_sigma)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    values_stellar_sigma_combined1=np.zeros(int(sum(stellar_sigma)))
    last_step=0
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()
        if i==0:
            d=0
            f=np.size(values_flux)
        else:
            d=last_step
            f=last_step+np.size(values_flux)

        values_flux_combined1[d:f]=values_flux.flatten()
        values_vel_combined1[d:f]=values_vel.flatten()
        values_ew_combined1[d:f]=values_ew.flatten()
        values_sigma_combined1[d:f]=(values_sigma.flatten())
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
        values_stellar_sigma_combined1[d:f]=values_stellar_sigma.flatten()
        
        last_step=last_step+np.size(values_flux)


    values=np.column_stack([values_flux_combined1,values_vel_combined1,values_ew_combined1,values_sigma_combined1,values_stellar_vel_combined1,values_stellar_sigma_combined1])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    

    x = np.arange(len(pca.explained_variance_ratio_))

    #PCA Screeplot
    plt.bar(x,pca.explained_variance_ratio_)
    pc_names=[]
    for a in range(len(pca.explained_variance_ratio_)):
        pc_names.append('PC'+str(a))
    plt.xticks(x,(pc_names))
    plt.title('Scree Plot ')
    plt.xlabel('Principal components')
    plt.ylabel('Variance Explained')
    plt.show()
    plt.figure()

    #PCA Profile Plot
    
    variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma']}
    variables=pd.DataFrame(data=variables)
    
    for b in range(Num_PCA_Vectors):    
        plt.plot(variables.loc[0,:],pca.components_[b,:],label='PC'+str(b))
    plt.title('Component Pattern Profiles ')
    plt.ylabel('Correlation')
    plt.xlabel('Variable')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot_global_single(plateifu,Num_PCA_Vectors, Num_Variables):
    
    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    haflux = maps.emline_gflux_ha_6564
    values_flux = haflux.value
    ivar_flux = haflux.ivar
    mask_flux = haflux.mask
    #haflux.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_vel = maps.emline_gvel_ha_6564
    values_vel = ha_vel.value
    ivar_vel = ha_vel.ivar
    mask_vel = ha_vel.mask
    #ha_vel.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_sigma = maps.emline_sigma_ha_6564
    values_sigma = ha_sigma.value
    ivar_sigma = ha_sigma.ivar
    mask_sigma = ha_sigma.mask
    #ha_sigma.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    ha_ew = maps.emline_gew_ha_6564
    values_ew = ha_vel.value
    ivar_ew = ha_vel.ivar
    mask_ew = ha_vel.mask
    #ha_ew.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    stellar_vel = maps.stellar_vel
    values_stellar_vel = stellar_vel.value
    ivar_stellar_vel = stellar_vel.ivar
    mask_stellar_vel = stellar_vel.mask
    #stellar_vel.plot()

    maps = Maps(plateifu=plateifu)
    
    # get an emission line map
    stellar_sigma = maps.stellar_sigma
    values_stellar_sigma = stellar_sigma.value
    ivar_stellar_sigma = stellar_sigma.ivar
    mask_stellar_sigma = stellar_sigma.mask
    #stellar_vel.plot()

    location=np.where(data.loc[:,'plateifu']==plateifu) #Find index of galaxy of in schema table to look up global properties
    
    mass=float(data.loc[location[0],'nsa_sersic_mass'])
    mass=mass*np.ones(np.size(stellar_vel))
    
    sfr=float(data.loc[location[0],'sfr_tot'])
    sfr=sfr*np.ones(np.size(stellar_vel))
   
    values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten(),mass,sfr])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    

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
    
    variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma'], 'col7':['Mass'], 'col8':['SFR']}
    variables=pd.DataFrame(data=variables)
    
    for i in range(Num_PCA_Vectors):    
        plt.plot(variables.loc[0,:],pca.components_[i,:],label='PC'+str(i))
    plt.title('Component Pattern Profiles')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot_global_multi(plateifu,Num_PCA_Vectors, Num_Variables):
    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        location=np.where(data.loc[:,'plateifu']==plateifu.iloc[i]) #Find index of galaxy of in schema table to look up global properties
        
        mass=float(data.loc[location[0],'nsa_sersic_mass'])
        mass=mass*np.ones(np.size(stellar_vel))
        
        sfr=float(data.loc[location[0],'sfr_tot'])
        sfr=sfr*np.ones(np.size(stellar_vel))
    
        values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten(),mass,sfr])
        values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
        pca = PCA(n_components=Num_PCA_Vectors)
        principalComponents = pca.fit_transform(values)
        
            
        x = np.arange(len(pca.explained_variance_ratio_))

        #PCA Screeplot
        plt.bar(x,pca.explained_variance_ratio_)
        pc_names=[]
        for i_var in range(len(pca.explained_variance_ratio_)):
            pc_names.append('PC'+str(i_var))
        plt.xticks(x,(pc_names))
        plt.title('Scree Plot '+str(plateifu.iloc[i]))
        plt.xlabel('Principal components')
        plt.ylabel('Variance Explained')
        plt.show()
        plt.figure()

        #PCA Profile Plot
        
        variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma'], 'col7':['Mass'], 'col8':['SFR']}
        variables=pd.DataFrame(data=variables)
        
        for i_pc in range(Num_PCA_Vectors):    
            plt.plot(variables.loc[0,:],pca.components_[i_pc,:],label='PC'+str(i_pc))
        plt.title('Component Pattern Profiles '+str(plateifu.iloc[i]))
        plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
        plt.legend()
        plt.show()
        plt.figure()
        pca_components[:,:,i]=pca.components_
        pca_explained_variance_ratio_[i,:]=pca.explained_variance_ratio_
    return(pca_components,pca_explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot_global(plateifu,Num_PCA_Vectors,Num_Variables):
    if np.size(plateifu)<2:
        return(galaxy_profile_plot_global_single(plateifu,Num_PCA_Vectors))
    else:
        return(galaxy_profile_plot_global_multi(plateifu,Num_PCA_Vectors,Num_Variables))

def galaxy_profile_plot_global_combined(plateifu,Num_PCA_Vectors,Num_Variables):
    
    values_flux_combined=[]
    values_vel_combined=[]
    values_ew_combined=[]
    values_sigma_combined=[]
    values_stellar_vel_combined=[]
    values_stellar_sigma_combined=[]
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_656len4
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux_combined.append(values_flux.flatten())
        values_vel_combined.append(values_vel.flatten())
        values_ew_combined.append(values_ew.flatten())
        values_sigma_combined.append(values_sigma.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())
        values_stellar_sigma_combined.append(values_stellar_sigma.flatten())
        

    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_flux=np.zeros(len(plateifu))
    ha_vel=np.zeros(len(plateifu))
    ha_ew=np.zeros(len(plateifu))
    ha_sigma=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    stellar_sigma=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_flux[j_galaxy]=len(values_flux_combined[j_galaxy])
        ha_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        ha_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])
        ha_ew[j_galaxy]=len(values_ew_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        stellar_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_flux_combined1=np.zeros(int(sum(ha_flux)))
    values_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_ew_combined1=np.zeros(int(sum(ha_ew)))
    values_sigma_combined1=np.zeros(int(sum(ha_sigma)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    values_stellar_sigma_combined1=np.zeros(int(sum(stellar_sigma)))
    mass_combined=np.zeros(int(sum(stellar_sigma)))
    sfr_combined=np.zeros(int(sum(stellar_sigma)))
    last_step=0
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()
        if i==0:
            d=0
            f=np.size(values_flux)
        else:
            d=last_step
            f=last_step+np.size(values_flux)

        values_flux_combined1[d:f]=values_flux.flatten()
        values_vel_combined1[d:f]=values_vel.flatten()
        values_ew_combined1[d:f]=values_ew.flatten()
        values_sigma_combined1[d:f]=(values_sigma.flatten())
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
        values_stellar_sigma_combined1[d:f]=values_stellar_sigma.flatten()
        
        location=np.where(data.loc[:,'plateifu']==plateifu.iloc[i]) #Find index of galaxy of in schema table to look up global properties
        mass_combined[d:f]=float(data.loc[location[0],'nsa_sersic_mass'])
        sfr_combined[d:f]=float(data.loc[location[0],'sfr_tot'])
    
        last_step=last_step+np.size(values_flux)


    values=np.column_stack([values_flux_combined1,values_vel_combined1,values_ew_combined1,values_sigma_combined1,values_stellar_vel_combined1,values_stellar_sigma_combined1,mass_combined,sfr_combined])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    

    x = np.arange(len(pca.explained_variance_ratio_))

    #PCA Screeplot
    plt.bar(x,pca.explained_variance_ratio_)
    pc_names=[]
    for a in range(len(pca.explained_variance_ratio_)):
        pc_names.append('PC'+str(a))
    plt.xticks(x,(pc_names))
    plt.title('Scree Plot ')
    plt.xlabel('Principal components')
    plt.ylabel('Variance Explained')
    plt.show()
    plt.figure()

    #PCA Profile Plot
    
    variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma'], 'col7':['Mass'], 'col8':['SFR']}
    variables=pd.DataFrame(data=variables)
    
    for b in range(Num_PCA_Vectors):    
        plt.plot(variables.loc[0,:],pca.components_[b,:],label='PC'+str(b))
    plt.title('Component Pattern Profiles ')
    plt.ylabel('Correlation')
    plt.xlabel('Variable')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

#Only works with multiple galaxies at once
def bootstrap_galaxy(plateifu, Num_PCA_Vectors, Num_Variables,reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,np.size(plateifu),reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        a=resample(plateifu)
        b=galaxy_PCA(a,Num_PCA_Vectors,Num_Variables)
        PC_Vector_Components[:,:,:,i_reps]=b[0]  

   
    # #Ensure the eigen values of the PC vectors are all of the same sign to avoid pseudo noise from anti-parallel vectors
    # for i_pc in range(Num_PCA_Vectors):
    #     for i_reps in range(reps):
    #         reference_vector=PC_Vector_Components[i_pc,:,:,0] #Decides what is parallel and anti parallel (choice is arbritary)
    #         vect1=PC_Vector_Components[i_pc,:,:,i_reps]
    #         if pearsonr(reference_vector,vect1)[0]<0: #Not testing for perfectly anti parallel vectors but "close enough"
    #             vect1=-1*vect1
    #         PC_Vector_Components[i_pc,:,:,i_reps]=vect1


    PC_Errors_STD=np.zeros([Num_PCA_Vectors,Num_Variables])
    for i_variables in range(Num_Variables):
        for i_pc in range(Num_PCA_Vectors):
            PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:,:])

    return(PC_Errors_STD)

def bootstrap_maps_single(plateifu, Num_PCA_Vectors, Num_Variables, reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux=resample(values_flux)
        values_vel=resample(values_vel)
        values_ew=resample(values_ew)
        values_sigma=resample(values_sigma)
        values_stellar_vel=resample(values_stellar_vel)
        values_stellar_sigma=resample(values_stellar_sigma)

    
        values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten()])
        values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
        pca = PCA(n_components=Num_PCA_Vectors)
        principalComponents = pca.fit_transform(values)
        
        
        PC_Vector_Components[:,:,i_reps]=pca.components_  

    for i_pc in range(Num_PCA_Vectors):
        for i_reps in range(reps):
            reference_vector=PC_Vector_Components[i_pc,:,0] #Decides what is parallel and anti parallel (choice is arbritary)
            vect1=PC_Vector_Components[i_pc,:,i_reps]
            if pearsonr(reference_vector,vect1)[0]<0: #Not testing for perfectly anti parallel vectors but "close enough"
                vect1=-1*vect1
            PC_Vector_Components[i_pc,:,i_reps]=vect1


    PC_Errors_STD=np.zeros([Num_PCA_Vectors,Num_Variables])
    for i_variables in range(Num_Variables):
        for i_pc in range(Num_PCA_Vectors):
            PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:])


    return(PC_Errors_STD)

def bootstrap_maps_multi(plateifu, Num_PCA_Vectors, Num_Variables, reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        plateifu_random=np.random.choice(plateifu)
        maps = Maps(plateifu=plateifu_random)
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu_random)
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu_random)
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu_random)
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu_random)
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu_random)
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux=resample(values_flux)
        values_vel=resample(values_vel)
        values_ew=resample(values_ew)
        values_sigma=resample(values_sigma)
        values_stellar_vel=resample(values_stellar_vel)
        values_stellar_sigma=resample(values_stellar_sigma)


        values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten()])
        values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
        pca = PCA(n_components=Num_PCA_Vectors)
        principalComponents = pca.fit_transform(values)
        
        PC_Vector_Components[:,:,i_reps]=pca.components_  


    for i_pc in range(Num_PCA_Vectors):
        for i_reps in range(reps):
            reference_vector=PC_Vector_Components[i_pc,:,0] #Decides what is parallel and anti parallel (choice is arbritary)
            vect1=PC_Vector_Components[i_pc,:,i_reps]
            if pearsonr(reference_vector,vect1)[0]<0: #Not testing for perfectly anti parallel vectors but "close enough"
                vect1=-1*vect1
            PC_Vector_Components[i_pc,:,i_reps]=vect1

    PC_Errors_STD=np.zeros([Num_PCA_Vectors,Num_Variables])
    for i_variables in range(Num_Variables):
        for i_pc in range(Num_PCA_Vectors):
            PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:])

    return(PC_Errors_STD)

def bootstrap_maps(plateifu, Num_PCA_Vectors, Num_Variables, reps):
    if np.size(plateifu)<2:
        return(bootstrap_maps_single(plateifu,Num_PCA_Vectors,Num_Variables,reps))
    else:
        return(bootstrap_maps_multi(plateifu,Num_PCA_Vectors,Num_Variables,reps))

#Only works with multiple galaxies at once
def galaxy_error_bar(plateifu,Num_PCA_Vectors,Num_Variables,reps):

    g=bootstrap_galaxy(plateifu,Num_PCA_Vectors,Num_Variables,reps)

    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
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
            plt.errorbar(variables.loc[0,:],pca.components_[b,:],yerr=g[b,:],label='PC'+str(b))
        plt.title('Component Pattern Profiles '+ str(plateifu.iloc[i]))
        plt.ylabel('Correlation')
        plt.xlabel('Variable')
        plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
        plt.legend()
        plt.show()
        plt.figure()
        pca_components[:,:,i]=pca.components_
        pca_explained_variance_ratio_[i,:]=pca.explained_variance_ratio_
        
    return(pca_components,pca_explained_variance_ratio_,g) #Returns PCA vector components, Variance ratios as an array

def maps_error_bar(plateifu,Num_PCA_Vectors,Num_Variables,reps):

    g=bootstrap_maps(plateifu,Num_PCA_Vectors,Num_Variables,reps)

    if np.size(plateifu)<2:
        pca_components=np.zeros([Num_PCA_Vectors,Num_Variables])
        pca_explained_variance_ratio_=np.zeros([Num_PCA_Vectors])
        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu)
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu)
        
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
        

        x = np.arange(len(pca.explained_variance_ratio_))

        #PCA Screeplot
        plt.bar(x,pca.explained_variance_ratio_)
        pc_names=[]
        for a in range(len(pca.explained_variance_ratio_)):
            pc_names.append('PC'+str(a))
        plt.xticks(x,(pc_names))
        plt.title('Scree Plot '+str(plateifu))
        plt.xlabel('Principal components')
        plt.ylabel('Variance Explained')
        plt.show()
        plt.figure()

        #PCA Profile Plot
        
        variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma']}
        variables=pd.DataFrame(data=variables)
        
        for b in range(Num_PCA_Vectors):    
            plt.errorbar(variables.loc[0,:],pca.components_[b,:],yerr=g[b,:],label='PC'+str(b))
        plt.title('Component Pattern Profiles '+ str(plateifu))
        plt.ylabel('Correlation')
        plt.xlabel('Variable')
        plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
        plt.legend()
        plt.show()
        plt.figure()
        pca_components[:,:]=pca.components_
        pca_explained_variance_ratio_[:]=pca.explained_variance_ratio_
    else:
        pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,np.size(plateifu)])
        pca_explained_variance_ratio_=np.zeros([np.size(plateifu),Num_PCA_Vectors])
        for i in range(np.size(plateifu)):
            maps = Maps(plateifu=plateifu.iloc[i])
            
            # get an emission line map
            haflux = maps.emline_gflux_ha_6564
            values_flux = haflux.value
            ivar_flux = haflux.ivar
            mask_flux = haflux.mask
            #haflux.plot()

            maps = Maps(plateifu=plateifu.iloc[i])
            
            # get an emission line map
            ha_vel = maps.emline_gvel_ha_6564
            values_vel = ha_vel.value
            ivar_vel = ha_vel.ivar
            mask_vel = ha_vel.mask
            #ha_vel.plot()

            maps = Maps(plateifu=plateifu.iloc[i])
            
            # get an emission line map
            ha_sigma = maps.emline_sigma_ha_6564
            values_sigma = ha_sigma.value
            ivar_sigma = ha_sigma.ivar
            mask_sigma = ha_sigma.mask
            #ha_sigma.plot()

            maps = Maps(plateifu=plateifu.iloc[i])
            
            # get an emission line map
            ha_ew = maps.emline_gew_ha_6564
            values_ew = ha_vel.value
            ivar_ew = ha_vel.ivar
            mask_ew = ha_vel.mask
            #ha_ew.plot()

            maps = Maps(plateifu=plateifu.iloc[i])
            
            # get an emission line map
            stellar_vel = maps.stellar_vel
            values_stellar_vel = stellar_vel.value
            ivar_stellar_vel = stellar_vel.ivar
            mask_stellar_vel = stellar_vel.mask
            #stellar_vel.plot()

            maps = Maps(plateifu=plateifu.iloc[i])
            
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
                plt.errorbar(variables.loc[0,:],pca.components_[b,:],yerr=g[b,:],label='PC'+str(b))
            plt.title('Component Pattern Profiles '+ str(plateifu.iloc[i]))
            plt.ylabel('Correlation')
            plt.xlabel('Variable')
            plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
            plt.legend()
            plt.show()
            plt.figure()
            pca_components[:,:,i]=pca.components_
            pca_explained_variance_ratio_[i,:]=pca.explained_variance_ratio_
        
    return(pca_components,pca_explained_variance_ratio_,g) #Returns PCA vector components, Variance ratios as an array

def galaxy_PCA_combined(plateifu,Num_PCA_Vectors,Num_Variables):
    
    values_flux_combined=[]
    values_vel_combined=[]
    values_ew_combined=[]
    values_sigma_combined=[]
    values_stellar_vel_combined=[]
    values_stellar_sigma_combined=[]
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_656len4
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux_combined.append(values_flux.flatten())
        values_vel_combined.append(values_vel.flatten())
        values_ew_combined.append(values_ew.flatten())
        values_sigma_combined.append(values_sigma.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())
        values_stellar_sigma_combined.append(values_stellar_sigma.flatten())
        

    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_flux=np.zeros(len(plateifu))
    ha_vel=np.zeros(len(plateifu))
    ha_ew=np.zeros(len(plateifu))
    ha_sigma=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    stellar_sigma=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_flux[j_galaxy]=len(values_flux_combined[j_galaxy])
        ha_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        ha_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])
        ha_ew[j_galaxy]=len(values_ew_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        stellar_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_flux_combined1=np.zeros(int(sum(ha_flux)))
    values_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_ew_combined1=np.zeros(int(sum(ha_ew)))
    values_sigma_combined1=np.zeros(int(sum(ha_sigma)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    values_stellar_sigma_combined1=np.zeros(int(sum(stellar_sigma)))
    mass_combined=np.zeros(int(sum(stellar_sigma)))
    sfr_combined=np.zeros(int(sum(stellar_sigma)))
    last_step=0
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()
        if i==0:
            d=0
            f=np.size(values_flux)
        else:
            d=last_step
            f=last_step+np.size(values_flux)

        values_flux_combined1[d:f]=values_flux.flatten()
        values_vel_combined1[d:f]=values_vel.flatten()
        values_ew_combined1[d:f]=values_ew.flatten()
        values_sigma_combined1[d:f]=(values_sigma.flatten())
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
        values_stellar_sigma_combined1[d:f]=values_stellar_sigma.flatten()
        
        location=np.where(data.loc[:,'plateifu']==plateifu.iloc[i]) #Find index of galaxy of in schema table to look up global properties
        mass_combined[d:f]=float(data.loc[location[0],'nsa_sersic_mass'])
        sfr_combined[d:f]=float(data.loc[location[0],'sfr_tot'])
    
        last_step=last_step+np.size(values_flux)


    values=np.column_stack([values_flux_combined1,values_vel_combined1,values_ew_combined1,values_sigma_combined1,values_stellar_vel_combined1,values_stellar_sigma_combined1,mass_combined,sfr_combined])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_PCA_combined_resampled(plateifu,Num_PCA_Vectors,Num_Variables):
    
    values_flux_combined=[]
    values_vel_combined=[]
    values_ew_combined=[]
    values_sigma_combined=[]
    values_stellar_vel_combined=[]
    values_stellar_sigma_combined=[]
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_656len4
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux_combined.append(values_flux.flatten())
        values_vel_combined.append(values_vel.flatten())
        values_ew_combined.append(values_ew.flatten())
        values_sigma_combined.append(values_sigma.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())
        values_stellar_sigma_combined.append(values_stellar_sigma.flatten())
        

    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_flux=np.zeros(len(plateifu))
    ha_vel=np.zeros(len(plateifu))
    ha_ew=np.zeros(len(plateifu))
    ha_sigma=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    stellar_sigma=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_flux[j_galaxy]=len(values_flux_combined[j_galaxy])
        ha_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        ha_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])
        ha_ew[j_galaxy]=len(values_ew_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        stellar_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_flux_combined1=np.zeros(int(sum(ha_flux)))
    values_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_ew_combined1=np.zeros(int(sum(ha_ew)))
    values_sigma_combined1=np.zeros(int(sum(ha_sigma)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    values_stellar_sigma_combined1=np.zeros(int(sum(stellar_sigma)))
    mass_combined=np.zeros(int(sum(stellar_sigma)))
    sfr_combined=np.zeros(int(sum(stellar_sigma)))
    last_step=0
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()
        if i==0:
            d=0
            f=np.size(values_flux)
        else:
            d=last_step
            f=last_step+np.size(values_flux)

        values_flux_combined1[d:f]=values_flux.flatten()
        values_vel_combined1[d:f]=values_vel.flatten()
        values_ew_combined1[d:f]=values_ew.flatten()
        values_sigma_combined1[d:f]=(values_sigma.flatten())
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
        values_stellar_sigma_combined1[d:f]=values_stellar_sigma.flatten()
        
        location=np.where(data.loc[:,'plateifu']==plateifu.iloc[i]) #Find index of galaxy of in schema table to look up global properties
        mass_combined[d:f]=float(data.loc[location[0],'nsa_sersic_mass'])
        sfr_combined[d:f]=float(data.loc[location[0],'sfr_tot'])
    
        last_step=last_step+np.size(values_flux)
    
    #Take the combined maps and resample before plugging into PCA 

    values_flux_combined1=resample(values_flux_combined1)
    values_vel_combined1=resample(values_vel_combined1)
    values_ew_combined1=resample(values_ew_combined1)
    values_sigma_combined1=resample(values_sigma_combined1)
    values_stellar_vel_combined1=resample(values_stellar_vel_combined1)
    values_stellar_sigma_combined1=resample(values_stellar_sigma_combined1)
    mass_combined=resample(mass_combined)
    sfr_combined=resample(sfr_combined)


    values=np.column_stack([values_flux_combined1,values_vel_combined1,values_ew_combined1,values_sigma_combined1,values_stellar_vel_combined1,values_stellar_sigma_combined1,mass_combined,sfr_combined])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def bootstrap_global(plateifu, Num_PCA_Vectors, Num_Variables, reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        b=galaxy_PCA_combined_resampled(plateifu,Num_PCA_Vectors,Num_Variables)
        PC_Vector_Components[:,:,i_reps]=b[0]  

    #Ensures all the eigenvalues of the PC vectors are of the same sign
    for i_pc in range(Num_PCA_Vectors):
        for i_reps in range(reps):
            reference_vector=PC_Vector_Components[i_pc,:,0] #Decides what is parallel and anti parallel (choice is arbritary)
            vect1=PC_Vector_Components[i_pc,:,i_reps]
            if pearsonr(reference_vector,vect1)[0]<0: #Not testing for perfectly anti parallel vectors but "close enough"
                vect1=-1*vect1
            PC_Vector_Components[i_pc,:,i_reps]=vect1
    
    PC_Errors_STD=np.zeros([Num_PCA_Vectors,Num_Variables])
    for i_variables in range(Num_Variables):
        for i_pc in range(Num_PCA_Vectors):
            PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:])

    return(PC_Errors_STD)

def combined_maps_error_bar(plateifu,Num_PCA_Vectors,Num_Variables,reps):

    g=bootstrap_global(plateifu,Num_PCA_Vectors,Num_Variables,reps)

    values_flux_combined=[]
    values_vel_combined=[]
    values_ew_combined=[]
    values_sigma_combined=[]
    values_stellar_vel_combined=[]
    values_stellar_sigma_combined=[]
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_656len4
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()

        values_flux_combined.append(values_flux.flatten())
        values_vel_combined.append(values_vel.flatten())
        values_ew_combined.append(values_ew.flatten())
        values_sigma_combined.append(values_sigma.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())
        values_stellar_sigma_combined.append(values_stellar_sigma.flatten())
        

    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_flux=np.zeros(len(plateifu))
    ha_vel=np.zeros(len(plateifu))
    ha_ew=np.zeros(len(plateifu))
    ha_sigma=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    stellar_sigma=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_flux[j_galaxy]=len(values_flux_combined[j_galaxy])
        ha_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        ha_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])
        ha_ew[j_galaxy]=len(values_ew_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])
        stellar_sigma[j_galaxy]=len(values_stellar_sigma_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_flux_combined1=np.zeros(int(sum(ha_flux)))
    values_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_ew_combined1=np.zeros(int(sum(ha_ew)))
    values_sigma_combined1=np.zeros(int(sum(ha_sigma)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    values_stellar_sigma_combined1=np.zeros(int(sum(stellar_sigma)))
    mass_combined=np.zeros(int(sum(stellar_sigma)))
    sfr_combined=np.zeros(int(sum(stellar_sigma)))
    last_step=0
    for i in range(len(plateifu)):
        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu.iloc[i])
        
        # get an emission line map
        stellar_sigma = maps.stellar_sigma
        values_stellar_sigma = stellar_sigma.value
        ivar_stellar_sigma = stellar_sigma.ivar
        mask_stellar_sigma = stellar_sigma.mask
        #stellar_vel.plot()
        if i==0:
            d=0
            f=np.size(values_flux)
        else:
            d=last_step
            f=last_step+np.size(values_flux)

        values_flux_combined1[d:f]=values_flux.flatten()
        values_vel_combined1[d:f]=values_vel.flatten()
        values_ew_combined1[d:f]=values_ew.flatten()
        values_sigma_combined1[d:f]=(values_sigma.flatten())
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
        values_stellar_sigma_combined1[d:f]=values_stellar_sigma.flatten()
        
        location=np.where(data.loc[:,'plateifu']==plateifu.iloc[i]) #Find index of galaxy of in schema table to look up global properties
        mass_combined[d:f]=float(data.loc[location[0],'nsa_sersic_mass'])
        sfr_combined[d:f]=float(data.loc[location[0],'sfr_tot'])
    
        last_step=last_step+np.size(values_flux)


    values=np.column_stack([values_flux_combined1,values_vel_combined1,values_ew_combined1,values_sigma_combined1,values_stellar_vel_combined1,values_stellar_sigma_combined1,mass_combined,sfr_combined])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)

    x = np.arange(len(pca.explained_variance_ratio_))

    #PCA Screeplot
    plt.bar(x,pca.explained_variance_ratio_)
    pc_names=[]
    for a in range(len(pca.explained_variance_ratio_)):
        pc_names.append('PC'+str(a))
    plt.xticks(x,(pc_names))
    plt.title('Scree Plot')
    plt.xlabel('Principal components')
    plt.ylabel('Variance Explained')
    plt.show()
    plt.figure()

    #PCA Profile Plot
    
    variables={'col1':['Ha Flux'], 'col2':['Ha Velocity'], 'col3':['Ha EW'], 'col4':['Ha Sigma'], 'col5':['Stellar Velocity'], 'col6':['Stellar Sigma'], 'col7':['Mass'], 'col8':['SFR']}
    variables=pd.DataFrame(data=variables)
    
    for b in range(Num_PCA_Vectors):    
        plt.errorbar(variables.loc[0,:],pca.components_[b,:],yerr=g[b,:],label='PC'+str(b))
    plt.title('Component Pattern Profiles')
    plt.ylabel('Correlation')
    plt.xlabel('Variable')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()

    return(pca.components_,pca.explained_variance_ratio_,g) #Returns PCA vector components, Variance ratios as an array

def galaxy_profile_plot_combined_vel(plateifu,Num_PCA_Vectors):
    
    values_ha_vel_combined=[]
    values_oii1_vel_combined=[]
    values_oii2_vel_combined=[]
    values_hthe_vel_combined=[]
    values_heta_vel_combined=[]
    values_neiii1_vel_combined=[]
    values_neiii2_vel_combined=[]
    values_hzet_vel_combined=[]
    values_heps_vel_combined=[]
    values_hdel_vel_combined=[]
    values_hgam_vel_combined=[]
    values_heii_vel_combined=[]
    values_hb_vel_combined=[]
    values_oiii1_vel_combined=[]
    values_oiii2_vel_combined=[]
    values_hei_vel_combined=[]
    values_oi1_vel_combined=[]
    values_oi2_vel_combined=[]
    values_nii1_vel_combined=[]
    values_nii2_vel_combined=[]
    values_sii1_vel_combined=[]
    values_sii2_vel_combined=[]
    values_stellar_vel_combined=[]
    for i in range(len(plateifu)):
        #H Alpha
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_ha_vel = ha_vel.value
        ivar_ha_vel = ha_vel.ivar
        mask_ha_vel = ha_vel.mask

        #OII 3727
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii1_vel = maps.emline_gvel_oii_3727
        values_oii1_vel = oii1_vel.value
        ivar_oii1_vel = oii1_vel.ivar
        mask_oii1_vel = oii1_vel.mask

        #OII 3729
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii2_vel = maps.emline_gvel_oii_3729
        values_oii2_vel = oii2_vel.value
        ivar_oii2_vel = oii2_vel.ivar
        mask_oii2_vel = oii2_vel.mask

        #H Theta 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hthe_vel = maps.emline_gvel_hthe_3798
        values_hthe_vel = hthe_vel.value
        ivar_hthe_vel = hthe_vel.ivar
        mask_hthe_vel = hthe_vel.mask

       #H Eta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heta_vel = maps.emline_gvel_heta_3836
        values_heta_vel = heta_vel.value
        ivar_heta_vel = heta_vel.ivar
        mask_heta_vel = heta_vel.mask

        #Ne III 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii1_vel = maps.emline_gvel_neiii1_3869
        values_neiii1_vel = neiii1_vel.value
        ivar_neiii1_vel = neiii1_vel.ivar
        mask_neiii1_vel = neiii1_vel.mask

        #Ne III 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii2_vel = maps.emline_gvel_neiii2_3968
        values_neiii2_vel = neiii2_vel.value
        ivar_neiii2_vel = neiii2_vel.ivar
        mask_neiii2_vel = neiii2_vel.mask

        #H Zeta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hzet_vel = maps.emline_gvel_hzet_3890
        values_hzet_vel = hzet_vel.value
        ivar_hzet_vel = hzet_vel.ivar
        mask_hzet_vel = hzet_vel.mask

        #H Episilon 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heps_vel = maps.emline_gvel_heps_3971
        values_heps_vel = heps_vel.value
        ivar_heps_vel = heps_vel.ivar
        mask_heps_vel = heps_vel.mask

        #H Delta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hdel_vel = maps.emline_gvel_hdel_4102
        values_hdel_vel = hdel_vel.value
        ivar_hdel_vel = hdel_vel.ivar
        mask_hdel_vel = hdel_vel.mask

        #H Gamma
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hgam_vel = maps.emline_gvel_hgam_4341
        values_hgam_vel = hgam_vel.value
        ivar_hgam_vel = hgam_vel.ivar
        mask_hgam_vel = hgam_vel.mask

        #He II 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heii_vel = maps.emline_gvel_heii_4687
        values_heii_vel = heii_vel.value
        ivar_heii_vel = heii_vel.ivar
        mask_heii_vel = heii_vel.mask

        #H Beta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hb_vel = maps.emline_gvel_hb_4862
        values_hb_vel = hb_vel.value
        ivar_hb_vel = hb_vel.ivar
        mask_hb_vel = hb_vel.mask

        #OIII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii1_vel = maps.emline_gvel_oiii1_4960
        values_oiii1_vel = oiii1_vel.value
        ivar_oiii1_vel = oiii1_vel.ivar
        mask_oiii1_vel = oiii1_vel.mask

        #OIII 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii2_vel = maps.emline_gvel_oiii2_5008
        values_oiii2_vel = oiii2_vel.value
        ivar_oiii2_vel = oiii2_vel.ivar
        mask_oiii2_vel = oiii2_vel.mask

        #He I
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hei_vel = maps.emline_gvel_hei_5877
        values_hei_vel = hei_vel.value
        ivar_hei_vel = hei_vel.ivar
        mask_hei_vel = hei_vel.mask

        #OI 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi1_vel = maps.emline_gvel_oi1_6302
        values_oi1_vel = oi1_vel.value
        ivar_oi1_vel = oi1_vel.ivar
        mask_oi1_vel = oi1_vel.mask

        #OI 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi2_vel = maps.emline_gvel_oi2_6365
        values_oi2_vel = oi2_vel.value
        ivar_oi2_vel = oi2_vel.ivar
        mask_oi2_vel = oi2_vel.mask

        #N II 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii1_vel = maps.emline_gvel_nii1_6549
        values_nii1_vel = nii1_vel.value
        ivar_nii1_vel = nii1_vel.ivar
        mask_nii1_vel = nii1_vel.mask

        #N II 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii2_vel = maps.emline_gvel_nii2_6585
        values_nii2_vel = nii2_vel.value
        ivar_nii2_vel = nii2_vel.ivar
        mask_nii2_vel = nii2_vel.mask

        #SII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii1_vel = maps.emline_gvel_sii1_6718
        values_sii1_vel = sii1_vel.value
        ivar_sii1_vel = sii1_vel.ivar
        mask_sii1_vel = sii1_vel.mask

        #SII 2 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii2_vel = maps.emline_gvel_sii2_6732
        values_sii2_vel = sii2_vel.value
        ivar_sii2_vel = sii2_vel.ivar
        mask_sii2_vel = sii2_vel.mask

        #Stellar Vel 
        maps = Maps(plateifu=plateifu.iloc[i])
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask


        values_ha_vel_combined.append(values_ha_vel.flatten())
        values_oii1_vel_combined.append(values_oii1_vel.flatten())
        values_oii2_vel_combined.append(values_oii2_vel.flatten())
        values_hthe_vel_combined.append(values_hthe_vel.flatten())
        values_heta_vel_combined.append(values_heta_vel.flatten())
        values_neiii1_vel_combined.append(values_neiii1_vel.flatten())
        values_neiii2_vel_combined.append(values_neiii2_vel.flatten())
        values_hzet_vel_combined.append(values_hzet_vel.flatten())
        values_heps_vel_combined.append(values_heps_vel.flatten())
        values_hdel_vel_combined.append(values_hdel_vel.flatten())
        values_hgam_vel_combined.append(values_hgam_vel.flatten())
        values_heii_vel_combined.append(values_heii_vel.flatten())
        values_hb_vel_combined.append(values_hb_vel.flatten())
        values_oiii1_vel_combined.append(values_oiii1_vel.flatten())
        values_oiii2_vel_combined.append(values_oiii2_vel.flatten())
        values_hei_vel_combined.append(values_hei_vel.flatten())
        values_oi1_vel_combined.append(values_oi1_vel.flatten())
        values_oi2_vel_combined.append(values_oi2_vel.flatten())
        values_nii1_vel_combined.append(values_nii1_vel.flatten())
        values_nii2_vel_combined.append(values_nii2_vel.flatten())
        values_sii1_vel_combined.append(values_sii1_vel.flatten())
        values_sii2_vel_combined.append(values_sii2_vel.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())


    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_vel=np.zeros(len(plateifu))
    oii1_vel=np.zeros(len(plateifu))
    oii2_vel=np.zeros(len(plateifu))
    hthe_vel=np.zeros(len(plateifu))
    heta_vel=np.zeros(len(plateifu))
    neiii1_vel=np.zeros(len(plateifu))
    neiii2_vel=np.zeros(len(plateifu))
    hzet_vel=np.zeros(len(plateifu))
    heps_vel=np.zeros(len(plateifu))
    hdel_vel=np.zeros(len(plateifu))
    hgam_vel=np.zeros(len(plateifu))
    heii_vel=np.zeros(len(plateifu))
    hb_vel=np.zeros(len(plateifu))
    oiii1_vel=np.zeros(len(plateifu))
    oiii2_vel=np.zeros(len(plateifu))
    hei_vel=np.zeros(len(plateifu))
    oi1_vel=np.zeros(len(plateifu))
    oi2_vel=np.zeros(len(plateifu))
    nii1_vel=np.zeros(len(plateifu))
    nii2_vel=np.zeros(len(plateifu))
    sii1_vel=np.zeros(len(plateifu))
    sii2_vel=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_vel[j_galaxy]=len(values_ha_vel_combined[j_galaxy])
        oii1_vel[j_galaxy]=len(values_oii1_vel_combined[j_galaxy])
        oii2_vel[j_galaxy]=len(values_oii2_vel_combined[j_galaxy])
        hthe_vel[j_galaxy]=len(values_hthe_vel_combined[j_galaxy])
        heta_vel[j_galaxy]=len(values_heta_vel_combined[j_galaxy])
        neiii1_vel[j_galaxy]=len(values_neiii1_vel_combined[j_galaxy])
        neiii2_vel[j_galaxy]=len(values_neiii2_vel_combined[j_galaxy])
        hzet_vel[j_galaxy]=len(values_hzet_vel_combined[j_galaxy])
        heps_vel[j_galaxy]=len(values_heps_vel_combined[j_galaxy])
        hdel_vel[j_galaxy]=len(values_hdel_vel_combined[j_galaxy])
        hgam_vel[j_galaxy]=len(values_hgam_vel_combined[j_galaxy])
        heii_vel[j_galaxy]=len(values_heii_vel_combined[j_galaxy])
        hb_vel[j_galaxy]=len(values_hb_vel_combined[j_galaxy])
        oiii1_vel[j_galaxy]=len(values_oiii1_vel_combined[j_galaxy])
        oiii2_vel[j_galaxy]=len(values_oiii2_vel_combined[j_galaxy])
        hei_vel[j_galaxy]=len(values_hei_vel_combined[j_galaxy])
        oi1_vel[j_galaxy]=len(values_oi1_vel_combined[j_galaxy])
        oi2_vel[j_galaxy]=len(values_oi2_vel_combined[j_galaxy])
        nii1_vel[j_galaxy]=len(values_nii1_vel_combined[j_galaxy])
        nii2_vel[j_galaxy]=len(values_nii2_vel_combined[j_galaxy])
        sii1_vel[j_galaxy]=len(values_sii1_vel_combined[j_galaxy])
        sii2_vel[j_galaxy]=len(values_sii2_vel_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_ha_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_oii1_vel_combined1=np.zeros(int(sum(oii1_vel)))
    values_oii2_vel_combined1=np.zeros(int(sum(oii2_vel)))
    values_hthe_vel_combined1=np.zeros(int(sum(hthe_vel)))
    values_heta_vel_combined1=np.zeros(int(sum(heta_vel)))
    values_neiii1_vel_combined1=np.zeros(int(sum(neiii1_vel)))
    values_neiii2_vel_combined1=np.zeros(int(sum(neiii2_vel)))
    values_hzet_vel_combined1=np.zeros(int(sum(hzet_vel)))
    values_heps_vel_combined1=np.zeros(int(sum(heps_vel)))
    values_hdel_vel_combined1=np.zeros(int(sum(hdel_vel)))
    values_hgam_vel_combined1=np.zeros(int(sum(hgam_vel)))
    values_heii_vel_combined1=np.zeros(int(sum(heii_vel)))
    values_hb_vel_combined1=np.zeros(int(sum(hb_vel)))
    values_oiii1_vel_combined1=np.zeros(int(sum(oiii1_vel)))
    values_oiii2_vel_combined1=np.zeros(int(sum(oiii2_vel)))
    values_hei_vel_combined1=np.zeros(int(sum(hei_vel)))
    values_oi1_vel_combined1=np.zeros(int(sum(oi1_vel)))
    values_oi2_vel_combined1=np.zeros(int(sum(oi2_vel)))
    values_nii1_vel_combined1=np.zeros(int(sum(nii1_vel)))
    values_nii2_vel_combined1=np.zeros(int(sum(nii2_vel)))
    values_sii1_vel_combined1=np.zeros(int(sum(sii1_vel)))
    values_sii2_vel_combined1=np.zeros(int(sum(sii2_vel)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    last_step=0
    for i in range(len(plateifu)):
        #H Alpha
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_ha_vel = ha_vel.value
        ivar_ha_vel = ha_vel.ivar
        mask_ha_vel = ha_vel.mask

        #OII 3727
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii1_vel = maps.emline_gvel_oii_3727
        values_oii1_vel = oii1_vel.value
        ivar_oii1_vel = oii1_vel.ivar
        mask_oii1_vel = oii1_vel.mask

        #OII 3729
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii2_vel = maps.emline_gvel_oii_3729
        values_oii2_vel = oii2_vel.value
        ivar_oii2_vel = oii2_vel.ivar
        mask_oii2_vel = oii2_vel.mask

        #H Theta 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hthe_vel = maps.emline_gvel_hthe_3798
        values_hthe_vel = hthe_vel.value
        ivar_hthe_vel = hthe_vel.ivar
        mask_hthe_vel = hthe_vel.mask

       #H Eta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heta_vel = maps.emline_gvel_heta_3836
        values_heta_vel = heta_vel.value
        ivar_heta_vel = heta_vel.ivar
        mask_heta_vel = heta_vel.mask

        #Ne III 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii1_vel = maps.emline_gvel_neiii1_3869
        values_neiii1_vel = neiii1_vel.value
        ivar_neiii1_vel = neiii1_vel.ivar
        mask_neiii1_vel = neiii1_vel.mask

        #Ne III 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii2_vel = maps.emline_gvel_neiii2_3968
        values_neiii2_vel = neiii2_vel.value
        ivar_neiii2_vel = neiii2_vel.ivar
        mask_neiii2_vel = neiii2_vel.mask

        #H Zeta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hzet_vel = maps.emline_gvel_hzet_3890
        values_hzet_vel = hzet_vel.value
        ivar_hzet_vel = hzet_vel.ivar
        mask_hzet_vel = hzet_vel.mask

        #H Episilon 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heps_vel = maps.emline_gvel_heps_3971
        values_heps_vel = heps_vel.value
        ivar_heps_vel = heps_vel.ivar
        mask_heps_vel = heps_vel.mask

        #H Delta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hdel_vel = maps.emline_gvel_hdel_4102
        values_hdel_vel = hdel_vel.value
        ivar_hdel_vel = hdel_vel.ivar
        mask_hdel_vel = hdel_vel.mask

        #H Gamma
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hgam_vel = maps.emline_gvel_hgam_4341
        values_hgam_vel = hgam_vel.value
        ivar_hgam_vel = hgam_vel.ivar
        mask_hgam_vel = hgam_vel.mask

        #He II 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heii_vel = maps.emline_gvel_heii_4687
        values_heii_vel = heii_vel.value
        ivar_heii_vel = heii_vel.ivar
        mask_heii_vel = heii_vel.mask

        #H Beta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hb_vel = maps.emline_gvel_hb_4862
        values_hb_vel = hb_vel.value
        ivar_hb_vel = hb_vel.ivar
        mask_hb_vel = hb_vel.mask

        #OIII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii1_vel = maps.emline_gvel_oiii1_4960
        values_oiii1_vel = oiii1_vel.value
        ivar_oiii1_vel = oiii1_vel.ivar
        mask_oiii1_vel = oiii1_vel.mask

        #OIII 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii2_vel = maps.emline_gvel_oiii2_5008
        values_oiii2_vel = oiii2_vel.value
        ivar_oiii2_vel = oiii2_vel.ivar
        mask_oiii2_vel = oiii2_vel.mask

        #He I
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hei_vel = maps.emline_gvel_hei_5877
        values_hei_vel = hei_vel.value
        ivar_hei_vel = hei_vel.ivar
        mask_hei_vel = hei_vel.mask

        #OI 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi1_vel = maps.emline_gvel_oi1_6302
        values_oi1_vel = oi1_vel.value
        ivar_oi1_vel = oi1_vel.ivar
        mask_oi1_vel = oi1_vel.mask

        #OI 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi2_vel = maps.emline_gvel_oi2_6365
        values_oi2_vel = oi2_vel.value
        ivar_oi2_vel = oi2_vel.ivar
        mask_oi2_vel = oi2_vel.mask

        #N II 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii1_vel = maps.emline_gvel_nii1_6549
        values_nii1_vel = nii1_vel.value
        ivar_nii1_vel = nii1_vel.ivar
        mask_nii1_vel = nii1_vel.mask

        #N II 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii2_vel = maps.emline_gvel_nii2_6585
        values_nii2_vel = nii2_vel.value
        ivar_nii2_vel = nii2_vel.ivar
        mask_nii2_vel = nii2_vel.mask

        #SII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii1_vel = maps.emline_gvel_sii1_6718
        values_sii1_vel = sii1_vel.value
        ivar_sii1_vel = sii1_vel.ivar
        mask_sii1_vel = sii1_vel.mask

        #SII 2 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii2_vel = maps.emline_gvel_sii2_6732
        values_sii2_vel = sii2_vel.value
        ivar_sii2_vel = sii2_vel.ivar
        mask_sii2_vel = sii2_vel.mask

        #Stellar Vel 
        maps = Maps(plateifu=plateifu.iloc[i])
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        if i==0:
            d=0
            f=np.size(values_ha_vel)
        else:
            d=last_step
            f=last_step+np.size(values_ha_vel)

        values_ha_vel_combined1[d:f]=values_ha_vel.flatten()
        values_oii1_vel_combined1[d:f]=values_oii1_vel.flatten()
        values_oii2_vel_combined1[d:f]=values_oii2_vel.flatten()
        values_hthe_vel_combined1[d:f]=values_hthe_vel.flatten()
        values_heta_vel_combined1[d:f]=values_heta_vel.flatten()
        values_neiii1_vel_combined1[d:f]=values_neiii1_vel.flatten()
        values_neiii2_vel_combined1[d:f]=values_neiii2_vel.flatten()
        values_hzet_vel_combined1[d:f]=values_hzet_vel.flatten()
        values_heps_vel_combined1[d:f]=values_heps_vel.flatten()
        values_hdel_vel_combined1[d:f]=values_hdel_vel.flatten()
        values_hgam_vel_combined1[d:f]=values_hgam_vel.flatten()
        values_heii_vel_combined1[d:f]=values_heii_vel.flatten()
        values_hb_vel_combined1[d:f]=values_hb_vel.flatten()
        values_oiii1_vel_combined1[d:f]=values_oiii1_vel.flatten()
        values_oiii2_vel_combined1[d:f]=values_oiii2_vel.flatten()
        values_hei_vel_combined1[d:f]=values_hei_vel.flatten()
        values_oi1_vel_combined1[d:f]=values_oi1_vel.flatten()
        values_oi2_vel_combined1[d:f]=values_oi2_vel.flatten()
        values_nii1_vel_combined1[d:f]=values_nii1_vel.flatten()
        values_nii2_vel_combined1[d:f]=values_nii2_vel.flatten()
        values_sii1_vel_combined1[d:f]=values_sii1_vel.flatten()
        values_sii2_vel_combined1[d:f]=values_sii2_vel.flatten()
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
       
        
        last_step=last_step+np.size(values_ha_vel)


    values=np.column_stack([values_ha_vel_combined1, values_oii1_vel_combined1,values_oii2_vel_combined1, values_hthe_vel_combined1, values_heta_vel_combined1,  values_neiii1_vel_combined1, values_neiii2_vel_combined1, values_hzet_vel_combined1, values_heps_vel_combined1, values_hdel_vel_combined1, values_hgam_vel_combined1, values_heii_vel_combined1, values_hb_vel_combined1, values_oiii1_vel_combined1, values_oiii2_vel_combined1, values_hei_vel_combined1, values_oi1_vel_combined1, values_oi2_vel_combined1, values_nii1_vel_combined1, values_nii2_vel_combined1, values_sii1_vel_combined1, values_sii2_vel_combined1, values_stellar_vel_combined1])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    

    x = np.arange(len(pca.explained_variance_ratio_))

    #PCA Screeplot
    plt.bar(x,pca.explained_variance_ratio_)
    pc_names=[]
    for a in range(len(pca.explained_variance_ratio_)):
        pc_names.append('PC'+str(a))
    plt.xticks(x,(pc_names))
    plt.title('Scree Plot ')
    plt.xlabel('Principal components')
    plt.ylabel('Variance Explained')
    plt.show()
    plt.figure()

    #PCA Profile Plot
    
    variables={'col1':['Ha'], 'col2':['OII(1)'], 'col3':['OII(2)'], 'col4':['H The'], 'col5':['H Eta'], 'col6':['Ne III(1)'], 'col7':['Ne III(2)'], 'col8':['H Zeta'], 'col9':['H Eps'], 'col10':['H Del'], 'col11':['H Gam'], 'col12':['He II'], 'col13':['Hb'], 'col14':['OIII(1)'], 'col15':['OIII(2)'], 'col16':['He I'], 'col17':['OI(1)'], 'col18':['OI(2)'], 'col19':['NII(1)'], 'col20':['NII(2)'], 'col21':['SII(1)'], 'col22':['SII(2)'], 'col23':['Stellar']}
    variables=pd.DataFrame(data=variables)
    
    for b in range(Num_PCA_Vectors):    
        plt.plot(variables.loc[0,:],pca.components_[b,:],label='PC'+str(b))
    plt.title('Component Pattern Profiles for all Kinematic Parameters')
    plt.ylabel('Correlation')
    plt.xlabel('Variable')
    plt.plot(variables.loc[0,:],np.zeros(len(variables.loc[0,:])),"--")
    plt.legend()
    plt.show()
    plt.figure()
    
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_combined_vel_PCA(plateifu,Num_PCA_Vectors,Group_Name):
    
    values_ha_vel_combined=[]
    values_oii1_vel_combined=[]
    values_oii2_vel_combined=[]
    values_hthe_vel_combined=[]
    values_heta_vel_combined=[]
    values_neiii1_vel_combined=[]
    values_neiii2_vel_combined=[]
    values_hzet_vel_combined=[]
    values_heps_vel_combined=[]
    values_hdel_vel_combined=[]
    values_hgam_vel_combined=[]
    values_heii_vel_combined=[]
    values_hb_vel_combined=[]
    values_oiii1_vel_combined=[]
    values_oiii2_vel_combined=[]
    values_hei_vel_combined=[]
    values_oi1_vel_combined=[]
    values_oi2_vel_combined=[]
    values_nii1_vel_combined=[]
    values_nii2_vel_combined=[]
    values_sii1_vel_combined=[]
    values_sii2_vel_combined=[]
    values_stellar_vel_combined=[]
    for i in range(len(plateifu)):
        #H Alpha
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_ha_vel = ha_vel.value
        ivar_ha_vel = ha_vel.ivar
        mask_ha_vel = ha_vel.mask

        #OII 3727
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii1_vel = maps.emline_gvel_oii_3727
        values_oii1_vel = oii1_vel.value
        ivar_oii1_vel = oii1_vel.ivar
        mask_oii1_vel = oii1_vel.mask

        #OII 3729
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii2_vel = maps.emline_gvel_oii_3729
        values_oii2_vel = oii2_vel.value
        ivar_oii2_vel = oii2_vel.ivar
        mask_oii2_vel = oii2_vel.mask

        #H Theta 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hthe_vel = maps.emline_gvel_hthe_3798
        values_hthe_vel = hthe_vel.value
        ivar_hthe_vel = hthe_vel.ivar
        mask_hthe_vel = hthe_vel.mask

       #H Eta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heta_vel = maps.emline_gvel_heta_3836
        values_heta_vel = heta_vel.value
        ivar_heta_vel = heta_vel.ivar
        mask_heta_vel = heta_vel.mask

        #Ne III 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii1_vel = maps.emline_gvel_neiii1_3869
        values_neiii1_vel = neiii1_vel.value
        ivar_neiii1_vel = neiii1_vel.ivar
        mask_neiii1_vel = neiii1_vel.mask

        #Ne III 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii2_vel = maps.emline_gvel_neiii2_3968
        values_neiii2_vel = neiii2_vel.value
        ivar_neiii2_vel = neiii2_vel.ivar
        mask_neiii2_vel = neiii2_vel.mask

        #H Zeta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hzet_vel = maps.emline_gvel_hzet_3890
        values_hzet_vel = hzet_vel.value
        ivar_hzet_vel = hzet_vel.ivar
        mask_hzet_vel = hzet_vel.mask

        #H Episilon 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heps_vel = maps.emline_gvel_heps_3971
        values_heps_vel = heps_vel.value
        ivar_heps_vel = heps_vel.ivar
        mask_heps_vel = heps_vel.mask

        #H Delta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hdel_vel = maps.emline_gvel_hdel_4102
        values_hdel_vel = hdel_vel.value
        ivar_hdel_vel = hdel_vel.ivar
        mask_hdel_vel = hdel_vel.mask

        #H Gamma
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hgam_vel = maps.emline_gvel_hgam_4341
        values_hgam_vel = hgam_vel.value
        ivar_hgam_vel = hgam_vel.ivar
        mask_hgam_vel = hgam_vel.mask

        #He II 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heii_vel = maps.emline_gvel_heii_4687
        values_heii_vel = heii_vel.value
        ivar_heii_vel = heii_vel.ivar
        mask_heii_vel = heii_vel.mask

        #H Beta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hb_vel = maps.emline_gvel_hb_4862
        values_hb_vel = hb_vel.value
        ivar_hb_vel = hb_vel.ivar
        mask_hb_vel = hb_vel.mask

        #OIII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii1_vel = maps.emline_gvel_oiii1_4960
        values_oiii1_vel = oiii1_vel.value
        ivar_oiii1_vel = oiii1_vel.ivar
        mask_oiii1_vel = oiii1_vel.mask

        #OIII 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii2_vel = maps.emline_gvel_oiii2_5008
        values_oiii2_vel = oiii2_vel.value
        ivar_oiii2_vel = oiii2_vel.ivar
        mask_oiii2_vel = oiii2_vel.mask

        #He I
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hei_vel = maps.emline_gvel_hei_5877
        values_hei_vel = hei_vel.value
        ivar_hei_vel = hei_vel.ivar
        mask_hei_vel = hei_vel.mask

        #OI 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi1_vel = maps.emline_gvel_oi1_6302
        values_oi1_vel = oi1_vel.value
        ivar_oi1_vel = oi1_vel.ivar
        mask_oi1_vel = oi1_vel.mask

        #OI 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi2_vel = maps.emline_gvel_oi2_6365
        values_oi2_vel = oi2_vel.value
        ivar_oi2_vel = oi2_vel.ivar
        mask_oi2_vel = oi2_vel.mask

        #N II 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii1_vel = maps.emline_gvel_nii1_6549
        values_nii1_vel = nii1_vel.value
        ivar_nii1_vel = nii1_vel.ivar
        mask_nii1_vel = nii1_vel.mask

        #N II 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii2_vel = maps.emline_gvel_nii2_6585
        values_nii2_vel = nii2_vel.value
        ivar_nii2_vel = nii2_vel.ivar
        mask_nii2_vel = nii2_vel.mask

        #SII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii1_vel = maps.emline_gvel_sii1_6718
        values_sii1_vel = sii1_vel.value
        ivar_sii1_vel = sii1_vel.ivar
        mask_sii1_vel = sii1_vel.mask

        #SII 2 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii2_vel = maps.emline_gvel_sii2_6732
        values_sii2_vel = sii2_vel.value
        ivar_sii2_vel = sii2_vel.ivar
        mask_sii2_vel = sii2_vel.mask

        #Stellar Vel 
        maps = Maps(plateifu=plateifu.iloc[i])
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask


        values_ha_vel_combined.append(values_ha_vel.flatten())
        values_oii1_vel_combined.append(values_oii1_vel.flatten())
        values_oii2_vel_combined.append(values_oii2_vel.flatten())
        values_hthe_vel_combined.append(values_hthe_vel.flatten())
        values_heta_vel_combined.append(values_heta_vel.flatten())
        values_neiii1_vel_combined.append(values_neiii1_vel.flatten())
        values_neiii2_vel_combined.append(values_neiii2_vel.flatten())
        values_hzet_vel_combined.append(values_hzet_vel.flatten())
        values_heps_vel_combined.append(values_heps_vel.flatten())
        values_hdel_vel_combined.append(values_hdel_vel.flatten())
        values_hgam_vel_combined.append(values_hgam_vel.flatten())
        values_heii_vel_combined.append(values_heii_vel.flatten())
        values_hb_vel_combined.append(values_hb_vel.flatten())
        values_oiii1_vel_combined.append(values_oiii1_vel.flatten())
        values_oiii2_vel_combined.append(values_oiii2_vel.flatten())
        values_hei_vel_combined.append(values_hei_vel.flatten())
        values_oi1_vel_combined.append(values_oi1_vel.flatten())
        values_oi2_vel_combined.append(values_oi2_vel.flatten())
        values_nii1_vel_combined.append(values_nii1_vel.flatten())
        values_nii2_vel_combined.append(values_nii2_vel.flatten())
        values_sii1_vel_combined.append(values_sii1_vel.flatten())
        values_sii2_vel_combined.append(values_sii2_vel.flatten())
        values_stellar_vel_combined.append(values_stellar_vel.flatten())


    #Makes arrays to the appropriate size to fit the pixel maps of the different galaxies
    ha_vel=np.zeros(len(plateifu))
    oii1_vel=np.zeros(len(plateifu))
    oii2_vel=np.zeros(len(plateifu))
    hthe_vel=np.zeros(len(plateifu))
    heta_vel=np.zeros(len(plateifu))
    neiii1_vel=np.zeros(len(plateifu))
    neiii2_vel=np.zeros(len(plateifu))
    hzet_vel=np.zeros(len(plateifu))
    heps_vel=np.zeros(len(plateifu))
    hdel_vel=np.zeros(len(plateifu))
    hgam_vel=np.zeros(len(plateifu))
    heii_vel=np.zeros(len(plateifu))
    hb_vel=np.zeros(len(plateifu))
    oiii1_vel=np.zeros(len(plateifu))
    oiii2_vel=np.zeros(len(plateifu))
    hei_vel=np.zeros(len(plateifu))
    oi1_vel=np.zeros(len(plateifu))
    oi2_vel=np.zeros(len(plateifu))
    nii1_vel=np.zeros(len(plateifu))
    nii2_vel=np.zeros(len(plateifu))
    sii1_vel=np.zeros(len(plateifu))
    sii2_vel=np.zeros(len(plateifu))
    stellar_vel=np.zeros(len(plateifu))
    for j_galaxy in range(len(plateifu)):
        ha_vel[j_galaxy]=len(values_ha_vel_combined[j_galaxy])
        oii1_vel[j_galaxy]=len(values_oii1_vel_combined[j_galaxy])
        oii2_vel[j_galaxy]=len(values_oii2_vel_combined[j_galaxy])
        hthe_vel[j_galaxy]=len(values_hthe_vel_combined[j_galaxy])
        heta_vel[j_galaxy]=len(values_heta_vel_combined[j_galaxy])
        neiii1_vel[j_galaxy]=len(values_neiii1_vel_combined[j_galaxy])
        neiii2_vel[j_galaxy]=len(values_neiii2_vel_combined[j_galaxy])
        hzet_vel[j_galaxy]=len(values_hzet_vel_combined[j_galaxy])
        heps_vel[j_galaxy]=len(values_heps_vel_combined[j_galaxy])
        hdel_vel[j_galaxy]=len(values_hdel_vel_combined[j_galaxy])
        hgam_vel[j_galaxy]=len(values_hgam_vel_combined[j_galaxy])
        heii_vel[j_galaxy]=len(values_heii_vel_combined[j_galaxy])
        hb_vel[j_galaxy]=len(values_hb_vel_combined[j_galaxy])
        oiii1_vel[j_galaxy]=len(values_oiii1_vel_combined[j_galaxy])
        oiii2_vel[j_galaxy]=len(values_oiii2_vel_combined[j_galaxy])
        hei_vel[j_galaxy]=len(values_hei_vel_combined[j_galaxy])
        oi1_vel[j_galaxy]=len(values_oi1_vel_combined[j_galaxy])
        oi2_vel[j_galaxy]=len(values_oi2_vel_combined[j_galaxy])
        nii1_vel[j_galaxy]=len(values_nii1_vel_combined[j_galaxy])
        nii2_vel[j_galaxy]=len(values_nii2_vel_combined[j_galaxy])
        sii1_vel[j_galaxy]=len(values_sii1_vel_combined[j_galaxy])
        sii2_vel[j_galaxy]=len(values_sii2_vel_combined[j_galaxy])
        stellar_vel[j_galaxy]=len(values_stellar_vel_combined[j_galaxy])

    #Stores the data of each pixel for each galaxy of the same variable in 1D array, this will be fed to PCA
    values_ha_vel_combined1=np.zeros(int(sum(ha_vel)))
    values_oii1_vel_combined1=np.zeros(int(sum(oii1_vel)))
    values_oii2_vel_combined1=np.zeros(int(sum(oii2_vel)))
    values_hthe_vel_combined1=np.zeros(int(sum(hthe_vel)))
    values_heta_vel_combined1=np.zeros(int(sum(heta_vel)))
    values_neiii1_vel_combined1=np.zeros(int(sum(neiii1_vel)))
    values_neiii2_vel_combined1=np.zeros(int(sum(neiii2_vel)))
    values_hzet_vel_combined1=np.zeros(int(sum(hzet_vel)))
    values_heps_vel_combined1=np.zeros(int(sum(heps_vel)))
    values_hdel_vel_combined1=np.zeros(int(sum(hdel_vel)))
    values_hgam_vel_combined1=np.zeros(int(sum(hgam_vel)))
    values_heii_vel_combined1=np.zeros(int(sum(heii_vel)))
    values_hb_vel_combined1=np.zeros(int(sum(hb_vel)))
    values_oiii1_vel_combined1=np.zeros(int(sum(oiii1_vel)))
    values_oiii2_vel_combined1=np.zeros(int(sum(oiii2_vel)))
    values_hei_vel_combined1=np.zeros(int(sum(hei_vel)))
    values_oi1_vel_combined1=np.zeros(int(sum(oi1_vel)))
    values_oi2_vel_combined1=np.zeros(int(sum(oi2_vel)))
    values_nii1_vel_combined1=np.zeros(int(sum(nii1_vel)))
    values_nii2_vel_combined1=np.zeros(int(sum(nii2_vel)))
    values_sii1_vel_combined1=np.zeros(int(sum(sii1_vel)))
    values_sii2_vel_combined1=np.zeros(int(sum(sii2_vel)))
    values_stellar_vel_combined1=np.zeros(int(sum(stellar_vel)))
    last_step=0
    for i in range(len(plateifu)):
        #H Alpha
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_ha_vel = ha_vel.value
        ivar_ha_vel = ha_vel.ivar
        mask_ha_vel = ha_vel.mask

        #OII 3727
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii1_vel = maps.emline_gvel_oii_3727
        values_oii1_vel = oii1_vel.value
        ivar_oii1_vel = oii1_vel.ivar
        mask_oii1_vel = oii1_vel.mask

        #OII 3729
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oii2_vel = maps.emline_gvel_oii_3729
        values_oii2_vel = oii2_vel.value
        ivar_oii2_vel = oii2_vel.ivar
        mask_oii2_vel = oii2_vel.mask

        #H Theta 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hthe_vel = maps.emline_gvel_hthe_3798
        values_hthe_vel = hthe_vel.value
        ivar_hthe_vel = hthe_vel.ivar
        mask_hthe_vel = hthe_vel.mask

       #H Eta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heta_vel = maps.emline_gvel_heta_3836
        values_heta_vel = heta_vel.value
        ivar_heta_vel = heta_vel.ivar
        mask_heta_vel = heta_vel.mask

        #Ne III 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii1_vel = maps.emline_gvel_neiii1_3869
        values_neiii1_vel = neiii1_vel.value
        ivar_neiii1_vel = neiii1_vel.ivar
        mask_neiii1_vel = neiii1_vel.mask

        #Ne III 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        neiii2_vel = maps.emline_gvel_neiii2_3968
        values_neiii2_vel = neiii2_vel.value
        ivar_neiii2_vel = neiii2_vel.ivar
        mask_neiii2_vel = neiii2_vel.mask

        #H Zeta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hzet_vel = maps.emline_gvel_hzet_3890
        values_hzet_vel = hzet_vel.value
        ivar_hzet_vel = hzet_vel.ivar
        mask_hzet_vel = hzet_vel.mask

        #H Episilon 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heps_vel = maps.emline_gvel_heps_3971
        values_heps_vel = heps_vel.value
        ivar_heps_vel = heps_vel.ivar
        mask_heps_vel = heps_vel.mask

        #H Delta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hdel_vel = maps.emline_gvel_hdel_4102
        values_hdel_vel = hdel_vel.value
        ivar_hdel_vel = hdel_vel.ivar
        mask_hdel_vel = hdel_vel.mask

        #H Gamma
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hgam_vel = maps.emline_gvel_hgam_4341
        values_hgam_vel = hgam_vel.value
        ivar_hgam_vel = hgam_vel.ivar
        mask_hgam_vel = hgam_vel.mask

        #He II 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        heii_vel = maps.emline_gvel_heii_4687
        values_heii_vel = heii_vel.value
        ivar_heii_vel = heii_vel.ivar
        mask_heii_vel = heii_vel.mask

        #H Beta
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hb_vel = maps.emline_gvel_hb_4862
        values_hb_vel = hb_vel.value
        ivar_hb_vel = hb_vel.ivar
        mask_hb_vel = hb_vel.mask

        #OIII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii1_vel = maps.emline_gvel_oiii1_4960
        values_oiii1_vel = oiii1_vel.value
        ivar_oiii1_vel = oiii1_vel.ivar
        mask_oiii1_vel = oiii1_vel.mask

        #OIII 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oiii2_vel = maps.emline_gvel_oiii2_5008
        values_oiii2_vel = oiii2_vel.value
        ivar_oiii2_vel = oiii2_vel.ivar
        mask_oiii2_vel = oiii2_vel.mask

        #He I
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        hei_vel = maps.emline_gvel_hei_5877
        values_hei_vel = hei_vel.value
        ivar_hei_vel = hei_vel.ivar
        mask_hei_vel = hei_vel.mask

        #OI 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi1_vel = maps.emline_gvel_oi1_6302
        values_oi1_vel = oi1_vel.value
        ivar_oi1_vel = oi1_vel.ivar
        mask_oi1_vel = oi1_vel.mask

        #OI 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        oi2_vel = maps.emline_gvel_oi2_6365
        values_oi2_vel = oi2_vel.value
        ivar_oi2_vel = oi2_vel.ivar
        mask_oi2_vel = oi2_vel.mask

        #N II 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii1_vel = maps.emline_gvel_nii1_6549
        values_nii1_vel = nii1_vel.value
        ivar_nii1_vel = nii1_vel.ivar
        mask_nii1_vel = nii1_vel.mask

        #N II 2
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        nii2_vel = maps.emline_gvel_nii2_6585
        values_nii2_vel = nii2_vel.value
        ivar_nii2_vel = nii2_vel.ivar
        mask_nii2_vel = nii2_vel.mask

        #SII 1
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii1_vel = maps.emline_gvel_sii1_6718
        values_sii1_vel = sii1_vel.value
        ivar_sii1_vel = sii1_vel.ivar
        mask_sii1_vel = sii1_vel.mask

        #SII 2 
        maps = Maps(plateifu=plateifu.iloc[i])
        # get an emission line map
        sii2_vel = maps.emline_gvel_sii2_6732
        values_sii2_vel = sii2_vel.value
        ivar_sii2_vel = sii2_vel.ivar
        mask_sii2_vel = sii2_vel.mask

        #Stellar Vel 
        maps = Maps(plateifu=plateifu.iloc[i])
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        if i==0:
            d=0
            f=np.size(values_ha_vel)
        else:
            d=last_step
            f=last_step+np.size(values_ha_vel)

        values_ha_vel_combined1[d:f]=values_ha_vel.flatten()
        values_oii1_vel_combined1[d:f]=values_oii1_vel.flatten()
        values_oii2_vel_combined1[d:f]=values_oii2_vel.flatten()
        values_hthe_vel_combined1[d:f]=values_hthe_vel.flatten()
        values_heta_vel_combined1[d:f]=values_heta_vel.flatten()
        values_neiii1_vel_combined1[d:f]=values_neiii1_vel.flatten()
        values_neiii2_vel_combined1[d:f]=values_neiii2_vel.flatten()
        values_hzet_vel_combined1[d:f]=values_hzet_vel.flatten()
        values_heps_vel_combined1[d:f]=values_heps_vel.flatten()
        values_hdel_vel_combined1[d:f]=values_hdel_vel.flatten()
        values_hgam_vel_combined1[d:f]=values_hgam_vel.flatten()
        values_heii_vel_combined1[d:f]=values_heii_vel.flatten()
        values_hb_vel_combined1[d:f]=values_hb_vel.flatten()
        values_oiii1_vel_combined1[d:f]=values_oiii1_vel.flatten()
        values_oiii2_vel_combined1[d:f]=values_oiii2_vel.flatten()
        values_hei_vel_combined1[d:f]=values_hei_vel.flatten()
        values_oi1_vel_combined1[d:f]=values_oi1_vel.flatten()
        values_oi2_vel_combined1[d:f]=values_oi2_vel.flatten()
        values_nii1_vel_combined1[d:f]=values_nii1_vel.flatten()
        values_nii2_vel_combined1[d:f]=values_nii2_vel.flatten()
        values_sii1_vel_combined1[d:f]=values_sii1_vel.flatten()
        values_sii2_vel_combined1[d:f]=values_sii2_vel.flatten()
        values_stellar_vel_combined1[d:f]=values_stellar_vel.flatten()
       
        
        last_step=last_step+np.size(values_ha_vel)


    values=np.column_stack([values_ha_vel_combined1, values_oii1_vel_combined1,values_oii2_vel_combined1, values_hthe_vel_combined1, values_heta_vel_combined1,  values_neiii1_vel_combined1, values_neiii2_vel_combined1, values_hzet_vel_combined1, values_heps_vel_combined1, values_hdel_vel_combined1, values_hgam_vel_combined1, values_heii_vel_combined1, values_hb_vel_combined1, values_oiii1_vel_combined1, values_oiii2_vel_combined1, values_hei_vel_combined1, values_oi1_vel_combined1, values_oi2_vel_combined1, values_nii1_vel_combined1, values_nii2_vel_combined1, values_sii1_vel_combined1, values_sii2_vel_combined1, values_stellar_vel_combined1])
    values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
    pca = PCA(n_components=Num_PCA_Vectors)
    principalComponents = pca.fit_transform(values)
    
    np.savetxt(Group_Name +str('_PC_Vectors'),pca.components_)
    np.savetxt(Group_Name +str('_PC_Variance'),pca.explained_variance_)
    
    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array




###################################IN TESTING PHASE######################################################

# def image_vector


#Importing All MaNGA Data from DPRall Schema
data=pd.read_csv('CompleteTable.csv')
   
e=pd.DataFrame(['7977-3704','8139-12701','8258-9102','8317-1901'])

b=galaxy_combined_vel_PCA(e.loc[:,0],3,'Test')

# c=galaxy_profile_plot_global_combined(data.loc[:,'plateifu'],3,8)

# a=combined_maps_error_bar(e.loc[:,0],3,8,10)


# im = Image('8139-12701',release='DR15')
# im.plot()














    