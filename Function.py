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
from marvin import config
from sklearn.decomposition import PCA
from sklearn.utils import resample

# set config attributes and turn on global downloads of Marvin data
config.setRelease('DR15')
config.mode = 'local'
config.download = True

plt.ion()

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
    

    return(pca.components_,pca.explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array
 
def galaxy_profile_plot_multi(plateifu,Num_PCA_Vectors,Num_Variables):
    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
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
        d=Num_PCA_Vectors*i
        pca_components[:,:,i]=pca.components_
        pca_explained_variance_ratio_[i,:]=pca.explained_variance_ratio_
        
    return(pca_components,pca_explained_variance_ratio_) #Returns PCA vector components, Variance ratios as an array

def galaxy_PCA_multi(plateifu,Num_PCA_Vectors,Num_Variables):
    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
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
        ha_ew = maps.emline_gew_ha_656len4
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

###################################IN TESTING PHASE######################################################

def bootstrap_galaxy(plateifu, Num_PCA_Vectors, Num_Variables,reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu),reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        a=resample(plateifu)
        b=galaxy_PCA(a,Num_PCA_Vectors,Num_Variables)
        PC_Vector_Components[:,:,:,i_reps]=b[0]  

    
    PC_Errors_STD=np.zeros([Num_PCA_Vectors,Num_Variables])
    for i_variables in range(Num_Variables):
        for i_pc in range(Num_PCA_Vectors):
            PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:,:])

    return(PC_Errors_STD)

def bootstrap_maps_single(plateifu, Num_PCA_Vectors, Num_Variables, reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        
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

    
    PC_Errors_STD=np.zeros([Num_PCA_Vectors,Num_Variables])
    for i_variables in range(Num_Variables):
        for i_pc in range(Num_PCA_Vectors):
            PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:])

    return(PC_Errors_STD)

def bootstrap_maps_multi(plateifu, Num_PCA_Vectors, Num_Variables, reps):
    PC_Vector_Components=np.zeros([Num_PCA_Vectors,Num_Variables,reps])
    for i_reps in range(reps): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "reps" number of times
        plateifu=np.random.choice(plateifu,1)
        maps = Maps(plateifu=plateifu[0])
        print(maps)
        # get an emission line map
        haflux = maps.emline_gflux_ha_6564
        values_flux = haflux.value
        ivar_flux = haflux.ivar
        mask_flux = haflux.mask
        #haflux.plot()

        maps = Maps(plateifu=plateifu[0])
        print(maps)
        # get an emission line map
        ha_vel = maps.emline_gvel_ha_6564
        values_vel = ha_vel.value
        ivar_vel = ha_vel.ivar
        mask_vel = ha_vel.mask
        #ha_vel.plot()

        maps = Maps(plateifu=plateifu[0])
        print(maps)
        # get an emission line map
        ha_sigma = maps.emline_sigma_ha_6564
        values_sigma = ha_sigma.value
        ivar_sigma = ha_sigma.ivar
        mask_sigma = ha_sigma.mask
        #ha_sigma.plot()

        maps = Maps(plateifu=plateifu[0])
        print(maps)
        # get an emission line map
        ha_ew = maps.emline_gew_ha_6564
        values_ew = ha_vel.value
        ivar_ew = ha_vel.ivar
        mask_ew = ha_vel.mask
        #ha_ew.plot()

        maps = Maps(plateifu=plateifu[0])
        print(maps)
        # get an emission line map
        stellar_vel = maps.stellar_vel
        values_stellar_vel = stellar_vel.value
        ivar_stellar_vel = stellar_vel.ivar
        mask_stellar_vel = stellar_vel.mask
        #stellar_vel.plot()

        maps = Maps(plateifu=plateifu[0])
        print(maps)
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

def galaxy_error_bar(plateifu,Num_PCA_Vectors,Num_Variables,reps):

    g=bootstrap_galaxy(plateifu,Num_PCA_Vectors,Num_Variables,reps)

    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
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

    pca_components=np.zeros([Num_PCA_Vectors,Num_Variables,len(plateifu)])
    pca_explained_variance_ratio_=np.zeros([len(plateifu),Num_PCA_Vectors])
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



   
e=pd.DataFrame(['7977-3704','8139-12701','8258-9102','8317-1901'])

#b=galaxy_profile_plot(e.loc[:,0],3,6)

#a=bootstrap_maps(e.loc[:,0],3,6,10)






# PC_Vector_Components=np.zeros([3,6,20])
# for i_20 in range(20): #Randomly samples the inputed data frame of galaxies and does PCA Analysis "20" number of times
    
#     maps = Maps(plateifu='7977-3704')
#     print(maps)
#     # get an emission line map
#     haflux = maps.emline_gflux_ha_6564
#     values_flux = haflux.value
#     ivar_flux = haflux.ivar
#     mask_flux = haflux.mask
#     #haflux.plot()

#     maps = Maps(plateifu='7977-3704')
#     print(maps)
#     # get an emission line map
#     ha_vel = maps.emline_gvel_ha_6564
#     values_vel = ha_vel.value
#     ivar_vel = ha_vel.ivar
#     mask_vel = ha_vel.mask
#     #ha_vel.plot()

#     maps = Maps(plateifu='7977-3704')
#     print(maps)
#     # get an emission line map
#     ha_sigma = maps.emline_sigma_ha_6564
#     values_sigma = ha_sigma.value
#     ivar_sigma = ha_sigma.ivar
#     mask_sigma = ha_sigma.mask
#     #ha_sigma.plot()

#     maps = Maps(plateifu='7977-3704')
#     print(maps)
#     # get an emission line map
#     ha_ew = maps.emline_gew_ha_6564
#     values_ew = ha_vel.value
#     ivar_ew = ha_vel.ivar
#     mask_ew = ha_vel.mask
#     #ha_ew.plot()

#     maps = Maps(plateifu='7977-3704')
#     print(maps)
#     # get an emission line map
#     stellar_vel = maps.stellar_vel
#     values_stellar_vel = stellar_vel.value
#     ivar_stellar_vel = stellar_vel.ivar
#     mask_stellar_vel = stellar_vel.mask
#     #stellar_vel.plot()

#     maps = Maps(plateifu='7977-3704')
#     print(maps)
#     # get an emission line map
#     stellar_sigma = maps.stellar_sigma
#     values_stellar_sigma = stellar_sigma.value
#     ivar_stellar_sigma = stellar_sigma.ivar
#     mask_stellar_sigma = stellar_sigma.mask
#     #stellar_vel.plot()

#     values_flux=resample(values_flux)
#     values_vel=resample(values_vel)
#     values_ew=resample(values_ew)
#     values_sigma=resample(values_sigma)
#     values_stellar_vel=resample(values_stellar_vel)
#     values_stellar_sigma=resample(values_stellar_sigma)


#     values=np.column_stack([values_flux.flatten(),values_vel.flatten(),values_ew.flatten(),values_sigma.flatten(),values_stellar_vel.flatten(), values_stellar_sigma.flatten()])
#     values = StandardScaler().fit_transform(values) #Scale the data to mean 0 and std of 1
#     pca = PCA(n_components=3)
#     principalComponents = pca.fit_transform(values)
    
    
#     PC_Vector_Components[:,:,i_20]=pca.components_  


# PC_Errors_STD=np.zeros([3,6])
# for i_variables in range(6):
#     for i_pc in range(3):
#         PC_Errors_STD[i_pc,i_variables]=np.std(PC_Vector_Components[i_pc,i_variables,:])










    