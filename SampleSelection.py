import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


plt.ion()

#Define a function that will create line to distinguish SFGs from QGs
def seperationline(x):
    m=(-2.6- -0.3)/(9.1-11.7)
    y=m*(x-11.7)-0.3
    return y

#Importing All MaNGA Data from DPRall Schema
data=pd.read_csv('CompleteTable.csv')

#Generating Distinction Line 
x1=np.linspace(7,13,len(data))
y1=seperationline(x1)

print(seperationline(7.96199))


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
plt.show()
plt.figure()

#Using sci kit learn DBSCAN function (min sample is 62, as this gets bigger we get very small and distinct groups, smaller and the two groups merge)


clustering= DBSCAN(eps=0.15, min_samples=62).fit(np.log10(data2[['nsa_sersic_mass', 'sfr_tot']])) #esp is radius epsilion from point and min_samples is min amount of samples in neighbourhood
cluster= clustering.labels_ 
plt.scatter(np.log10(data2['nsa_sersic_mass']),np.log10(data2['sfr_tot']),c=cluster,alpha=0.2)
plt.colorbar()
plt.show()
plt.figure()

