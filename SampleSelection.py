import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd

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

#Using np.where to classify galaxies as QG, SFG or GVG
QG=np.where(np.log10(data.loc[:,'sfr_tot'])<seperationline(np.log10(data.loc[:,'nsa_sersic_mass']))) #Creates an array of the indicies where this condition is true
QG=data.loc[QG] #Taking the rows of data that correspond to QGs

SFG=np.where(np.log10(data.loc[:,'sfr_tot'])>seperationline(np.log10(data.loc[:,'nsa_sersic_mass'])))
SFG=data.loc[SFG]

GVG=np.where(np.log10(data.loc[:,'sfr_tot'])==y1)
GVG=data.loc[GVG]

#Testing if this was successful using a plot 
plt.title('Mass Vs SFR of Galaxies in MaNGA')
plt.xlabel('Log of Mass')
plt.ylabel('Log of SFR')
plt.scatter(np.log10(QG.loc[:,'nsa_sersic_mass']),np.log10(QG.loc[:,'sfr_tot']),c='red')
plt.scatter(np.log10(SFG.loc[:,'nsa_sersic_mass']),np.log10(SFG.loc[:,'sfr_tot']), c='blue')
plt.scatter(np.log10(GVG.loc[:,'nsa_sersic_mass']),np.log10(GVG.loc[:,'sfr_tot']), c='green')
plt.plot(x1,y1)
plt.show()



