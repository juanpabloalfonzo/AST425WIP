import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

#Define a function that will create line to distinguish SFGs from QGs
def seperationline(x):
    m=(-2.6- -0.3)/(9.1-11.7)
    y=m*(x-11.7)-0.3
    return y


#Importing Data from SQL search 
data = np.genfromtxt('MassVSFR.csv', delimiter=',')
data1= np.genfromtxt('ha_flux.csv', delimiter=',')

mangaid=data[:,0]
mangaid=np.delete(mangaid,0) #Deletes title of column in spreadsheet to avoid Nan value

mass=data[:,2]
mass=np.delete(mass,0)
logmass=np.log10(mass)

SFR=data[:,1]
SFR=np.delete(SFR,0)
logSFR=np.log10(SFR)

ha_flux=data1[:,1]
ha_flux=np.delete(ha_flux,0)

#Creating x points to allow the plotting of the distinguishing line
x1=np.linspace(7,13,9724)
y1=seperationline(x1)


plt.title('Mass Vs SFR of Galaxies in MaNGA')
plt.xlabel('Log of Mass')
plt.ylabel('Log of SFR')

# plt.scatter(logmass,logSFR, c=np.log10(ha_flux), vmin=-2, vmax=-0.8, cmap='viridis', alpha=0.1)
plt.hist2d(logmass,logSFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.plot(x1,y1)
plt.colorbar()
plt.savefig('Distinction Line.png')
plt.show()

#Putting Manga ID and logSFR/logmass in one array so galaxies can be tracked
logSFR_ID=np.vstack((mangaid,logSFR))
logmass_ID=np.vstack((mangaid,logmass))

#Removing NaN elments in logSFR and logmass to avoid problems when using np.where 
if np.isnan(np.sum(logSFR)):
    logSFR = logSFR[~np.isnan(logSFR)] # removes nan elements from logSFR

np.savetxt('Nan fixed logSFR', logSFR_ID)

if np.isnan(np.sum(logmass)):
    logmass = logmass[~np.isnan(logmass)] # removes nan elements from logmass



#Using np.where to find galaxy information around the distinction line
# QGSFR=np.where(logSFR_ID[:,1] < y1, logSFR,logSFR) #Galaxies bellow the line, hence QG


