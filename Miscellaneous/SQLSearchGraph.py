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

np.savetxt('mangaid', mangaid)

mass=data[:,2]
mass=np.delete(mass,0)
logmass=np.log10(mass)

SFR=data[:,1]
SFR=np.delete(SFR,0)
logSFR=np.log10(SFR)

ha_flux=data1[:,1]
ha_flux=np.delete(ha_flux,0)

#Creating x points to allow the plotting of the distinguishing line
x1=np.linspace(7,13,9718)
y1=seperationline(x1)


plt.title('SFR vs Mass of Galaxies in MaNGA')
plt.xlabel(r'$log(M/M_{\odot})$')
plt.ylabel(r'$log(SFR/M_{\odot})$')
plt.scatter(logmass,logSFR, c=np.log10(ha_flux), vmin=-2, vmax=-0.8, cmap='viridis', alpha=0.1)
plt.hist2d(logmass,logSFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.plot(x1,y1,label='Separation Line')
plt.legend(loc='bottom left')
plt.colorbar().set_label('Ha Flux')
plt.savefig('Distinction Line.png')
plt.show()

#Putting Manga ID and logSFR/logmass in one array so galaxies can be tracked
logSFR_ID=np.column_stack((np.transpose(mangaid),np.transpose(logSFR),np.transpose(logmass)))

# Removing all rows with a NaN and infinity elments in logSFR and logmass to avoid problems with np.where

logSFR_ID=logSFR_ID[~np.isnan(logSFR_ID).any(axis=1)]
logSFR_ID=logSFR_ID[~np.isinf(logSFR_ID).any(axis=1)]

np.savetxt('Quiescent Galaxies From Distinction Line',logSFR_ID)

# Using np.where to find galaxy information around the distinction line
QG=logSFR_ID[~((logSFR_ID[:,1]>y1))]

SFG=logSFR_ID[~((logSFR_ID[:,1]<y1))]

np.savetxt('Quiescent Galaxies From Distinction Line',QG)
np.savetxt('Star-Forming Galaxies From Distinction Line',SFG)

#Checking if Classification above and below line actually worked
plt.scatter(SFG[:,2],SFG[:,1],marker='.')
plt.scatter(QG[:,2],QG[:,1],marker='s')
plt.xlim(6,14)
plt.plot(x1,y1)
plt.show()