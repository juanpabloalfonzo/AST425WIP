import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

data = np.genfromtxt('MassVSFR.csv', delimiter=',')
data1= np.genfromtxt('ha_flux.csv', delimiter=',')

mass=data[:,2]
mass=np.delete(mass,0)
logmass=np.log10(mass)

SFR=data[:,1]
SFR=np.delete(SFR,0)
logSFR=np.log10(SFR)

ha_flux=data1[:,1]
ha_flux=np.delete(ha_flux,0)



def fit(m,x,b):
    return m*x+b

#Use curve fit function to create a line of best fit
#Fit1, Fit2 = curve_fit(fit, logmass, logSFR)








plt.title('Mass Vs SFR of Galaxies in MaNGA')
plt.xlabel('Log of Mass')
plt.ylabel('Log of SFR')
plt.scatter(logmass,logSFR, c=ha_flux, cmap='seismic')
#plt.plot(logmass,position(Fit1[0],Fit[1],Fit1[2]), label='Line of Best Fit')
plt.show()