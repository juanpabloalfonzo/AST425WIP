import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
#import scikit-learn as sl

def seperationline(x):
    m=(-2.6- -0.3)/(9.1-11.7)
    y=m*(x-11.7)-0.3
    return y



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


x1=np.linspace(7,13,50)
y1=seperationline(x1)


plt.title('Mass Vs SFR of Galaxies in MaNGA')
plt.xlabel('Log of Mass')
plt.ylabel('Log of SFR')
# plt.scatter(logmass,logSFR, c=np.log10(ha_flux), vmin=-2, vmax=-0.8, cmap='viridis', alpha=0.1)
plt.hist2d(logmass,logSFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.plot(x1,y1)
plt.colorbar()
# plt.hist(np.log10(ha_flux),bins=np.linspace(-2,-0.8,41))
#plt.plot(logmass,position(Fit1[0],Fit[1],Fit1[2]), label='Line of Best Fit')
plt.show()