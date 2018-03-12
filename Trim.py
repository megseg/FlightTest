from math import *
import numpy as np
from AirspeedReduction import *
import matplotlib.pyplot as plt

#Placeholder values:
S=70
CD=0.3
CmTc=0.5
Cmde=0.05

f=np.genfromtxt("PFD.csv",delimiter=",")#,dtype='str')
thrust=np.genfromtxt("thrust.dat",dtype='float')

eET=f[:,2][58:65]
ehp=f[:,3][58:65]*0.3048
eIAS=(f[:,4][58:65]+2)*1852/3600
ea=f[:,5][58:65]
ede=f[:,6][58:65]
edetr=f[:,7][58:65]
eFe=f[:,8][58:65]
eFFL=f[:,9][58:65]*0.453592/3600
eFFR=f[:,10][58:65]*0.453592/3600
eFused=f[:,11][58:65]*0.453592
eTAT=f[:,12][58:65]+273.15

eVt=EquivalentAirspeed(eIAS,Tm)[0]
eVe=EquivalentAirspeed(eIAS,Tm)[1]
erho=EquivalentAirspeed(eIAS,Tm)[2]
eVer=ReducedEquivalent(eVe,Ws,W)
eThrust=thrust[:,0]+thrust[:,1]

Tc=eThrust/(1/2*erho*eVt**2*S)
Tcs=rhoISA*eVer**2*CD/(erho*eVt**2)

eqde=ede-CmTc/Cmde*(Tcs-Tc)
plt.scatter(eVer,eqde)
plt.show()
