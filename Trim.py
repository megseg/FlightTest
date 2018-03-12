from math import *
import numpy as np
from AirspeedReduction import *

f=np.genfromtxt("PFD.csv",delimiter=",")#,dtype='str')

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
eVer=ReducedEquivalent(eVe,Ws,W)


