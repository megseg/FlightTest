import numpy as np
# Reading in columns of values from the static measurement data sheet
f=np.genfromtxt("PFD.csv",delimiter=",")#,dtype='str')
thrust=np.genfromtxt("thrust.dat",dtype='float')

# For Cl-Cd
cET=f[:,2][27:32] #Elapsed time
chp=f[:,3][27:32]*0.3048 #Pressure height [m]
cIAS=(f[:,4][27:32]-2)*1852/3600 #Indicated airspeed (already calibtated and converted to [m/s])
ca=f[:,5][27:32] #Angle of attack [deg]
cFFL=f[:,6][27:32]*0.453592/3600 #Fuel flow left [kg/s]
cFFR=f[:,7][27:32]*0.453592/3600 #Fuel flow right [kg/s]
cFused=f[:,8][27:32]*0.453592 #Fuel used
cTAT=f[:,9][27:32]+273.15 #Total measured air temperature


# For elevator deflection
eET=f[:,2][58:65] #Elapsed time 
ehp=f[:,3][58:65]*0.3048 #Pressure height [m]
eIAS=(f[:,4][58:65]-2)*1852/3600 #Indicated airspeed (already calibtated and converted to [m/s])
ea=f[:,5][58:65] #Angle of attack [deg]
ede=f[:,6][58:65] #Elevator deflection angle delta [deg]
edetr=f[:,7][58:65] #Elevator trim angle [deg]
eFe=f[:,8][58:65] #Elevator stick force [N]
eFFL=f[:,9][58:65]*0.453592/3600 #Fuel flow left [kg/s]
eFFR=f[:,10][58:65]*0.453592/3600 #Fuel flow right [kg/s]
eFused=f[:,11][58:65]*0.453592 #Fuel used
eTAT=f[:,12][58:65]+273.15 #Total measured air temperature


# For elevator trim
deET=f[:,2][74:76]
dehp=f[:,3][74:76]*0.3048
deIAS=(f[:,4][74:76]-2)*1852/3600 #Indicated airspeed (already calibrated and converted to [m/s])
deTAT=f[:,12][74:76]+273.15
dede=f[74,6]-f[75,6]
deFused=f[:,11][74:76]*0.453592 #Fuel used

# Importing thrusts from thrust.dat
eThrust=thrust[:,0][0:7]+thrust[:,1][0:7]
eSthrust=thrust[:,0][8:15]+ thrust[:,1][8:15]

ET=np.hstack(([0.0],cET,eET,deET))
Fused=np.hstack(([0.0],cFused,eFused,deFused))
