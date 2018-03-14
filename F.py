from math import *
from isaConstants import *
import numpy as np
from D import *
import matplotlib.pyplot as plt

def EquivalentAirspeed(Vc,Tm):
    p=pISA*(1+Lambda*hp/TISA)**(-g0/(Lambda*R))

    M1=(1+(sh-1)/(2*sh)*rhoISA/pISA*Vc**2)**(sh/(sh-1))
    M2=1+pISA/p*(M1-1)
    M3=2/(sh-1)*M2**((sh-1)/sh)-1
    M=np.sqrt(M3)

    T = Tm/(1+(sh-1)/2*M**2)

    a=np.sqrt(sh*R*T)

    Vt=M*a

    rho=p/(R*T)

    Ve=Vt*np.sqrt(rho/rhoISA) #The equivalent airspeed
    return Vt,Ve,rho


def ReducedEquivalent(Ve,Ws,W):
    Ver=Ve*np.sqrt(Ws/W)
    return Ver

def ReducedElevatorDeflection(eIAS,Tm,Ws,W,eThrust,ede,CmTc,Cmde,S,CD):
    eVt=EquivalentAirspeed(eIAS,Tm)[0]
    eVe=EquivalentAirspeed(eIAS,Tm)[1]
    erho=EquivalentAirspeed(eIAS,Tm)[2]
    eVer=np.sort(ReducedEquivalent(eVe,Ws,W))

    Tc=eThrust/(1/2*erho*eVt**2*S)
    Tcs=rhoISA*eVer**2*CD/(erho*eVt**2)
   
    eqde=np.sort(ede-CmTc/Cmde*(Tcs-Tc))
    plt.gca().invert_yaxis() 
    plt.plot(eVer,eqde)
    plt.title("Reduced elevator angle in function of the reduced equivalent airspeed",fontsize=15)
    plt.xlabel("Reduced equivalent airspeed [m/s]",fontsize=15)
    plt.ylabel("Reduced elevator angle [deg]",fontsize=15)
    plt.show()






