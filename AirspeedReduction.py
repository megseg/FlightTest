from math import *
from isaConstants import *
import numpy as np
# Placeholder values:
Ws=60500
W=60000
Vc=140
Tm=300
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
    return Vt,Ve

def ReducedEquivalent(Ve,Ws,W):
    Ver=Ve*np.sqrt(Ws/W)
    return Ver

