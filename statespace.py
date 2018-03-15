import numpy as np
import control.matlab as mat
from Cit_par import *
import matplotlib.pyplot as plt
import scipy.signal as sg
import control


#-------------------------------------- values for symmetric matrices------------------------------------
xu      = V0/c*CXu/2/muc
xa      = V0/c*CXa/2/muc
xtheta  = V0/c*CZ0/2/muc

xdele   = V0/c*CXde/2/muc
xdelt   = V0/c*CXq/2/muc                #pitch rate is the same as trim deflection rate


zu      = V0/c*CZu/(2*muc-CZadot)
za      = V0/c*CZa/(2*muc-CZadot)
ztheta  = V0/c*CX0/(2*muc-CZadot)
zq      = V0/c*(2*muc + CZq)/(2*muc-CZadot)
zdele   = V0/c*CZde/(2*muc-CZadot)
zdelt   = V0/c*CZq/(2*muc-CZadot)       #pitch rate is the same as trim deflection rate


mu      = V0/c * (Cmu + CZu*Cmadot/(2*muc - CZadot))/(2*muc*KY2)
ma      = V0/c * (Cma + CZa*Cmadot/(2*muc - CZadot))/(2*muc*KY2)
mtheta  = -V0/c* (CX0*Cmadot/(2*muc - CZadot))/(2*muc*KY2)
mq      = V0/c * (Cmq + Cmadot*(2*muc + CZq)/(2*muc - CZadot))/(2*muc*KY2)
mdele   = V0/c * (Cmde + CZde*Cmadot/(2*muc - CZadot))/(2*muc*KY2)
mdelt   = V0/c * (Cmq + CZq*Cmadot/(2*muc - CZadot))/(2*muc*KY2)    #pitch rate is the same as trip deflection rate

#-----------------------------------symmetric matrices-----------------------------------------------------

A_sym = np.matrix([[xu , xa , xtheta , 0],
                   [zu , za , ztheta , zq],
                   [0 , 0 , 0 , (V0/c)**2],
                   [mu , ma , mtheta , mq]])

B_sym = np.matrix([[xdele , xdelt],
                   [zdele , zdelt],
                   [0 , 0],
                   [mdele , mdelt]])

B2_sym = np.matrix([[xdele ],
                   [zdele ],
                   [0 ],
                   [mdele ]])

C_sym = np.matrix([[1. , 0 , 0 , 0],
                   [0 , 1. , 0 , 0],
                   [0 , 0 , 1. , 0],
                   [0 , 0 , 0 , 1.]])

D_sym = np.zeros([4,2])

symsys = mat.ss(A_sym,B_sym,C_sym,D_sym)



H = mat.tf(symsys)

t = np.arange(0,100.01,0.01)

y, t = control.forced_response(symsys,t)
#y, t = mat.impulse(H,t)

plt.plot(t,y)
plt.show()






















#---------------------------------values for asymmetric matrices------------------------------------------------

yb      = V0/b*CYb/2/mub
yphi    = V0/b*CL/2/mub
yp      = V0/b*CYp/2/mub
yr      = V0/b*(CYr-4*mub)/2/mub

ydelr   = V0/b*CYdr/2/mub


lb      = V0/b*(Clb*KZ2+Cnb*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))

lp      = V0/b*(Clp*KZ2+Cnp*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
lr      = V0/b*(Clr*KZ2+Cnr*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
ldela   = V0/b*(Clda*KZ2+Cnda*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
ldelr   = V0/b*(Cldr*KZ2+Cndr*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))


nb      = V0/b*(Clb*KXZ + Cnb*KX2)/(4*mub*(KX2*KZ2-KXZ**2))

np1      = V0/b*(Clp*KXZ + Cnp*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
nr      = V0/b*(Clr*KXZ + Cnr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
ndela   = V0/b*(Clda*KXZ + Cnda*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
ndelr   = V0/b*(Cldr*KXZ + Cndr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))

#------------------------------------Asymmetric matrices------------------------------------------------------

A_asym = np.matrix([[yb , yphi , yp , yr],
                    [0 , 0 , 2*V0/b , 0],
                    [lb , 0 , lp , lr],
                    [nb , 0 , np1 , nr]])

B_asym = np.matrix([[0 , ydelr],
                    [0 , 0],
                    [ldela , ldelr],
                    [ndela , ndelr]])

C_asym = np.matrix([[1. , 0 , 0 , 0],
                    [0 , 1. , 0 , 0],
                    [0 , 0 , 1. , 0],
                    [0 , 0 , 0 , 1.]])

D_asym = np.zeros([4,2])

asymsys = mat.ss(A_asym,B_asym,C_asym,D_asym)

