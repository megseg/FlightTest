from D import *
from Cit_par import *
import F
from math import *
import matplotlib.pyplot as plt
Cmde = F.ElevatorEffectiveness()
Cma = F.LongitudinalStability(ede,ea,Cmde)

cVt=F.EquivalentAirspeed(cIAS,cTAT,chp)[0]
cVe=F.EquivalentAirspeed(cIAS,cTAT,chp)[1]
crho=F.EquivalentAirspeed(cIAS,cTAT,chp)[2]
cM=F.EquivalentAirspeed(cIAS,cTAT,chp)[3]
cT=F.EquivalentAirspeed(cIAS,cTAT,chp)[4]

CD0_fit = np.polyfit(F.C_D(cThrust,cVt,crho), F.C_L(cET,cVt, crho)**2, 1)

#print C_D_0
p = np.poly1d(CD0_fit)
CD0 = -CD0_fit[1]/CD0_fit[0]
#print C_D_0_correct

## Oswald efficiency factor ##
e = F.C_L(cVe, crho, S)**2/(F.C_D(cThrust,cVe,crho))/(pi*A)

#CL-alpha graph
plt.plot(ca, F.C_L(cVe, crho, S))
plt.title('$C_L-alpha$ curve, clean configuration')
plt.ylabel('Lift coefficient')
plt.xlabel('Angle of attack in degrees')
plt.text(5.0, 0.7, 'Mach number range: $0.28-0.59$')
plt.text(4.5, 0.6, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

#CL-CD graph
plt.plot(F.C_L(cVe, crho, S), F.C_D(cThrust,cVe,crho))
plt.title('$C_L-C_D$ curve, clean configuration')
plt.ylabel('Drag coefficient')
plt.xlabel('Lift coefficient')
plt.text(1.1, 0.055, 'Mach number range: $0.28-0.59$')
plt.text(1.0, 0.05, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

#CD-alpha graph
plt.plot(ca, F.C_D(cThrust,cVe,crho))
plt.title('$C_D-alpha$ curve, clean configuration')
plt.ylabel('Drag coefficient')
plt.xlabel('Angle of attack in degrees')
plt.text(5.0, 0.055, 'Mach number range: $0.28-0.59$')
plt.text(4.5, 0.05, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

#CL**2-CD graph
plt.plot(F.C_D(cThrust,cVe,crho), F.C_L(cVe, crho, S)**2)
plt.plot(F.C_D(cThrust,cVe,crho), p(F.C_D(cThrust,cVe,crho)))
plt.title('$C_L^2-C_D$ curve, clean configuration')
plt.ylabel('Lift coefficient squared')
plt.xlabel('Drag coefficient')
plt.text(0.078, 0.6, 'Mach number range: $0.28-0.59$')
plt.text(0.073, 0.4, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

F.ReducedElevatorDeflection(eIAS,eTAT,Ws,eThrust,eSthrust,ede,Cmde,CmTc,S,ehp,eET)
F.ReducedElevatorControlForce(eFe,eIAS,eTAT,ehp)



