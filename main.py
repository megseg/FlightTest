from D import *
from Cit_par import *
import F

Cmde = F.ElevatorEffectiveness()
Cma = F.LongitudinalStability(ede,ea,Cmde)

F.ReducedElevatorDeflection(eIAS,eTAT,Ws,eThrust,eSthrust,ede,Cmde,CmTc,S,ehp,eET)
F.ReducedElevatorControlForce(eFe,eIAS,eTAT,ehp)
