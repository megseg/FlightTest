from math import *
from isaConstants import *
import numpy as np
from D import *
import matplotlib.pyplot as plt
from Cit_par import *
import scipy as sc
import Xcg
from scipy import interpolate
S1=110.4

# Define function that returns weight in function of time
def Weight(TOW,time):
    Burnt=interpolate.interp1d(ET,Fused)
    W = TOW - Burnt(time)
    return W
    
def C_L(time, Vt, rho, S):              ## Returns CL for differennt speeds and altitudes
    C = Weight(TOW,ctime)/(0.5*rho*Vt**2*S)
    return C

def Reynolds(rho, Vt, T):
    mu = 1.458*10**(-6)*T**(3/2)/(T+S1)
    Re = Vt*c*rho/mu
    return Re


# Outputs Vt, Ve, rho and M (in that order). Takes in values in SI units with Vc being the calibrated airspeed, Tm being the total measured temperature and hp being the pressure height
def EquivalentAirspeed(Vc,Tm,hp):
    p=pISA*(1+Lambda*hp/TISA)**(-g0/(Lambda*R))

    M1=(1+(sh-1)/(2*sh)*rhoISA/pISA*Vc**2)**(sh/(sh-1))
    M2=1+pISA/p*(M1-1)
    M3=2/(sh-1)*(M2**((sh-1)/sh)-1)
    M=np.sqrt(M3)

    T = Tm/(1+(sh-1)/2*M**2)

    a=np.sqrt(sh*R*T)

    Vt=M*a

    rho=p/(R*T)

    Ve=Vt*np.sqrt(rho/rhoISA) #The equivalent airspeed
    return Vt,Ve,rho,M,T


# Reduces the equivalent airspeed by using weight and standard weight ratio
def ReducedEquivalent(Ve,Ws,W):
    Ver=Ve*np.sqrt(Ws/W)
    return Ver


# Calculates and outputs the trust coefficient Tc
def ThrustCoefficient(ethrust,eIAS,Tm,ehp):
    rho=EquivalentAirspeed(eIAS,Tm,ehp)[2]
    Vt=EquivalentAirspeed(eIAS,Tm,ehp)[0]
    tc=ethrust/(1/2*rho*Vt**2*S)
    return tc


# Calculates and outputs the standard thrust coefficient Tcs
def StandardThrustCoefficient(eSthrust,eIAS,Tm,ehp):
    rho=EquivalentAirspeed(eIAS,Tm,ehp)[2]
    Vt=EquivalentAirspeed(eIAS,Tm,ehp)[0]
    tcs=eSthrust/(1/2*rho*Vt**2*S)
    return tcs


# Calculates the elevator effectiveness Cmde and returns it
def ElevatorEffectiveness():
    AverageSpeed=np.mean(deIAS)
    AverageTemperature=np.mean(deTAT)
    AverageHeight=np.mean(dehp)
    Vt=EquivalentAirspeed(AverageSpeed,AverageTemperature,AverageHeight)[0]
    rho=EquivalentAirspeed(AverageSpeed,AverageTemperature,AverageHeight)[2] 
    W=np.mean(Weight(TOW,deET)) 
    CN=W/(1/2*rho*Vt**2*S) 
    Cmde=-1/(dede)*CN*Xcg.Xcgdiff/c
    return Cmde

# Calculates and plots the reduced elevator deflection curve
def ReducedElevatorDeflection(eIAS,Tm,Ws,eThrust,eSthrust,ede,Cmde,CmTc,S,ehp,eET):
    eVt=EquivalentAirspeed(eIAS,Tm,ehp)[0]
    eVe=EquivalentAirspeed(eIAS,Tm,ehp)[1] 
    erho=EquivalentAirspeed(eIAS,Tm,ehp)[2]
    W=Weight(TOW,eET)
    eVer=np.sort(ReducedEquivalent(eVe,Ws,W))

    Tc=ThrustCoefficient(eThrust,eIAS,Tm,ehp) 
    Tcs=StandardThrustCoefficient(eSthrust,eIAS,Tm,ehp)

    eqde=np.sort(ede-CmTc/Cmde*(Tcs-Tc))
    plt.gca().invert_yaxis() 
    plt.plot(eVer,eqde)
    plt.title("Reduced elevator angle in function of the reduced equivalent airspeed",fontsize=15)
    plt.xlabel("Reduced equivalent airspeed [m/s]",fontsize=15)
    plt.ylabel("Reduced elevator angle [deg]",fontsize=15)
    plt.show()

# Calculates and returns the longitudinal stability
def LongitudinalStability(ede,ea,Cmde):
    deda=np.polyfit(ede,ea,1)[0]
    Cma=-deda*Cmde
    return Cma

# Calculates and plots the ReducedElevatorControlForce
def ReducedElevatorControlForce(eFe,Vc,Tm,hp):
    Ve=EquivalentAirspeed(Vc,Tm,hp)[1]
    Ver=ReducedEquivalent(Ve,Ws,Weight(TOW,eET))
    Fer=eFe*Ver**2/Ve**2
    VerFer=np.asarray(np.column_stack((Ver,Fer)))
    VerFerSorted=VerFer[VerFer[:,0].argsort()]
    plt.gca().invert_yaxis() 
    plt.title("Reduced elevator control force curve",fontsize=15) 
    plt.xlabel("Reduced equivalent airspeed [m/s]",fontsize=15)
    plt.ylabel("Reduced control force [N]",fontsize=15)
    plt.plot(VerFerSorted[:,0],VerFerSorted[:,1])
    plt.show()



