# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:40:55 2018

@author: ManouschkavanBeek
"""
#from Trim import *
from Cit_par import *
import matplotlib.pyplot as plt
import numpy as np
from isaConstants import *
from math import *

## Defining constants ##

OEW = 9165                              ## lbs
fueltotal = 2567                        ## lbs
S=30                                    ## m**2
Ws=60500
W=60000
b = 1.458*10**(-6)                      
A = 110.4                               ## K
c = 2.0569                              ## m
b      = 15.911	          # wing span [m]
a      = b ** 2 / S      # wing aspect ratio [ ]

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

## ## Importing data from excel ##
f=np.genfromtxt("PFD.csv",delimiter=",")#,dtype='str')
thrust=np.genfromtxt("thrust.dat")

cET=f[:,2][27:33]
chp=f[:,3][27:33]*0.3048
cIAS=(f[:,4][27:33]+2)*1852/3600
ca=f[:,5][27:33]
cFFL=f[:,6][27:33]*0.453592/3600
cFFR=f[:,7][27:33]*0.453592/3600
cFused=f[:,8][27:33]*0.453592
cTAT=f[:,9][27:33]+273.15
eMass=f[:,7][7:15]
mass = sum(eMass)
print 

#print crho 
#print cVe
#print chp
#print cFFL
#print cFFR

def weight_1(cFused):          #returns a list with the total mass of the aircraft in kg
    totalmass = (mass + OEW*0.45359237 + fueltotal*0.45359237 - cFused)*g
    return totalmass

## Laatste entry van array verwijderen, lekker beunen##
#cVt = np.delete(cVt, 6)
#cVe = np.delete(cVe, 6)
#crho = np.delete(crho, 6)
#cM = np.delete(cM, 6)

def C_L(cVe, crho, S):              ## Returns CL for differennt speeds and altitudes
    C_Lhallo = weight_1(cFused)/(0.5*crho*np.square(cVe)*S)
    #print C_L
    return C_Lhallo

#totalmass = weight_1(eMass, cFused)
#print C_L(eVe, erho, S)

#print ehp

def EquivalentAirspeed_CL(cIAS,cTAT, chp):          ##Input: Vc and TAT and pressure altitude
                                                    ##Output: TAS, EAS, rho and Mach number
    p=pISA*(1+Lambda*chp/TISA)**(-g0/(Lambda*R))

    M1 = (1+(sh-1)/(2*sh)*rhoISA/pISA*np.square(cIAS))**(sh/(sh-1))-1
    M2 = (1+pISA/p*M1)**((sh-1)/sh)-1
    M3 = 2/(sh-1)*M2
    M = np.sqrt(M3)
    
    T = cTAT/(1+(sh-1)/2*M**2)

    a=np.sqrt(sh*R*T)

    Vt=M*a

    rho=p/(R*T)

    Ve=Vt*np.sqrt(rho/rhoISA) #The equivalent airspeed
    return Vt,Ve,rho,M,T



cVt=EquivalentAirspeed_CL(cIAS,cTAT,chp)[0]
cVe=EquivalentAirspeed_CL(cIAS,cTAT,chp)[1]
crho=EquivalentAirspeed_CL(cIAS,cTAT,chp)[2]
cM=EquivalentAirspeed_CL(cIAS,cTAT,chp)[3]
cT=EquivalentAirspeed_CL(cIAS,cTAT,chp)[4]
cThrust=thrust[:,0]+thrust[:,1]
#print cThrust
DTISA = cTAT - (TISA + Lambda*chp)
#print chp
print cM
#print DTISA
#print cFFL
#print cFFR
def Reynolds(crho, cVt, cT):
    mu = b*cT**(3/2)/(cT+A)
    Re = cVt*c*crho/mu
    return Re

print Reynolds(crho, cVt, cT)
   
## Test by printing ##
#print cVt
#print cVe
#print crho
#print cM
#print cVer
#print cThrust
#print DTISA
#print chp

#Tc=cThrust/(1/2*crho*cVt**2*S)
#Tcs=rhoISA*cVer**2*CD/(crho*cVt**2)

def ReducedEquivalent(Ve,Ws,W):
    Ver=Ve*np.sqrt(Ws/W)
    return Ver
    
cVer=ReducedEquivalent(cVe,Ws,W)

def DragCoefficient(cThrust,cVe,crho):          #input: Thrust, EAS and rho
                                                #output: drag coefficient
    a = np.square(cVe)*S*1/2*crho
    C_D = cThrust/a
    return C_D
    
#print DragCoefficient(cThrust,cVe,crho)

## make a fit for the C_D_0 calculation ##
C_D_0 = np.polyfit(DragCoefficient(cThrust,cVe,crho), C_L(cVe, crho, S)**2, 1)
#print C_D_0
p = np.poly1d(C_D_0)
C_D_0_correct = -C_D_0[1]/C_D_0[0]
#print C_D_0_correct

## Oswald efficiency factor ##
e = C_L(cVe, crho, S)**2/(DragCoefficient(cThrust,cVe,crho))/(pi*a)
#print e

## Plot all the graphs ##
#CL-alpha graph
plt.plot(ca, C_L(cVe, crho, S))
plt.title('$C_L-alpha$ curve, clean configuration')
plt.ylabel('Lift coefficient')
plt.xlabel('Angle of attack in degrees')
plt.text(5.0, 0.7, 'Mach number range: $0.28-0.59$')
plt.text(4.5, 0.6, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

#CL-CD graph
plt.plot(C_L(cVe, crho, S), DragCoefficient(cThrust,cVe,crho))
plt.title('$C_L-C_D$ curve, clean configuration')
plt.ylabel('Drag coefficient')
plt.xlabel('Lift coefficient')
plt.text(1.1, 0.055, 'Mach number range: $0.28-0.59$')
plt.text(1.0, 0.05, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

#CD-alpha graph
plt.plot(ca, DragCoefficient(cThrust,cVe,crho))
plt.title('$C_D-alpha$ curve, clean configuration')
plt.ylabel('Drag coefficient')
plt.xlabel('Angle of attack in degrees')
plt.text(5.0, 0.055, 'Mach number range: $0.28-0.59$')
plt.text(4.5, 0.05, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()

#CL**2-CD graph
plt.plot(DragCoefficient(cThrust,cVe,crho), C_L(cVe, crho, S)**2)
plt.plot(DragCoefficient(cThrust,cVe,crho), p(DragCoefficient(cThrust,cVe,crho)))
plt.title('$C_L^2-C_D$ curve, clean configuration')
plt.ylabel('Lift coefficient squared')
plt.xlabel('Drag coefficient')
plt.text(0.078, 0.6, 'Mach number range: $0.28-0.59$')
plt.text(0.073, 0.4, 'Reynols number range: $1.04\cdot10^8-2.11\cdot10^8$')
plt.show()