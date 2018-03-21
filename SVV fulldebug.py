# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

p0 = 101325
T0 = 288.15
rh0 = 1.225
g0 = 9.81
R = 287.05 
aisa = -0.0065
wplane = 9165
wfuel = 4050
wpass = 712 * 2.20462262
gam = 1.4
S = 30

b = pd.read_excel('SVVtest1.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
#print b
a = b.values
#print a

hm = (np.array([a[0,0],a[1,0],a[2,0],a[3,0],a[4,0],a[5,0]]))*0.3048
#print hm

T1 = T0 - (0.0065*hm)
print T1
T = a[:,6] + 273.15
print T

p = p0*((T/T0)**(-g0/(aisa*R)))
#print p

rh = rh0*(T/T0)**(-((g0/(aisa*R))+1))
#print rh

wused = (np.array([a[0,5],a[1,5],a[2,5],a[3,5],a[4,5],a[5,5]]))
#print wused

w = ((wplane + wfuel + wpass - wused) / 2.20462262) * g0
#print w

vc = np.array([(a[0,1]*0.51444444444),(a[1,1]*0.51444444444),(a[2,1]*0.51444444444),(a[3,1]*0.51444444444),(a[4,1]*0.51444444444),(a[5,1]*0.51444444444)])
#vc1 = np.array([(vc[0]**2),(vc[1]**2),(vc[2]**2),(vc[3]**2),(vc[4]**2),(vc[5]**2)])
#print vc1
vc1 = np.square(vc)
#print vc2
#M = math.sqrt(((2/(gam-1))*((1+(p0/p)*((1+(((gam-1)*rh0*(vc2))/(2*gam*p0))-1)))**((gam-1)/gam)))-1)
#print M

Mpart1 = 2/(gam-1)
Mpart2 = ((1 + ((gam-1)/(2*gam))*(rh0/p0)*vc1)**(gam/(gam-1)))-1
Mpart3 = ((1 + ((p0/p)*Mpart2))**((gam-1)/gam))
Mpart4 = Mpart1*(Mpart3 - 1)
#M1 = math.sqrt(Mpart4[0])
#M2 = math.sqrt(Mpart4[1])
#M3 = math.sqrt(Mpart4[2])
#M4 = math.sqrt(Mpart4[3])
#M5 = math.sqrt(Mpart4[4])
#M6 = math.sqrt(Mpart4[5])
#M = np.array([M1, M2, M3, M4, M5, M6])
M = np.sqrt(Mpart4)
#print "Mach numbers are" 
#print M

MT = (T/(1+(((gam-1)/2)*M**2)))
#print MT
#MT1 = np.array([math.sqrt(gam*R*MT[0]),math.sqrt(gam*R*MT[1]),math.sqrt(gam*R*MT[2]),math.sqrt(gam*R*MT[3]),math.sqrt(gam*R*MT[4]),math.sqrt(gam*R*MT[5])])
#print "Mach speeds are" 
#print MT1
MT2 = np.sqrt(gam * R * MT)
#print MT2
VT = M * MT2
#print "VT values are" 
#print VT

#print "VE values are" 
VE = VT * (np.sqrt(rh/rh0))
#print VE

Cl = (2 * w) / (rh * ((VE)**2) * S )
#print Cl

plotalpha = np.array(a[:,2]) *  0.01745329252
#print plotalpha
plotalphaang = np.array(a[:,2])

#plt.plot(plotalpha, Cl, 'ro')
#plt.show()
plt.plot(plotalphaang, Cl, 'ro')
plt.xlabel("Alpha (deg)")
plt.ylabel("CL (-)")
plt.ylim(ymin=0)
plt.savefig('clalpha')
plt.show()
plt.close()

clgradient = (Cl[5]-Cl[0])/ (plotalpha[5]-plotalpha[0])
#print "Cl/a (rad)"
#print clgradient
clgradientang = (Cl[5]-Cl[0]) / (a[5,2]-a[0,2])
#print "Cl/a (angle)"
#print clgradientang

T_corr1 = T0 - (0.0065*hm)
T_corr = T - T_corr1
print 'joe'
print T_corr

ff_l = (np.array(a[:,3]))/(3600*2.20462262)
ff_r = (np.array(a[:,4]))/(3600*2.20462262)

Thrust = np.array([[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],])

def thrustexe(hm,M,T_corr,ff_l,ff_r):
    i = 0
    for i in range (6):
        lst = np.array([hm[i],M[i],T_corr[i],ff_l[i],ff_r[i]])
        np.savetxt("matlab.dat",lst)
        os.system("thrust.exe")
        result = np.genfromtxt("thrust.dat")
        henk = result[0]
        piet = result[1]
        #print henk
        #print piet
        Thrust[i,0] = henk
        Thrust[i,1] = piet
        #print result
        if i == 5:
            return result
        
thrustexe(hm,M,T_corr,ff_l,ff_r)
#print Thrust
th_total = np.array([(Thrust[0,0] + Thrust[0,1]),(Thrust[1,0] + Thrust[1,1]),(Thrust[2,0] + Thrust[2,1]),(Thrust[3,0] + Thrust[3,1]),(Thrust[4,0] + Thrust[4,1]),(Thrust[5,0] + Thrust[5,1])])

#print 'thrust'
#print th_total
#plt.plot(plotalphaang, th_total, 'ro')
#plt.show

Cd = (2 * th_total) / (rh * ((VE)**2) * S )
#print 'Cd'
#print Cd

#plt.plot(plotalphaang, Cd, 'ro')
#plt.show()

cdgradient = (Cd[5]-Cd[0])/ (plotalpha[5]-plotalpha[0])
#print "Cd/a (rad)"
#print cdgradient
cdgradientang = (Cd[5]-Cd[0]) / (a[5,2]-a[0,2])
#print "Cd/a (angle)"
#print cdgradientang

#plt.plot((Cl**2), Cd, 'ro')
#plt.show()

clcd = Cl / Cd
#print clcd
#plt.plot(plotalphaang, clcd, 'ro')
#plt.show()
Cl2 = Cl**2

clcdplot = Cl2 / Cd
#plt.plot(plotalphaang, clcdplot, 'ro')
#plt.plot(plotalphaang, m*plotalphaang + c, 'r')
#plt.show()

citparS = 30
citparb = 15.911
aspect = (citparb **2) / citparS
#print aspect

#linear regression of Cl^2 / Cd
Cl3 = np.vstack([Cl2, np.ones(len(Cl2))]).T
m, c = np.linalg.lstsq(Cl3, Cd)[0]
#print (m, c)
#gradclcd = ((Cd[5] - Cd[0]) / ((Cl[5]**2) - (Cl[0]**2)))
#gradclcd = (clcdplot[5] - clcdplot[0]) / (plotalpha[5] - plotalpha[0])
#print gradclcd
oswald = 1 / (m * np.pi * aspect)
#print oswald

cd0 = Cd - ((Cl**2)/(np.pi * aspect * oswald))
#print 'cd0: '
#print cd0
plt.plot(Cl2, Cd, 'ro')
plt.plot(Cl2, m*Cl2 + c, 'g')
plt.xlabel("CL^2 (-)")
plt.ylabel("CD (-)")
plt.ylim(ymin=0)
plt.savefig('clcdcurve')
plt.show()
plt.close()

excel = pd.DataFrame({'Equivalent airspeed': VE,
                      'Thrust': th_total,
                      'Cl': Cl,
                      'Cd': Cd,
                      'cd0': cd0})
excel.to_excel('svvresults.xlsx', sheet_name = 'Sheet1')

d = pd.read_excel('SVVtest1.xlsx', 'Sheet2', index_col=None, na_values=['NA'])
print d
c = d.values

hm = c[:,0]*0.3048
#print hm

T = T0 - (0.0065*hm)
#print T

p = p0*((T/T0)**(-g0/(aisa*R)))
#print p

rh = rh0*(T/T0)**(-((g0/(aisa*R))+1))
#print rh

wused = c[:,8]
#print wused

w = ((wplane + wfuel + wpass - wused) / 2.20462262) * g0
#print w

vc = c[:,1]*0.51444444444
#print vc
#vc1 = np.array([(vc[0]**2),(vc[1]**2),(vc[2]**2),(vc[3]**2),(vc[4]**2),(vc[5]**2)])
#print vc1
vc1 = np.square(vc)
#print vc2
#M = math.sqrt(((2/(gam-1))*((1+(p0/p)*((1+(((gam-1)*rh0*(vc2))/(2*gam*p0))-1)))**((gam-1)/gam)))-1)
#print M

Mpart1 = 2/(gam-1)
Mpart2 = ((1 + ((gam-1)/(2*gam))*(rh0/p0)*vc1)**(gam/(gam-1)))-1
Mpart3 = ((1 + ((p0/p)*Mpart2))**((gam-1)/gam))
Mpart4 = Mpart1*(Mpart3 - 1)
#M1 = math.sqrt(Mpart4[0])
#M2 = math.sqrt(Mpart4[1])
#M3 = math.sqrt(Mpart4[2])
#M4 = math.sqrt(Mpart4[3])
#M5 = math.sqrt(Mpart4[4])
#M6 = math.sqrt(Mpart4[5])
#M = np.array([M1, M2, M3, M4, M5, M6])
M = np.sqrt(Mpart4)
#print "Mach numbers are" 
#print M

MT = (T/(1+(((gam-1)/2)*M**2)))
#print MT
#MT1 = np.array([math.sqrt(gam*R*MT[0]),math.sqrt(gam*R*MT[1]),math.sqrt(gam*R*MT[2]),math.sqrt(gam*R*MT[3]),math.sqrt(gam*R*MT[4]),math.sqrt(gam*R*MT[5])])
#print "Mach speeds are" 
#print MT1
MT2 = np.sqrt(gam * R * MT)
#print MT2
VT = M * MT2
#print "VT values are" 
#print VT

#print "VE values are" 
VE = VT * (np.sqrt(rh/rh0))
#print VE

Cl = (2 * w) / (rh * ((VE)**2) * S )
#print Cl

plotalpha = c[:,2] *  0.01745329252
#print plotalpha
plotalphaang = c[:,2]

#plt.plot(plotalpha, Cl, 'ro')
#plt.show()
#plt.plot(plotalphaang, Cl, 'ro')
#plt.show()

clgradient = (Cl[6]-Cl[0])/ (plotalpha[6]-plotalpha[0])
#print "Cl/a (rad)"
#print clgradient
clgradientang = (Cl[6]-Cl[0]) / (c[6,2]-c[0,2])
#print "Cl/a (angle)"
#print clgradientang

T_corr1 = c[:,9]+273.15
T_corr = T - T_corr1
#print T_corr

ff_l = c[:,6]/(3600*2.20462262)
ff_r = c[:,7]/(3600*2.20462262)

Thrust = np.array([[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],])

def thrustexe(hm,M,T_corr,ff_l,ff_r):
    i = 0
    for i in range (7):
        lst = np.array([hm[i],M[i],T_corr[i],ff_l[i],ff_r[i]])
        np.savetxt("matlab.dat",lst)
        os.system("thrust.exe")
        result = np.genfromtxt("thrust.dat")
        henk = result[0]
        piet = result[1]
        #print henk
        #print piet
        Thrust[i,0] = henk
        Thrust[i,1] = piet
        #print result
        if i == 6:
            return result
        
thrustexe(hm,M,T_corr,ff_l,ff_r)
#print Thrust
#th_total = np.array([(Thrust[0,0] + Thrust[0,1]),(Thrust[1,0] + Thrust[1,1]),(Thrust[2,0] + Thrust[2,1]),(Thrust[3,0] + Thrust[3,1]),(Thrust[4,0] + Thrust[4,1]),(Thrust[5,0] + Thrust[5,1]),(Thrust[6,0] + Thrust[6,1])])
#print th_total
th_total = np.sum(Thrust[:,:], axis=1)
#print th2

#print 'thrust'
#print th_total
#plt.plot(plotalphaang, th_total, 'ro')
#plt.show

Cd = (2 * th_total) / (rh * ((VE)**2) * S )
#print 'Cd'
#print Cd

#plt.plot(plotalphaang, Cd, 'ro')
#plt.show()

cdgradient = (Cd[6]-Cd[0])/ (plotalpha[6]-plotalpha[0])
#print "Cd/a (rad)"
#print cdgradient
cdgradientang = (Cd[6]-Cd[0]) / (c[6,2]-c[0,2])
#print "Cd/a (angle)"
#print cdgradientang

#plt.plot((Cl**2), Cd, 'ro')
#plt.show()

clcd = Cl / Cd
#print clcd
#plt.plot(plotalphaang, clcd, 'ro')
#plt.show()
Cl2 = Cl**2

clcdplot = Cl2 / Cd
#plt.plot(plotalphaang, clcdplot, 'ro')
#plt.plot(plotalphaang, m*plotalphaang + c, 'r')
#plt.show()


#linear regression of Cl^2 / Cd
Cl3 = np.vstack([Cl2, np.ones(len(Cl2))]).T
m, c = np.linalg.lstsq(Cl3, Cd)[0]
#print (m, c)
#gradclcd = ((Cd[5] - Cd[0]) / ((Cl[5]**2) - (Cl[0]**2)))
#gradclcd = (clcdplot[5] - clcdplot[0]) / (plotalpha[5] - plotalpha[0])
#print gradclcd
oswald = 1 / (m * np.pi * aspect)
#print oswald

cd0 = Cd - ((Cl**2)/(np.pi * aspect * oswald))
#print 'cd0: '
#print cd0
#plt.plot(Cl2, Cd, 'ro')
#plt.plot(Cl2, m*Cl2 + c, 'g')
#plt.show()

W_s = 60500
mdot_fsl = 0.048
mdot_fsr = 0.048
V_Etilda = VE * np.sqrt(W_s / w)
#print w
#print VE
#print V_Etilda
Thrustcorr = np.array([[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],])

def thrustexecorr(hm,M,T_corr,mdot_fsl,mdot_fsr):
    i = 0
    for i in range (7):
        lst = np.array([hm[i],M[i],T_corr[i],mdot_fsl,mdot_fsr])
        np.savetxt("matlab.dat",lst)
        os.system("thrust.exe")
        result = np.genfromtxt("thrust.dat")
        henk = result[0]
        piet = result[1]
        #print henk
        #print piet
        Thrustcorr[i,0] = henk
        Thrustcorr[i,1] = piet
        #print result
        if i == 6:
            return result

thrustexecorr(hm,M,T_corr,mdot_fsl,mdot_fsr)
th_corr = np.sum(Thrustcorr[:,:], axis=1)
print th_corr

thcorr = np.sqrt(th_corr/th_total)
print thcorr





d = pd.read_excel('SVVtest1.xlsx', 'Sheet2', index_col=None, na_values=['NA'])
print d
c = d.values

hm = c[:,0]*0.3048
T = c[:,9] + 273.15
p = p0*((T/T0)**(-g0/(aisa*R)))
rh = rh0*(T/T0)**(-((g0/(aisa*R))+1))
wused = c[:,8]
w = ((wplane + wfuel + wpass - wused) / 2.20462262) * g0
    
vc = c[:,1]*0.51444444444
vc1 = np.square(vc)

Mpart1 = 2/(gam-1)
Mpart2 = ((1 + ((gam-1)/(2*gam))*(rh0/p0)*vc1)**(gam/(gam-1)))-1
Mpart3 = ((1 + ((p0/p)*Mpart2))**((gam-1)/gam))
Mpart4 = Mpart1*(Mpart3 - 1)
M = np.sqrt(Mpart4)
#print "Mach numbers are" 
#print M

MT = (T/(1+(((gam-1)/2)*M**2)))
MT2 = np.sqrt(gam * R * MT)
VT = M * MT2
VE = VT * (np.sqrt(rh/rh0))
#print VE

Cl = (2 * w) / (rh * ((VE)**2) * S )
#print Cl

plotalpha = c[:,2] *  0.01745329252
#print plotalpha
plotalphaang = c[:,2]

#plt.plot(plotalpha, Cl, 'ro')
#plt.show()
#plt.plot(plotalphaang, Cl, 'ro')
#plt.show()
Clgr = np.vstack([Cl, np.ones(len(Cl))]).T
m, ca = np.linalg.lstsq(Clgr, plotalphaang)[0]
print (m, ca)
#plt.plot(plotalphaang, Cl, 'ro')
#plt.plot(m*Cl + ca, Cl, 'g')
#plt.show()
alpha_0 = ca

clgradient = (Cl[6]-Cl[0])/ (plotalpha[6]-plotalpha[0])
#print "Cl/a (rad)"
#print clgradient
clgradientang = (Cl[6]-Cl[0]) / (c[6,2]-c[0,2])
#print "Cl/a (angle)"
#print clgradientang

T_corr1 = T0 - (0.0065*hm)
T_corr = T - T_corr1
#print T_corr

ff_l = c[:,6]/(3600*2.20462262)
ff_r = c[:,7]/(3600*2.20462262)

Thrust = np.array([[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],])

def thrustexe(hm,M,T_corr,ff_l,ff_r):
    i = 0
    for i in range (7):
        lst = np.array([hm[i],M[i],T_corr[i],ff_l[i],ff_r[i]])
        np.savetxt("matlab.dat",lst)
        os.system("thrust.exe")
        result = np.genfromtxt("thrust.dat")
        henk = result[0]
        piet = result[1]
        #print henk
        #print piet
        Thrust[i,0] = henk
        Thrust[i,1] = piet
        #print result
        if i == 6:
            return result
        
thrustexe(hm,M,T_corr,ff_l,ff_r)
#print Thrust
#th_total = np.array([(Thrust[0,0] + Thrust[0,1]),(Thrust[1,0] + Thrust[1,1]),(Thrust[2,0] + Thrust[2,1]),(Thrust[3,0] + Thrust[3,1]),(Thrust[4,0] + Thrust[4,1]),(Thrust[5,0] + Thrust[5,1]),(Thrust[6,0] + Thrust[6,1])])
#print th_total
th_total = np.sum(Thrust[:,:], axis=1)
#print th2

#print 'thrust'
#print th_total
#plt.plot(plotalphaang, th_total, 'ro')
#plt.show

Cd = (2 * th_total) / (rh * ((VE)**2) * S )
#print 'Cd'
#print Cd

#plt.plot(plotalphaang, Cd, 'ro')
#plt.show()

cdgradient = (Cd[6]-Cd[0])/ (plotalpha[6]-plotalpha[0])
#print "Cd/a (rad)"
#print cdgradient
cdgradientang = (Cd[6]-Cd[0]) / (c[6,2]-c[0,2])
#print "Cd/a (angle)"
#print cdgradientang

#plt.plot((Cl**2), Cd, 'ro')
#plt.show()

clcd = Cl / Cd
#print clcd
#plt.plot(plotalphaang, clcd, 'ro')
#plt.show()
Cl2 = Cl**2

clcdplot = Cl2 / Cd
#plt.plot(plotalphaang, clcdplot, 'ro')
#plt.plot(plotalphaang, m*plotalphaang + c, 'r')
#plt.show()


#linear regression of Cl^2 / Cd
Cl3 = np.vstack([Cl2, np.ones(len(Cl2))]).T
m, cad = np.linalg.lstsq(Cl3, Cd)[0]
#print (m, c)
#gradclcd = ((Cd[5] - Cd[0]) / ((Cl[5]**2) - (Cl[0]**2)))
#gradclcd = (clcdplot[5] - clcdplot[0]) / (plotalpha[5] - plotalpha[0])
#print gradclcd
oswald = 1 / (m * np.pi * aspect)
#print oswald

cd0 = Cd - ((Cl**2)/(np.pi * aspect * oswald))
#print 'cd0: '
#print cd0
#plt.plot(Cl2, Cd, 'ro')
#plt.plot(Cl2, m*Cl2 + c, 'g')
#plt.show()

W_s = 60500
mdot_fsl = 0.048
mdot_fsr = 0.048
T_corr1 = T0 - (0.0065*hm)
T_corr = T - T_corr1
V_Etilda = VE * np.sqrt(W_s / w)
#print w
#print VE
#print V_Etilda
Thrustcorr = np.array([[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],])

def thrustexecorr(hm,M,T_corr,mdot_fsl,mdot_fsr):
    i = 0
    for i in range (7):
        lst = np.array([hm[i],M[i],T_corr[i],mdot_fsl,mdot_fsr])
        np.savetxt("matlab.dat",lst)
        os.system("thrust.exe")
        result = np.genfromtxt("thrust.dat")
        henk = result[0]
        piet = result[1]
        #print henk
        #print piet
        Thrustcorr[i,0] = henk
        Thrustcorr[i,1] = piet
        #print result
        if i == 6:
            return result

thrustexecorr(hm,M,T_corr,mdot_fsl,mdot_fsr)
th_corr = np.sum(Thrustcorr[:,:], axis=1)
#print th_corr

cm_delta = -1.1642
cm_0 = 0.0297
cm_alpha = -0.5626
cm_tc = -0.0064
CN_alpha = Cl / ((plotalphaang)-ca)
print CN_alpha

#delta_eeq = (-1/cm_delta)*(cm_0 + ((cm_alpha)))
deltae_meas = c[:,3]
#print deltae_meas
tcs = (th_corr)/(0.5 * rh0 * (V_Etilda**2) * 2 * (0.868**2))
print tcs
tc = (th_total)/(0.5 * rh * (VE**2) * 2 * (0.868**2))
print tc
deltae_eq = deltae_meas - ((1/cm_delta)*cm_tc*(tcs-tc)) 
print deltae_eq
plt.gca().invert_yaxis()
plt.plot(V_Etilda, deltae_eq, 'ro')
plt.xlabel("VE (m/s)")
plt.ylabel("Elevator trim (deg)")
plt.savefig('elevtrim.png')
plt.show()
plt.close()

F_E = c[:,5] * (W_s / w)
print F_E
plt.gca().invert_yaxis()
plt.plot(V_Etilda, F_E, 'ro')
plt.xlabel("VE (m/s)")
plt.ylabel("Stick Force (N)")
plt.savefig('elevcontrol.png')
plt.show()
plt.close()





d1 = pd.read_excel('SVVtest1.xlsx', 'Sheet3', index_col=None, na_values=['NA'])
print d1
c1 = d1.values
#print c1

hm = c1[:,0]*0.3048
T = c1[:,9] + 273.15
p = p0*((T/T0)**(-g0/(aisa*R)))
rh = rh0*(T/T0)**(-((g0/(aisa*R))+1))
print rh
wused = c1[:,8]
w = ((wplane + wfuel + wpass - wused) / 2.20462262) * g0
    
vc = c1[:,1]*0.51444444444
vc1 = np.square(vc)

Mpart1 = 2/(gam-1)
Mpart2 = ((1 + ((gam-1)/(2*gam))*(rh0/p0)*vc1)**(gam/(gam-1)))-1
Mpart3 = ((1 + ((p0/p)*Mpart2))**((gam-1)/gam))
Mpart4 = Mpart1*(Mpart3 - 1)
M = np.sqrt(Mpart4)
#print "Mach numbers are" 
#print M

MT = (T/(1+(((gam-1)/2)*M**2)))
MT2 = np.sqrt(gam * R * MT)
VT = M * MT2
VE = VT * (np.sqrt(rh/rh0))
print VE

Cl = (w) / (0.5 * rh * ((VE)**2) * S )
#Cl = 0.415
print Cl

T_corr1 = T0 - (0.0065*hm)
T_corr = T - T_corr1
#print T_corr

delta_de = (c1[0,3]-c1[1,3]) * 0.01745329252
#delta_de = 0.6 * 0.01745329252
#delta_de = 0.017


xcgexcel = pd.read_excel('SVVtest1.xlsx', 'Sheet4', index_col=None, na_values=['NA'])
#print xcgexcel
valuesxcg = xcgexcel.values

momentip = valuesxcg[:,1] * valuesxcg[:,2]
#print momentip

totalmomentip = np.sum(momentip[:])
#print totalmomentip
#totalmoment = totalmomentip * 0.112984829
#print totalmoment
weightxcg = np.sum(valuesxcg[:,1])
#print weightxcg
xcgin = totalmomentip / weightxcg
#print xcgin
xcg1 = xcgin * 0.0254
#print xcg1

xcgexcel = pd.read_excel('SVVtest1.xlsx', 'Sheet5', index_col=None, na_values=['NA'])
#print xcgexcel
valuesxcg = xcgexcel.values

momentip = valuesxcg[:,1] * valuesxcg[:,2]
#print momentip

totalmomentip = np.sum(momentip[:])
#print totalmomentip
#totalmoment = totalmomentip * 0.112984829
#print totalmoment
weightxcg = np.sum(valuesxcg[:,1])
#print weightxcg
xcgin = totalmomentip / weightxcg
#print xcgin
xcg2 = xcgin * 0.0254
#print xcg2

delta_xcg = xcg1 - xcg2
#print delta_xcg
chord = 2.0569
cm_delta2 = (-1/delta_de)*Cl*(delta_xcg/chord)
print cm_delta2

alphaslope = np.vstack([c[:,2], np.ones(len(c[:,2]))]).T
m3, c3 = np.linalg.lstsq(alphaslope, c[:,3])[0]
#print m3, c3
#cm_alpha = -cm_delta2*((c[0,3]-c[3,3])/(c[0,2]-c[3,2]))
cm_alpha = -cm_delta2*(m3)
print cm_alpha
