import numpy as np
import Xcg as ding

# Citation 550 - Linear simulation

# xcg = 0.25 * c

# Stationary flight condition

hp0    = 1      	      # pressure altitude in the stationary flight condition [m]
V0     = 221*0.514444            # true airspeed in the stationary flight condition [m/sec]
alpha0 = 1.8/180*np.pi            # angle of attack in the stationary flight condition [rad]
th0    = 1/180*np.pi            # pitch angle in the stationary flight condition [rad]

# Aircraft mass
m      = (9165+2567+822-658)*0.453592            # mass [kg]

# aerodynamic properties
CD0    = 0.04            # Zero lift drag coefficient [ ] from cl-cd curve
CLa    = 5.084            # Slope of CL-alpha curve [ ] from cl-cd curve
e      = 0.8            # Oswald factor [ ]

# Aircraft geometry

S      = 30.00	          # wing area [m^2]
Sh     = 0.2 * S         # stabiliser area [m^2]
Sh_S   = Sh / S	          # [ ]
lh     = 0.71 * 5.968    # tail length [m]
c      = 2.0569	          # mean aerodynamic cord [m]
lh_c   = lh / c	          # [ ]
b      = 15.911	          # wing span [m]
bh     = 5.791	          # stabilser span [m]
A      = b ** 2 / S      # wing aspect ratio [ ]
Ah     = bh ** 2 / Sh    # stabilser aspect ratio [ ]
Vh_V   = 1	          # [ ]
ih     = -2 * np.pi / 180   # stabiliser angle of incidence [rad]

# Constant values concerning atmosphere and gravity

rho0   = 1.2250          # air density at sea level [kg/m^3] 
labda = -0.0065         # temperature gradient in ISA [K/m]
Temp0  = 288.15          # temperature at sea level in ISA [K]
R      = 287.05          # specific gas constant [m^2/sec^2K]
g      = 9.81            # [m/sec^2] (gravity constant)

# air density [kg/m^3]  
rho    = rho0 * np.power( ((1+(labda * hp0 / Temp0))), (-((g / (labda*R)) + 1)))
W      = m * g            # [N]       (aircraft weight)
# Constant values concerning aircraft inertia

muc    = m / (rho * S * c)
mub    = m / (rho * S * b)
KX2    = 0.019
KZ2    = 0.042
KXZ    = 0.002
KY2    = 1.25 * 1.114

# Aerodynamic constants

Cmac   = 0                      # Moment coefficient about the aerodynamic centre [ ]
CNwa   = CLa                    # Wing normal force slope [ ]
CNha   = 2 * np.pi * Ah / (Ah + 2) # Stabiliser normal force slope [ ]
depsda = 4 / (A + 2)            # Downwash gradient [ ]

# Longitudinal stability
Cma    = CNwa * (ding.cgloc(1) - 261.56*0.0254-0.25*c)/c - CNha*(1-depsda)*Vh_V**2*Sh*lh/S/c            # longitudinal stabilty [ ]
Cmde   = 1            # elevator effectiveness [ ]

# Lift and drag coefficient

CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (np.pi * A * e) # Drag coefficient [ ]

# Stabiblity derivatives

CX0    = W * np.sin(th0) / (0.5 * rho * V0 ** 2 * S)
CXu    = -0.02792
CXa    = -0.47966
CXadot = +0.08330
CXq    = -0.28170
CXde   = -0.03728

CZ0    = -W * np.cos(th0) / (0.5 * rho * V0 ** 2 * S)
CZu    = -0.37616
CZa    = -5.74340
CZadot = -0.00350
CZq    = -5.66290
CZde   = -0.69612

Cmu    = +0.06990
Cmadot = +0.17800
Cmq    = -8.79415

CYb    = -0.7500
CYbdot =  0     
CYp    = -0.0304
CYr    = +0.8495
CYda   = -0.0400
CYdr   = +0.2300

Clb    = -0.10260
Clp    = -0.71085
Clr    = +0.23760
Clda   = -0.23088
Cldr   = +0.03440

Cnb    =  +0.1348
Cnbdot =   0     
Cnp    =  -0.0602
Cnr    =  -0.2061
Cnda   =  -0.0120
Cndr   =  -0.0939



# Values we define ourself;
TOW = (9165+2567+822)*0.453592*g            # mass [kg]
Ws  = 60500 #[N]
CmTc = -0.0064
