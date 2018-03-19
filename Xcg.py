import numpy as np
#from Cit_par import *

#datum line is nose of the aircraft
def cgloc(state):
    m_payload = 95+92+65+69+76+77+80+91+110+67

    if state == 1:      #normal state
        Xcg_payload = 0.0254*(131*(95+92) + 214*(65+69) + 251*(76+77) + 288*(80+91) + 170*(110+67))/m_payload
    if state == 2:      #moved cg
        Xcg_payload = 0.0254*(131*(95+92) + 150*91 + 214*(65+69) + 251*(76+77) + 288*(80) + 170*(110+67))/m_payload

    m_fuel = 2567*0.453592

    Xcg_fuel = ((-713100+741533)*0.67+713100)/2567*0.0254

    m_BEM = 9165*0.453592

    Xcg_BEM = 292.18*0.0254

    Xcg = (m_payload*Xcg_payload +m_fuel*Xcg_fuel + m_BEM*Xcg_BEM)/(m_payload+m_fuel+m_BEM)
    return Xcg


#---------------------------------------longitudinal stability ----------------------

Xcg1 = cgloc(1)
Xcg2 = cgloc(2)
Xcgdiff=Xcg1-Xcg2
