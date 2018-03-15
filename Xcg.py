import numpy as np

#datum line is nose of the aircraft

m_payload = 95+92+65+69+76+77+80+91+110+67

Xcg_payload = 0.0254*(131*(95+92) + 214*(65+69) + 251*(76+77) + 288*(80+91) + 170*(110+67))/m_payload
Xcg_payload2 = 0.0254*(131*(95+92) + 150*91 + 214*(65+69) + 251*(76+77) + 288*(80) + 170*(110+67))/m_payload

m_fuel = 2567*0.453592

Xcg_fuel = ((-713100+741533)*0.67+713100)/2567*0.0254

m_BEM = 9165*0.453592

Xcg_BEM = 292.18*0.0254

Xcg = (m_payload*Xcg_payload +m_fuel*Xcg_fuel + m_BEM*Xcg_BEM)/(m_payload+m_fuel+m_BEM)

Xcg2 = (m_payload*Xcg_payload2 +m_fuel*Xcg_fuel + m_BEM*Xcg_BEM)/(m_payload+m_fuel+m_BEM)