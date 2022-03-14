import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import iso_1D as iso
import warnings
from ambiance import Atmosphere
from collections.abc import Iterable

# Fonction to approximate the design point of each section of a Ramjet

Altitude = Atmosphere(20000)
P_1 = Altitude.pressure        #Pa
T_1 = Altitude.temperature     #K
Rho_1 = Altitude.density       #kg/m³

#Initialisation
P_0 = 100e3
T_0 = 200
geo = iso.Geometry(iso.DATA_BASENAME)
gas = iso.IdealGas(gamma=1.4)
init_cond0 = iso.InitialConditions(P_0, T_0)
flow = iso.IsoEcoulement(geo, init_cond0, gas)
P_01 = flow._p_ratio_mach(2.8)*P_1
T_01 = flow._T_ratio_mach(2.8)*T_1
#manquerait probablement la densité

#Station 1
init_cond = iso.InitialConditions(P_01, T_01)
flow = iso.IsoEcoulement(geo, init_cond, gas)
print('Station 1')
print('P_1/P_0=',P_1/P_01)
print('T_1/T_0=',T_1/T_01)
print('P_1 =',P_1)
print('T_1 =',T_1)
print('\n')

#Station 2
P_2_P_0 = 1 / flow._p_ratio_mach(0.5)
T_2_T_0 = 1 / flow._T_ratio_mach(0.5)
P_2 = P_1*(P_01/P_1)*P_2_P_0
T_2 = T_1*(T_01/T_1)*T_2_T_0
#T_2 =

print('Station 2')
print('P_2/P_0=',P_2_P_0)
print('T_2/T_0=',T_2_T_0)
print('P_2 = ',P_2)
print('T_2 = ',T_2)