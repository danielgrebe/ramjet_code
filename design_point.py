import numpy as np
import iso_1D as iso
from ambiance import Atmosphere
from scipy.optimize import fsolve

GAMMA_COMBUSTION = 1.3  # Gamma des gaz de combustion
MASSE_MOLAIRE_COMBUSTION = 27.6
TEMP_COMBUSTION = 2400  # K
MACH_COMBUSTION = 0.5
M_FIN_COMBUSTION = 0.85
g_0 = 9.81  # N/kg
THRUST = 10e3  # N

# Fonction to approximate the design point of each section of a Ramjet

Altitude = Atmosphere(20000)
P_1 = Altitude.pressure        #Pa
T_1 = Altitude.temperature     #K
Rho_1 = Altitude.density       #kg/m³

# Initialisation
P_0 = 100e3
T_0 = 200
geo = iso.Geometry(iso.DATA_BASENAME)
gas = iso.IdealGas(gamma=1.4)
init_cond0 = iso.InitialConditions(P_0, T_0)
flow = iso.IsoEcoulement(geo, init_cond0, gas)
P_01 = flow._p_ratio_mach(2.8)*P_1
T_01 = flow._T_ratio_mach(2.8)*T_1
#manquerait probablement la densité
rho_01 = gas.density(T_01, P_01)

# Station 1
init_cond = iso.InitialConditions(P_01, T_01)
flow = iso.IsoEcoulement(geo, init_cond, gas)
u_1 = 2.8 * gas.speed_of_sound(T_1)
print('Station 1')
print('P_1/P_0=',P_1/P_01)
print('T_1/T_0=',T_1/T_01)
print('P_1 =',P_1)
print('T_1 =',T_1)
print('u_1 =',u_1)
print('\n')

# Station 2
M_2 = MACH_COMBUSTION
P_2_P_0 = 1 / flow._p_ratio_mach(M_2)
T_2_T_0 = 1 / flow._T_ratio_mach(M_2)
P_2 = P_1*(P_01/P_1)*P_2_P_0
T_2 = T_1*(T_01/T_1)*T_2_T_0
rho_2 = gas.density(T_2, P_2)
u_2 = M_2 * gas.speed_of_sound(T_2)
#T_2 =

print('Station 2')
print('P_2/P_0=',P_2_P_0)
print('T_2/T_0=',T_2_T_0)
print('P_2 = ',P_2)
print('T_2 = ',T_2)
print('rho_2 = ',rho_2)

# Station 3
# On présume combustion isobar
exhaust = iso.IdealGas(gamma=GAMMA_COMBUSTION, 
                       masse_molaire=MASSE_MOLAIRE_COMBUSTION)
exit_flow = iso.IsoEcoulement(geo, init_cond, exhaust)
P_3 = P_2
T_3 = TEMP_COMBUSTION
rho_3 = exhaust.density(T_3, P_3)

# on calcul la vitesse en présumant un section constante
u_3 = rho_2 * u_2 / rho_3
M_3 = u_3 / exhaust.speed_of_sound(T_3)

print('Station 3 (section constante)')
print('P_3 = ',P_3)
print('T_3 = ',T_3)
print('rho_3 = ',rho_3)
print('M_3 = ',M_3)

# on calcul le ratio de section necessaire pour avoir un mach
# fixe à la fin de combustion
M_3 = M_FIN_COMBUSTION
u_3 = M_3 * exhaust.speed_of_sound(T_3)
A_3_A_2 = u_2 * rho_2 / u_3 / rho_3
P_03 = exit_flow._p_ratio_mach(M_3) * P_3
T_03 = exit_flow._T_ratio_mach(M_3) * T_3

print('Station 3 (Mach fixe)')
print('M_3 = ',M_3)
print('P_3 = ',P_3)
print('T_3 = ',T_3)
print('rho_3 = ',rho_3)
print('A_3_A_2 = ',A_3_A_2)

# Station 4
P_4 = P_1
P_04 = P_03
T_04 = T_03
M_4 = fsolve(lambda m: exit_flow._p_ratio_mach(m) - P_04 / P_4, 3)
T_4 = T_04 / exit_flow._T_ratio_mach(M_4)
A_4_A_star = exit_flow._area_ratio_mach(M_4)
u_4 = M_4 * exhaust.speed_of_sound(T_4)

print('Station 4')
print('M_4 = ',M_4)
print('P_4 = ',P_4)
print('T_4 = ',T_4)
print('u_4 = ',u_4)
print('A_4_A_star = ',A_4_A_star)

# Résumé
isp = (u_4 - u_1) / g_0
debit = THRUST / g_0 / isp

print("Résumé")
print("Ce ramjet idéale a une ISP de {:.0f} s.".format(isp[0]))
print("Il faut donc un débit d'air de {:.2f} kg/s pour avoir une"
      " poussée de {:.0f} kN".format(debit[0], THRUST * 1e-3))







