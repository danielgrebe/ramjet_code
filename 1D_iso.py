#!python

import numpy as	np
# import plotly.graph_objs as go
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

# Propriétés du gaz
CO2 = 0.1  # fraction massique du CO_2
MASSE_MOLAIRE = 2*14.5e-3  # kg/mol
IDEAL_GAS = 8.314  # J/(K*mol)
#R = IDEAL_GAS / MASSE_MOLAIRE
GAMMA = 1.4
C_p = 1.006e3  # J/(kg*K)
C_v = C_p / GAMMA  # J/(kg*K)
R = C_p -C_v
H_s = CO2 * 591e3  # J/kg enthalpy of sublimation du mélange

# Conditions initials
P_0 = 2e5  # Pa
T_0 = 300  # K
u_0 = 0  # m/s
rho_0 = P_0 / (R * T_0)

# Géométrie
H_star = 0.005  # m
X_star = 0.05
# Uncomment following lines for Rayleigh flow
# X_points = np.array([0, X_star, 0.17, 0.23])  # m
# H_points = np.array([0.05, H_star, 0.024, 0.024])  # m
X_points = np.array([0, X_star, 0.17])  # m
H_points = np.array([0.05, H_star, 0.024])  # m
NOZZLE_DEPTH = 0.01  # m
A_star = H_star * NOZZLE_DEPTH
A_Astar = interp1d(X_points, H_points/H_star)  # ratio
x_pre_col = interp1d(H_points[:2]/H_star, X_points[:2])
x_post_col = interp1d(H_points[1:]/H_star, X_points[1:])

# Paramètres condensation
COND_LEN = 0.05  # m: longueur sur lequel tout le CO2 condense
dqdx = H_s / COND_LEN  # J/(kg*m)

def x_A_Astar(A_Astar, Ma):
	if Ma < 1:
		return x_pre_col(A_Astar)
	elif Ma > 1:
		return x_post_col(A_Astar)
	else:
		return X_star

# Discrétisation
n = int(1e5)
dx = (X_points[-1] - X_points[0]) / n
# EPS_MAX = 1e-15  # convergence condition for pressure residual
# MAX_ITER = 10
# RELAX = 1

def area_ratio_Ma(Ma):
	"""Ratio A/A* en fonction du nombre de Mach"""
	return 1 / Ma * ((1 + (GAMMA - 1) / 2 * Ma ** 2 ) / 
		( (GAMMA + 1) / 2 )) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))

def T_ratio_Ma(Ma):
	"""Ratio T_0/T en fonction du nombre de Mach

	Hypothèses:
	 - gaz parfait
	"""
	return 1 + (GAMMA - 1) / 2 * Ma ** 2

def p_ratio_Ma(Ma):
	"""Ratio p_0/p en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return T_ratio_Ma(Ma) ** (GAMMA / (GAMMA - 1))

def rho_ratio_Ma(Ma):
	"""Ratio rho_0/rho en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return T_ratio_Ma(Ma) ** (1 / (GAMMA - 1))

def rho_ratio_T(T, T0):
	"""Ratio rho_0/rho en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return (T0/T) ** (1 / (GAMMA - 1))

def speed_of_sound_Ma(Ma):
	"""Vitesse du son en fonction du nombre de mach
	
	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(GAMMA * R * T_0 / T_ratio_Ma(Ma))

def speed_of_sound_T(T):
	"""Vitesse du son en fonction de la temperature (K)
	
	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(GAMMA * R * T)

def x_Ma(Ma):
	"""Position en fonction du nombre de Mach
	"""
	return x_A_Astar(area_ratio_Ma(Ma), Ma)

def dM2_heat(M, dA_A, q, T, cp=C_p, gamma=GAMMA):
	return M ** 2 * (dA_A * -2 * (1 + (gamma - 1)/2 * M ** 2) / (1 - M ** 2)
		+ q / cp / T * (1 + gamma * M ** 2)/(1 - M ** 2))

def dT_heat(M, dA_A, q, T, cp=C_p, gamma=GAMMA):
	return T * (dA_A * (gamma - 1) * M ** 2 / (1 - M ** 2) +
			q / cp / T * (1 - gamma * M ** 2) / (1 - M ** 2))

def drho_heat(M, dA_A, q, T, rho, cp=C_p, gamma=GAMMA):
	return rho * (dA_A * M ** 2 / (1 - M**2) - q / cp / T / (1 - M ** 2))

func = lambda mach: area_ratio_Ma(mach) - H_points[-1]/H_star
Ma_max = fsolve(func, 3)
func1 = lambda mach: area_ratio_Ma(mach) - H_points[0]/H_star
Ma_min = fsolve(func1, 0.1)
Ma_lin = np.linspace(Ma_min, Ma_max, n)
X = np.array([x_Ma(Ma) for Ma in Ma_lin])

# Mass flow, choked flow
m_dot = P_0/p_ratio_Ma(1)*np.sqrt(GAMMA / (R * T_0/T_ratio_Ma(1))) * A_star

# Déterminer début de la condensation
Ma0_cond = 2  # nombre de Mach ou la condensation début.

# position et conditions au début de la condensation
x0_cond = x_Ma(Ma0_cond)
# Uncomment for  Rayleigh flow
# x0_cond = 0.17
func2 =lambda mach: area_ratio_Ma(mach) - A_Astar(x0_cond)
Ma0_cond = fsolve(func2, 3)
u0_cond = Ma0_cond * speed_of_sound_Ma(Ma0_cond)
p0_cond = P_0 / p_ratio_Ma(Ma0_cond) 
T0_cond = T_0 / T_ratio_Ma(Ma0_cond)
rho0_cond = p0_cond / (R * T0_cond)
print(x0_cond)
print("---------CONDITIONS INITIALES-----------")
print("Mach:")
print(Ma0_cond)
print("pression:")
print(p0_cond)
print("Température:")
print(T0_cond)
print("densité:")
print(rho0_cond)

# Discrétisation de la condensation
x_cond = np.arange(x0_cond, X_points[-1], dx)
A_cond = A_star * A_Astar(x_cond)

Ma_cond = np.zeros(x_cond.shape)
Ma_cond[0] = Ma0_cond

u_cond = np.zeros(x_cond.shape)
u_cond[0] = u0_cond

p_cond = np.zeros(x_cond.shape)
p_cond[0] = p0_cond

T_cond = np.zeros(x_cond.shape)
T_cond[0] = T0_cond

# T_stag_cond = np.zeros(x_cond.shape)
# T_stag_cond[0] = T_0

rho_cond = np.zeros(x_cond.shape)
rho_cond[0] = rho0_cond

# Solve écoulement
for i in range(len(x_cond)-1):
	if (x_cond[i]-x_cond[0]) < COND_LEN:
		q = dqdx * (x_cond[i+1]-x_cond[i])
	else:
		q = 0
	# T_stag_cond[i+1] = q/C_p + T_stag_cond[i]
	Ma_cond[i+1] = np.sqrt(Ma_cond[i]**2 + dM2_heat(Ma_cond[i], 
		(A_cond[i+1]-A_cond[i])/A_cond[i], q, T_cond[i]))
	T_cond[i+1] = T_cond[i] + dT_heat(Ma_cond[i], 
		(A_cond[i+1]-A_cond[i])/A_cond[i], q, T_cond[i])

	u_cond[i+1] = Ma_cond[i+1] * speed_of_sound_T(T_cond[i+1])
	rho_cond[i+1] = rho_cond[i] + drho_heat(Ma_cond[i], 
		(A_cond[i+1]-A_cond[i])/A_cond[i], q, T_cond[i], rho_cond[i])

# Calculate pressure:
p_cond = rho_cond*R*T_cond

print("---------CONDITIONS FINALES-------------")
print("Mach:")
print(Ma_cond[-1])
print("pression:")
print(p_cond[-1])
print("Température:")
print(T_cond[-1])
print("densité:")
print(rho_cond[-1])

# conservation checks
print("Masse (kg/s)")
print(rho_cond[0]*u_cond[0]*A_cond[0] - rho_cond[-1]*u_cond[-1]*A_cond[-1])
print("Energy (J/kg)")
print(C_p*T_cond[0] + u_cond[0]**2/2 - C_p*T_cond[-1] - u_cond[-1]**2/2)
print("Momentum (N)")
print(u_cond[0] * m_dot - u_cond[-1] * m_dot 
	+ sum(p_cond[1:] * (A_cond[1:]-A_cond[:-1])) 
	+ p_cond[0]*A_cond[0] 
	- p_cond[-1]*A_cond[-1])
print("entropy (J/kgK)")
print(C_v*np.log(T_cond[-1]/T_cond[0]) - R*np.log(rho_cond[-1]/rho_cond[0]))
	

#-----------------------------------------------------------------------------
#                               PLOTTING
#-----------------------------------------------------------------------------
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
ax0.plot(X, P_0/p_ratio_Ma(Ma_lin))
ax0.plot(x_cond, p_cond)
ax0.set_xlabel('x (m)')
ax0.set_ylabel('P (Pa)')
ax1.plot(X, T_0/T_ratio_Ma(Ma_lin))
ax1.plot(x_cond, T_cond)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('T (K)')
ax2.plot(X, Ma_lin)
ax2.plot(x_cond, Ma_cond)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('Mach')
# ax2.plot(X, rho_0/rho_ratio_Ma(Ma_lin))
# ax2.plot(x_cond, rho_cond)
# ax2.set_xlabel('x (m)')
# ax2.set_ylabel('rho (kg/m³)')
ax3.plot(X, speed_of_sound_Ma(Ma_lin) * Ma_lin)
ax3.plot(x_cond, u_cond)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('u (m/s)')
# plt.figure()
# plt.plot(x_cond, A_cond)
# plt.plot(X, A_star * A_Astar(X))
plt.show()



