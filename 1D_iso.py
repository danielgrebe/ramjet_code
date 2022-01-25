#!python

import numpy as	np
# import plotly.graph_objs as go
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Propriétés du gaz
CO2 = 0.1  # fraction massique du CO_2
MASSE_MOLAIRE = 14.5e-3  # kg/mol
IDEAL_GAS = 8.314  # J/(K*mol)
R = IDEAL_GAS / MASSE_MOLAIRE
gamma = 1.4
C_p = 1.006  # J/(kg*K)
C_v = C_p / gamma  # J/(kg*K)
H_s = CO2 * 591e3  # J/kg enthalpy of sublimation du mélange

# Conditions initials
P_0 = 1e5  # Pa
T_0 = 300  # K
u_0 = 0  # m/s

# Géométrie
H_star = 0.005  # m
X_star = 0.05
X_points = np.array([0, X_star, 0.30])  # m
H_points = np.array([0.05, H_star, 0.024])  # m
NOZZLE_DEPTH = 0.01  # m
A_star = H_star * NOZZLE_DEPTH
A_Astar = interp1d(X_points, H_points/H_star)  # ratio
x_pre_col = interp1d(H_points[:2]/H_star, X_points[:2])
x_post_col = interp1d(H_points[1:]/H_star, X_points[1:])
def x_A_Astar(A_Astar, Ma):
	if Ma < 1:
		return x_pre_col(A_Astar)
	elif Ma > 1:
		return x_post_col(A_Astar)
	else:
		return X_star

# Discrétisation
n = 1000
dx = (X_points[-1] - X_points[0]) / n

def area_ratio_Ma(Ma):
	"""Ratio A/A* en fonction du nombre de Mach"""
	return 1 / Ma * ((1 + (gamma - 1) / 2 * Ma ** 2 ) / 
		( (gamma + 1) / 2 )) ** ((gamma + 1) / (2 * (gamma - 1)))

def temp_ratio_Ma(Ma):
	"""Ratio T_0/T en fonction du nombre de Mach

	Hypothèses:
	 - gaz parfait
	"""
	return 1 + (gamma - 1) / 2 * Ma ** 2

def p_ratio_Ma(Ma):
	"""Ratio p_0/p en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return temp_ratio_Ma(Ma) ** (gamma / (gamma - 1))

def rho_ratio_Ma(Ma):
	"""Ratio rho_0/rho en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return temp_ratio_Ma(Ma) ** (1 / (gamma - 1))

def speed_of_sound_Ma(Ma):
	"""Vitesse du son en fonction du nombre de mach
	
	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(gamma * R * T_0 / temp_ratio_Ma(Ma))

def x_Ma(Ma):
	return x_A_Astar(area_ratio_Ma(Ma), Ma)

func = lambda mach: area_ratio_Ma(mach) - H_points[-1]/H_star
Ma_max = fsolve(func, 3)
func1 = lambda mach: area_ratio_Ma(mach) - H_points[0]/H_star
Ma_min = fsolve(func1, 0.1)
# print(Ma_max)
# print(Ma_min)
Ma_lin = np.linspace(Ma_min, Ma_max, n)
X = np.array([x_Ma(Ma) for Ma in Ma_lin])
# print(X)
# print(P_0/p_ratio_Ma(Ma_lin))

# Mass flow, choked flow
# m_dot = P_0/p_ratio_Ma(1)*np.sqrt(gamma / (R * T_0/temp_ratio_Ma(1))) * A_star


# Plotting
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=X,
# 						 y=P_0/p_ratio_Ma(Ma_lin),
# 						 mode='lines',
# 						 name='Pression (Pa)'))
# fig.show()
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
ax0.plot(X, P_0/p_ratio_Ma(Ma_lin))
ax0.set_xlabel('x (m)')
ax0.set_ylabel('P (Pa)')
ax1.plot(X, T_0/temp_ratio_Ma(Ma_lin))
ax1.set_xlabel('x (m)')
ax1.set_ylabel('T (K)')
ax2.plot(X, speed_of_sound_Ma(Ma_lin))
ax2.set_xlabel('x (m)')
ax2.set_ylabel('a (m/s)')
ax3.plot(X, Ma_lin)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('Mach')
plt.show()



