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
R = IDEAL_GAS / MASSE_MOLAIRE
gamma = 1.4
C_p = 1.006  # J/(kg*K)
C_v = C_p / gamma  # J/(kg*K)
H_s = CO2 * 591e3  # J/kg enthalpy of sublimation du mélange

# Conditions initials
P_0 = 1e5  # Pa
T_0 = 300  # K
u_0 = 0  # m/s
rho_0 = P_0 / (R * T_0)

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
n = int(1e4)
dx = (X_points[-1] - X_points[0]) / n
EPS_MAX = 1e-15  # convergence condition for pressure residual
MAX_ITER = 10
RELAX = 1

def area_ratio_Ma(Ma):
	"""Ratio A/A* en fonction du nombre de Mach"""
	return 1 / Ma * ((1 + (gamma - 1) / 2 * Ma ** 2 ) / 
		( (gamma + 1) / 2 )) ** ((gamma + 1) / (2 * (gamma - 1)))

def T_ratio_Ma(Ma):
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
	return T_ratio_Ma(Ma) ** (gamma / (gamma - 1))

def rho_ratio_Ma(Ma):
	"""Ratio rho_0/rho en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return T_ratio_Ma(Ma) ** (1 / (gamma - 1))

def speed_of_sound_Ma(Ma):
	"""Vitesse du son en fonction du nombre de mach
	
	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(gamma * R * T_0 / T_ratio_Ma(Ma))

def speed_of_sound_T(T):
	"""Vitesse du son en fonction de la temperature (K)
	
	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(gamma * R * T)

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
m_dot = P_0/p_ratio_Ma(1)*np.sqrt(gamma / (R * T_0/T_ratio_Ma(1))) * A_star
print(m_dot)

# Déterminer début de la condensation
Ma0_cond = 2  # nombre de Mach ou la condensation début.

# position et conditions au début de la condensation
x0_cond = x_Ma(Ma0_cond)
u0_cond = Ma0_cond * speed_of_sound_Ma(Ma0_cond)
p0_cond = P_0 / p_ratio_Ma(Ma0_cond) 
T0_cond = T_0 / T_ratio_Ma(Ma0_cond)
rho0_cond = p0_cond / (R * T0_cond)
print(x0_cond)
print(u0_cond)
print(p0_cond)
print(T0_cond)
print(rho0_cond)

# Discrétisation de la condensation
x_cond = np.arange(x0_cond, X_points[-1], dx)
A_cond = A_star * A_Astar(x_cond)

u_cond = np.zeros(x_cond.shape)
u_cond[0] = u0_cond

p_cond = np.zeros(x_cond.shape)
p_cond[0] = p0_cond

T_cond = np.zeros(x_cond.shape)
T_cond[0] = T0_cond

rho_cond = np.zeros(x_cond.shape)
rho_cond[0] = rho0_cond

# Solve écoulement
for i in range(len(x_cond)-1):
	dA = A_cond[i+1] - A_cond[i]
	# print(dA)
	q = 0
	# construct matrix
	A = np.array([[rho_cond[i]*u_cond[i], R*T_cond[i], rho_cond[i]*R],
				  [rho_cond[i]*A_cond[i], u_cond[i]*A_cond[i], 0], 
				  [u_cond[i], 0, C_p]])
	# b = np.array([-rho_cond[i] * u_cond[i] * dA, q, -p_cond[i]*dA, 0])
	b = np.array([0, -rho_cond[i] * u_cond[i] * dA, q])

	# solve matrix
	# (du, drho, dp, dT), null, null, null = np.linalg.lstsq(A, b) 
	du, drho, dT = np.linalg.solve(A, b) 
	# out = np.linalg.solve(A, b) 
	# print(out.shape)
	# print(du)

	u_cond[i+1] = du + u_cond[i]
	# p_cond[i+1] = dp + p_cond[i]
	T_cond[i+1] = dT + T_cond[i]
	rho_cond[i+1] = drho + rho_cond[i]

# Calculate pressure:
p_cond = rho_cond*R*T_cond

# conservation checks
print(rho_cond[0]*u_cond[0]*A_cond[0] - rho_cond[-1]*u_cond[-1]*A_cond[-1])
print(C_p*T_cond[0] + u_cond[0]**2/2 - C_p*T_cond[-1] - u_cond[-1]**2/2)
print(u_cond[0] * m_dot - u_cond[-1] * m_dot 
	+ sum(p_cond[1:] * (A_cond[1:]-A_cond[:-1])) 
	+ p_cond[0]*A_cond[0] 
	- p_cond[-1]*A_cond[-1])
print(C_v*np.log(T_cond[-1]/T_cond[0]) - R*np.log(rho_cond[-1]/rho_cond[0]))
#print(sum(A_cond[1:]-A_cond[:-1]) - A_cond[-1]+A_cond[0])
# print(rho_cond[0]*u_cond[0]*A_cond[0] - rho_cond[1]*u_cond[1]*A_cond[1])
# print(C_p*T_cond[0] + u_cond[0]**2/2 - C_p*T_cond[1] - u_cond[1]**2/2)
# print(u_cond[0] * m_dot - u_cond[1] * m_dot 
# 	+ p_cond[1] * (A_cond[1]-A_cond[0]) 
# 	+ p_cond[0]*A_cond[0] 
# 	- p_cond[1]*A_cond[1])
	


# Plotting
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=X,
# 						 y=P_0/p_ratio_Ma(Ma_lin),
# 						 mode='lines',
# 						 name='Pression (Pa)'))
# fig.show()
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
ax0.plot(X, P_0/p_ratio_Ma(Ma_lin))
ax0.plot(x_cond, p_cond)
ax0.set_xlabel('x (m)')
ax0.set_ylabel('P (Pa)')
ax1.plot(X, T_0/T_ratio_Ma(Ma_lin))
ax1.plot(x_cond, T_cond)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('T (K)')
ax2.plot(X, rho_0/rho_ratio_Ma(Ma_lin))
ax2.plot(x_cond, rho_cond)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('rho (kg/m³)')
ax3.plot(X, speed_of_sound_Ma(Ma_lin) * Ma_lin)
ax3.plot(x_cond, u_cond)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('u (m/s)')
# plt.figure()
# plt.plot(x_cond, A_cond)
# plt.plot(X, A_star * A_Astar(X))
plt.show()



